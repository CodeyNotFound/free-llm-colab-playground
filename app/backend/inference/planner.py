from __future__ import annotations

import math
import re

from .models import FitStatus, HardwareProfile, MemoryEstimate, ModelProfile, OffloadPlan


def estimate_kv_cache_mb(
    context_size: int,
    *,
    parameter_count: str = "Unknown",
    kv_cache_type: str = "f16",
) -> int:
    """Conservative KV estimate when exact GGUF tensor metadata is not yet loaded."""
    match = re.search(r"([\d.]+)\s*[bB]", parameter_count)
    billions = float(match.group(1)) if match else 8.0
    # KV grows linearly with context; width roughly grows with model scale.
    base_at_4k = max(256.0, 140.0 * math.sqrt(max(billions, 1.0)))
    type_factor = {"f32": 2.0, "f16": 1.0, "q8_0": 0.55, "q4_0": 0.32}.get(kv_cache_type.lower(), 1.0)
    return math.ceil(base_at_4k * (context_size / 4096) * type_factor)


def plan_offload(
    hardware: HardwareProfile,
    model: ModelProfile,
    *,
    context_size: int = 8192,
    runtime_reserve_mb: int | None = None,
    kv_cache_type: str = "f16",
    gpu_layers_override: int | None = None,
) -> OffloadPlan:
    warnings: list[str] = []
    reserve = runtime_reserve_mb or max(1536, round(hardware.vram_mb * 0.10))
    kv_mb = estimate_kv_cache_mb(context_size, parameter_count=model.parameter_count, kv_cache_type=kv_cache_type)
    workspace_mb = max(512, round(model.size_mb * 0.03))
    free_vram = hardware.free_vram_mb if hardware.cuda_available else 0
    safe_gpu_budget = max(0, free_vram - reserve - kv_mb - workspace_mb)
    cpu_budget = max(0, hardware.available_ram_mb - 2048)
    total_layers = max(model.total_layers, 1)
    natural_layers = min(total_layers, math.floor(total_layers * safe_gpu_budget / max(model.size_mb, 1)))
    gpu_layers = natural_layers if gpu_layers_override is None else max(0, min(total_layers, gpu_layers_override))
    gpu_weights = round(model.size_mb * gpu_layers / total_layers)
    cpu_weights = max(0, model.size_mb - gpu_weights)
    gpu_memory = gpu_weights + reserve + kv_mb + workspace_mb if gpu_layers else 0
    total_required = model.size_mb + kv_mb + workspace_mb + 1024

    if total_required > free_vram + cpu_budget:
        status = FitStatus.INSUFFICIENT
        warnings.append("Estimated model and runtime memory exceed currently available VRAM + RAM.")
        recommendation = "Choose a smaller quantization/model or reduce context."
    elif gpu_layers >= total_layers:
        status = FitStatus.FULL_GPU
        recommendation = "Excellent fit: weights are estimated to remain on the GPU."
    elif gpu_layers / total_layers >= 0.35:
        status = FitStatus.HYBRID
        recommendation = "Reasonable hybrid fit: GPU handles as many layers as the safety budget permits."
    else:
        status = FitStatus.CPU_HEAVY
        warnings.append("Most weights will remain in system RAM; generation may be slow.")
        recommendation = "Usable only if slower CPU-heavy inference is acceptable."

    if context_size >= 32768:
        warnings.append("Large context substantially increases KV-cache memory and OOM risk.")
    if not hardware.cuda_available:
        warnings.append("No CUDA GPU was detected; the plan uses CPU inference.")
    if cpu_weights + 1536 > cpu_budget:
        warnings.append("The CPU-resident portion may leave too little RAM for the OS and prompt data.")

    estimate = MemoryEstimate(
        weights_mb=model.size_mb,
        runtime_reserve_mb=reserve,
        kv_cache_mb=kv_mb,
        workspace_mb=workspace_mb,
        total_required_mb=total_required,
        safe_gpu_weight_budget_mb=safe_gpu_budget,
        available_cpu_budget_mb=cpu_budget,
    )
    return OffloadPlan(
        gpu_layers=gpu_layers,
        total_layers=total_layers,
        estimated_gpu_memory_mb=gpu_memory,
        estimated_cpu_memory_mb=cpu_weights + 1024,
        context_size=context_size,
        status=status,
        estimate=estimate,
        warnings=warnings,
        recommendation=recommendation,
    )


def plan_markdown(plan: OffloadPlan) -> str:
    warnings = "\n".join(f"- ⚠️ {warning}" for warning in plan.warnings)
    return (
        f"## Estimate: {plan.status.value}\n\n"
        f"- **GPU layers:** {plan.gpu_layers} / {plan.total_layers}\n"
        f"- **Estimated GPU memory:** {plan.estimated_gpu_memory_mb / 1024:.1f} GiB\n"
        f"- **Estimated CPU memory:** {plan.estimated_cpu_memory_mb / 1024:.1f} GiB\n"
        f"- **Context:** {plan.context_size:,} tokens\n"
        f"- **Safe GPU weight budget:** {plan.estimate.safe_gpu_weight_budget_mb / 1024:.1f} GiB\n\n"
        f"{plan.recommendation}\n\n{warnings}\n\n"
        "*This is an approximate pre-load estimate. Architecture, KV layout, llama.cpp version, "
        "batch size, and prompt length affect actual memory and speed.*"
    )
