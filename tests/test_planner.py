from app.backend.inference.models import FitStatus, HardwareProfile, ModelProfile
from app.backend.inference.planner import estimate_kv_cache_mb, plan_offload


def hardware(vram: int = 15360, ram: int = 24576) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Tesla T4",
        vram_mb=vram,
        free_vram_mb=vram,
        system_ram_mb=ram,
        available_ram_mb=ram,
        cpu_cores=4,
        cuda_available=True,
    )


def model(size: int, layers: int = 40) -> ModelProfile:
    return ModelProfile("org/model", "model-Q4_K_M.gguf", size, "Q4_K_M", "8B", total_layers=layers)


def test_kv_cache_scales_with_context() -> None:
    assert estimate_kv_cache_mb(8192, parameter_count="8B") == 2 * estimate_kv_cache_mb(4096, parameter_count="8B")


def test_small_model_is_fully_gpu_resident() -> None:
    plan = plan_offload(hardware(), model(4800), context_size=4096)
    assert plan.status == FitStatus.FULL_GPU
    assert plan.gpu_layers == plan.total_layers
    assert plan.estimated_gpu_memory_mb < 15360


def test_medium_model_uses_hybrid_offload() -> None:
    plan = plan_offload(hardware(), model(18000, 64), context_size=8192)
    assert plan.status == FitStatus.HYBRID
    assert 0 < plan.gpu_layers < plan.total_layers
    assert plan.estimated_cpu_memory_mb > 1024


def test_impossible_model_is_rejected() -> None:
    plan = plan_offload(hardware(ram=8192), model(50000, 80), context_size=32768)
    assert plan.status == FitStatus.INSUFFICIENT
    assert plan.warnings


def test_gpu_override_is_clamped() -> None:
    plan = plan_offload(hardware(), model(4800), gpu_layers_override=999)
    assert plan.gpu_layers == plan.total_layers
