from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FitStatus(str, Enum):
    FULL_GPU = "Fully GPU-resident"
    HYBRID = "Hybrid GPU + CPU"
    CPU_HEAVY = "Heavy CPU offload"
    INSUFFICIENT = "Insufficient memory"


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str = "No CUDA GPU detected"
    vram_mb: int = 0
    free_vram_mb: int = 0
    system_ram_mb: int = 0
    available_ram_mb: int = 0
    cpu_cores: int = 1
    cpu_architecture: str = "Unknown"
    cuda_available: bool = False
    python_version: str = "Unknown"
    operating_system: str = "Unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    repo_id: str
    filename: str
    size_mb: int
    quantization: str = "Unknown"
    parameter_count: str = "Unknown"
    architecture: str = "Unknown"
    total_layers: int = 0
    context_length: int | None = None
    model_family: str = "Unknown"
    is_moe: bool | None = None
    active_parameters: str = "Unknown"


@dataclass(frozen=True)
class MemoryEstimate:
    weights_mb: int
    runtime_reserve_mb: int
    kv_cache_mb: int
    workspace_mb: int
    total_required_mb: int
    safe_gpu_weight_budget_mb: int
    available_cpu_budget_mb: int


@dataclass(frozen=True)
class OffloadPlan:
    gpu_layers: int
    total_layers: int
    estimated_gpu_memory_mb: int
    estimated_cpu_memory_mb: int
    context_size: int
    status: FitStatus
    estimate: MemoryEstimate
    warnings: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class GGUFFile:
    repo_id: str
    filename: str
    size_bytes: int
    quantization: str
    split_files: tuple[str, ...] = ()

    @property
    def size_mb(self) -> int:
        return round(self.size_bytes / 1024 / 1024)

    @property
    def display_size(self) -> str:
        gib = self.size_bytes / 1024**3
        return f"{gib:.2f} GiB"


@dataclass(frozen=True)
class ModelMetadata:
    repo_id: str
    name: str
    author: str
    downloads: int | None = None
    likes: int | None = None
    license: str = "Unknown"
    architecture: str = "Unknown"
    parameter_count: str = "Unknown"
    active_parameters: str = "Unknown"
    context_length: int | None = None
    model_family: str = "Unknown"
    is_moe: bool | None = None
    description: str = "Unknown"
    tags: tuple[str, ...] = ()
