from __future__ import annotations

import json
import platform
import shutil
import subprocess

import psutil

from .models import HardwareProfile


def parse_nvidia_smi_csv(output: str) -> tuple[str, int, int] | None:
    """Parse `name,memory.total,memory.free` from the first nvidia-smi row."""
    row = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not row:
        return None
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 3:
        return None
    try:
        return parts[0], int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        return None


def detect_gpu() -> tuple[str, int, int, bool]:
    if not shutil.which("nvidia-smi"):
        return "No CUDA GPU detected", 0, 0, False
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        parsed = parse_nvidia_smi_csv(result.stdout)
        if parsed:
            return *parsed, True
    except (OSError, subprocess.SubprocessError):
        pass
    return "CUDA GPU unavailable", 0, 0, False


def detect_hardware() -> HardwareProfile:
    gpu_name, vram_mb, free_vram_mb, cuda_available = detect_gpu()
    memory = psutil.virtual_memory()
    return HardwareProfile(
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        free_vram_mb=free_vram_mb,
        system_ram_mb=round(memory.total / 1024 / 1024),
        available_ram_mb=round(memory.available / 1024 / 1024),
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_architecture=platform.machine() or "Unknown",
        cuda_available=cuda_available,
        python_version=platform.python_version(),
        operating_system=f"{platform.system()} {platform.release()}",
    )


def hardware_summary(profile: HardwareProfile) -> str:
    cuda = "Available" if profile.cuda_available else "Unavailable"
    return (
        "## Your hardware\n\n"
        f"- **GPU:** {profile.gpu_name}\n"
        f"- **VRAM:** {profile.free_vram_mb / 1024:.1f} GiB free / "
        f"{profile.vram_mb / 1024:.1f} GiB total\n"
        f"- **System RAM:** {profile.available_ram_mb / 1024:.1f} GiB available / "
        f"{profile.system_ram_mb / 1024:.1f} GiB total\n"
        f"- **CPU:** {profile.cpu_cores} physical cores ({profile.cpu_architecture})\n"
        f"- **CUDA:** {cuda}\n"
        f"- **Python / OS:** {profile.python_version} / {profile.operating_system}"
    )


def hardware_json(profile: HardwareProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2)
