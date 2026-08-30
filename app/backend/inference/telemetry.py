from __future__ import annotations

import csv
import io
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil


@dataclass(frozen=True)
class TelemetrySnapshot:
    timestamp: float
    ram_used_mb: int
    ram_total_mb: int
    gpu_utilization_percent: float | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot() -> TelemetrySnapshot:
    ram = psutil.virtual_memory()
    gpu_util: float | None = None
    vram_used: int | None = None
    vram_total: int | None = None
    if shutil.which("nvidia-smi"):
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
            row = next(csv.reader(io.StringIO(result.stdout)))
            gpu_util, vram_used, vram_total = float(row[0]), int(row[1]), int(row[2])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError, StopIteration):
            pass
    return TelemetrySnapshot(
        timestamp=time.time(),
        ram_used_mb=round(ram.used / 1024 / 1024),
        ram_total_mb=round(ram.total / 1024 / 1024),
        gpu_utilization_percent=gpu_util,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
    )
