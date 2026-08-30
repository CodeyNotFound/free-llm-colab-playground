"""Notebook-friendly progress reporting for long external builds."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_CMAKE_PROGRESS = re.compile(r"\[\s*(\d{1,3})%\]")
_STAGE_PREFIX = "::playground-stage::"


@dataclass(frozen=True)
class BuildUpdate:
    stage: str
    percent: int | None
    elapsed_seconds: float
    eta_seconds: float | None


def parse_build_line(line: str) -> tuple[str | None, int | None]:
    """Extract an installer stage marker or a CMake percentage."""
    stage = line.removeprefix(_STAGE_PREFIX).strip() if line.startswith(_STAGE_PREFIX) else None
    match = _CMAKE_PROGRESS.search(line)
    percent = min(100, int(match.group(1))) if match else None
    return stage, percent


def estimate_eta(elapsed_seconds: float, percent: int | None) -> float | None:
    """Estimate remaining time from observed progress, avoiding noisy early guesses."""
    if percent is None or percent < 2 or percent >= 100:
        return None
    return max(0.0, elapsed_seconds * (100 - percent) / percent)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculating…"
    total = max(0, round(seconds))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def _render(update: BuildUpdate) -> str:
    progress = f"{update.percent}%" if update.percent is not None else "working"
    return (
        f"⏳ {update.stage} — {progress} — elapsed {format_duration(update.elapsed_seconds)} "
        f"— ETA {format_duration(update.eta_seconds)}"
    )


def run_with_progress(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    heartbeat_seconds: float = 10.0,
) -> None:
    """Run a command while keeping one live Colab output line updated."""
    process = subprocess.Popen(
        command,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    output: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        for line in process.stdout:
            output.put(line.rstrip())
        output.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    start = time.monotonic()
    stage = "Preparing llama.cpp runtime"
    percent: int | None = None
    recent_output: deque[str] = deque(maxlen=30)
    display_handle = None
    try:
        from IPython.display import display

        display_handle = display("Starting…", display_id=True)
    except (ImportError, NameError):
        pass

    def show() -> None:
        elapsed = time.monotonic() - start
        message = _render(BuildUpdate(stage, percent, elapsed, estimate_eta(elapsed, percent)))
        if display_handle is not None:
            display_handle.update(message)
        else:
            print(message, flush=True)

    finished = False
    while not finished:
        try:
            line = output.get(timeout=heartbeat_seconds)
        except queue.Empty:
            show()
            continue
        if line is None:
            finished = True
            continue
        recent_output.append(line)
        next_stage, next_percent = parse_build_line(line)
        if next_stage:
            stage = next_stage
            percent = None
            show()
        elif next_percent is not None:
            percent = next_percent
            show()
        elif os.environ.get("PLAYGROUND_VERBOSE_BUILD") == "1":
            print(line, flush=True)

    return_code = process.wait()
    elapsed = time.monotonic() - start
    if return_code:
        print("\nBuild failed. Last output:\n" + "\n".join(recent_output), flush=True)
        raise subprocess.CalledProcessError(return_code, command)
    final = f"✅ llama.cpp runtime ready in {format_duration(elapsed)}"
    if display_handle is not None:
        display_handle.update(final)
    else:
        print(final, flush=True)
