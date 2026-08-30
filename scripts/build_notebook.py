"""Rebuild the checked-in Colab notebook from small, reviewable cells."""

from __future__ import annotations

import json
from pathlib import Path

TARGET = Path("colab/Free_LLM_Colab_Playground.ipynb")


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
# 🧠 Free LLM Colab Playground

### Run your own open-weight LLM on a free Colab GPU

This guided notebook searches Hugging Face for **GGUF** models, estimates a safe GPU/CPU split,
downloads the selected quantization, starts an authenticated `llama-server`, and opens a modern
chat UI. Your GPU does as much work as practical; model layers that do not fit can remain in RAM.

**Designed for:** free Colab GPUs such as T4, while detecting L4/A100/CPU-only runtimes dynamically.

**Before you begin**

- In Colab choose **Runtime → Change runtime type → T4 GPU** (availability varies).
- Runtimes, downloaded files, Gradio links, and API tunnels are temporary.
- Large contexts need substantial KV-cache memory; start with 4K or 8K.
- Never share the temporary API URL and key. Anyone with both can spend your runtime resources.
- Uploaded documents are untrusted reference material and are never executed.

Documentation: [Getting started](../docs/getting-started.md) ·
[Model selection](../docs/model-selection.md) · [Security](../docs/security.md)
"""
    ),
    markdown(
        """
## 1 · Install the playground

If you opened this notebook from GitHub, set `REPOSITORY_URL` to that repository once. The cell
clones the source into the temporary Colab runtime and installs only the small Python dependency set.
Re-running it is safe.
"""
    ),
    code(
        """
from pathlib import Path
import os
import subprocess
import sys

REPOSITORY_URL = "https://github.com/CodeyNotFound/free-llm-colab-playground.git"
PROJECT_DIR = Path("/content/free-llm-colab-playground")

if not PROJECT_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(PROJECT_DIR)], check=True)

os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print(f"Playground installed from {PROJECT_DIR}")
"""
    ),
    markdown(
        """
## 2 · Detect your hardware

Nothing here assumes a T4. The planner uses currently free VRAM (not just total VRAM), available
system RAM, CPU cores, architecture, CUDA availability, Python, and OS information.
"""
    ),
    code(
        """
from app.backend.inference.hardware import detect_hardware, hardware_summary

hardware = detect_hardware()
print(hardware_summary(hardware))

if not hardware.cuda_available:
    print("\\n⚠️ No CUDA GPU detected. Enable a GPU runtime or expect CPU-only inference.")
"""
    ),
    markdown(
        """
## 3 · Prepare llama.cpp with CUDA

The fast path downloads a checksum-verified T4 runtime built by this project's GitHub Actions. If it
is unavailable or incompatible, the notebook safely falls back to compiling official `llama.cpp`.
The live status shows the current stage, elapsed time, build percentage, and approximate ETA.
"""
    ),
    code(
        """
import os
from pathlib import Path

from app.backend.build_progress import run_with_progress

env = {**os.environ, "BUILD_JOBS": "2"}
run_with_progress(["bash", "scripts/install_llama_cpp.sh"], env=env)
server_path = Path(Path(".runtime/llama_server_path.txt").read_text().strip()).resolve()
os.environ["LLAMA_SERVER_PATH"] = str(server_path)
print(f"Using {server_path}")
"""
    ),
    markdown(
        """
## 4 · Launch the guided application

The application provides the remaining flow in one place:

1. Search Hugging Face and inspect real model metadata.
2. Discover actual GGUF files and split-file sizes.
3. Estimate VRAM reserve, KV cache, RAM use, and GPU layers.
4. Download without re-downloading cached files.
5. Start the model, stream chat, attach documents, and inspect telemetry/logs.
6. Optionally create an authenticated Cloudflare Quick Tunnel for OpenAI-compatible clients.

Click the public Gradio link printed by the cell. The API tunnel is **not** created automatically.
"""
    ),
    code(
        """
import gradio as gr

import secrets

from app.frontend.ui import CSS, create_app

demo = create_app()
ui_password = secrets.token_urlsafe(12)
print(f"Private UI login: colab / {ui_password}")
demo.queue(default_concurrency_limit=2).launch(
    share=True,
    auth=("colab", ui_password),
    auth_message="Private Colab session. Use the credentials printed in this notebook output.",
    debug=False,
    show_error=False,
    prevent_thread_lock=True,
    theme=gr.themes.Soft(),
    css=CSS,
    max_file_size="8mb",
    enable_monitoring=False,
)
"""
    ),
    markdown(
        """
## 5 · Use the model

In the browser UI, follow **Model → Start & Nerd Mode → Chat**. Simple defaults reserve headroom
instead of filling every MiB of VRAM. Advanced users can override GPU layers, batch/microbatch,
threads, KV type, flash attention, and context size.

To connect SillyTavern, OpenCode, Python, curl, or another OpenAI-compatible client, open **API**,
read the warning, create the temporary tunnel, and copy the Base URL, key, and model ID. Use
`GET /v1/models` and `POST /v1/chat/completions`; streaming uses SSE through llama-server.
"""
    ),
    markdown(
        """
## 6 · Stop and disconnect

Use **Stop tunnel** when external access is no longer needed, then **Stop model**. Colab may also
terminate the runtime at any time. A runtime reset removes downloaded models, generated credentials,
local logs, and URLs. The generated API key is never committed or printed to server logs.

### Common pitfalls

- **CUDA out of memory:** reduce context, lower GPU layers, or choose Q4 instead of Q5/Q6.
- **System RAM exhausted:** choose a smaller GGUF; model weights must still be stored for MoE models.
- **Slow hybrid generation:** expected when many layers remain on CPU.
- **Tunnel disappeared:** Quick Tunnel URLs last only for the process/runtime.

### Optional exercise

Compare 4K and 16K context estimates for the same Q4 file before downloading. Inspect how the KV
reserve changes, then choose the safer plan. No model data is downloaded by the estimator.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Free LLM Colab Playground", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Built {TARGET} with {len(cells)} cells")
