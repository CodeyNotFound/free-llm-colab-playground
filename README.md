# Free LLM Colab Playground

> **Run your own LLM on a free Colab GPU.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CodeyNotFound/free-llm-colab-playground/blob/main/colab/Free_LLM_Colab_Playground.ipynb)
[![Tests](https://github.com/CodeyNotFound/free-llm-colab-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/CodeyNotFound/free-llm-colab-playground/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Free LLM Colab Playground is a guided model browser, hardware planner, downloader, chat app, and
authenticated OpenAI-compatible API for GGUF models. It is built around `llama.cpp` and designed for
temporary free Google Colab GPUs such as the Tesla T4.

## Why?

Free Colab gives people temporary GPU access, but running a local model normally requires knowing
about GGUF, quantization, CUDA builds, GPU layers, CPU offload, KV cache, HTTP servers, and tunnels.
This project turns those decisions into a guided flow:

```text
Search → inspect real GGUF files → estimate fit → download → start → chat → connect apps
```

It does **not** reject a model merely because its weights exceed VRAM. The planner reserves VRAM for
CUDA/runtime overhead, workspace, and KV cache, then puts as many model layers on the GPU as the safe
budget allows. Remaining layers stay in system RAM for hybrid inference.

## Features

- Guided Colab notebook and modern Gradio UI
- Dynamic GPU, VRAM, RAM, CPU, CUDA, Python, and OS detection
- Hugging Face API model search, metadata, GGUF discovery, split-file support, and caching
- Approximate, explainable VRAM/RAM/KV-cache planning with automatic GPU-layer selection
- Beginner defaults plus Nerd Mode controls for context, layers, batches, threads, KV type, and flash attention
- CUDA-enabled `llama-server` build and supervised subprocess lifecycle
- Streaming multi-turn chat, system prompt, sampling controls, stop sequences, clear, and regenerate
- TXT/Markdown/JSON/CSV/source/PDF attachments with size limits and prompt-injection boundaries
- Explicit model reasoning display only when the server returns a reasoning field
- `nvidia-smi`/psutil telemetry, per-request estimates, and redacted server logs
- Authenticated `GET /v1/models` and `POST /v1/chat/completions`
- Random login protection for the temporary shared Gradio UI
- Optional free Cloudflare Quick Tunnel, random session key, API test, and copyable examples
- Connection guides for SillyTavern, OpenCode, Python, and curl

## Quick start

1. Open the notebook with the badge and enable a GPU runtime.
2. Set the repository URL in the setup cell and run the notebook top to bottom.
3. Open the Gradio link, search for a model, and inspect its GGUF options.
4. Start with the recommended Q4 quantization and 4K–8K context.
5. Download, start the model, and chat.
6. If an external client is needed, open **API**, create a temporary tunnel, and copy the URL, key,
   and model ID.

See [Getting started](docs/getting-started.md) for the full walkthrough.

## Interface preview

```text
┌───────────────────────────────────────────────────────────┐
│ 🧠 Free LLM Colab Playground       🟢 model · Hybrid · 8K │
├───────────────────────────────────────────────────────────┤
│ Model │ Chat │ Files │ Start & Nerd │ API │ Connect       │
├───────────────────────────────────────────────────────────┤
│ Search Hugging Face → choose actual GGUF → estimate fit   │
│                                                           │
│ User   Explain this uploaded code                          │
│ Model  …streaming response…                               │
└───────────────────────────────────────────────────────────┘
```

Screenshots will be added after validation on a public Colab GPU; this repository does not include
fabricated UI captures.

## Hardware expectations

These are ranges, not rules. Architecture, quantization, context, KV type, free RAM, and current VRAM
use all matter.

| Territory | Approximate model sizes | Typical experience on a T4 |
|---|---:|---|
| Excellent | 1B–14B | Often mostly or fully GPU at Q4/Q5 |
| Hybrid | 20B–32B | GPU + CPU; speed depends heavily on offload and CPU |
| Very large | 40B+ | Increasingly RAM-bound and slow |
| Generally impractical | 70B | Usually exceeds free-tier RAM or is unacceptably slow |

MoE active parameters can reduce compute per token, but **all model weights still need storage**.
Read [Hardware and fitting](docs/hardware.md) and [Model selection](docs/model-selection.md).

## Supported clients

- Built-in browser chat
- SillyTavern
- OpenCode
- Python OpenAI SDK
- curl and other OpenAI-compatible clients

The local server supports streaming SSE. Cloudflare currently documents that **Quick Tunnels do not
support SSE**, so external clients should use non-streaming mode through this free tunnel; built-in
chat still streams directly from localhost. See [API](docs/api.md).

## Security

The model binds to localhost and requires a freshly generated `colab-…` bearer key. Shared Gradio
sessions also receive a random login printed only in notebook output. Public API access is opt-in, the
key is redacted from launch logs, uploads are never executed, filenames are sanitized, and extracted
document text is separated as untrusted material. Quick Tunnel is a development tool, not production
hosting. Stop it immediately when finished. Read [Security](docs/security.md).

## Local development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
ruff check .
free-llm-playground --host 127.0.0.1
```

Install/build `llama-server` separately or run `scripts/install_llama_cpp.sh` on Linux with CMake.
Set `LLAMA_SERVER_PATH` when it is not on `PATH`.

## Limitations

- Pre-load fitting is intentionally approximate; exact tensor/KV allocation is only known at load time.
- Free Colab hardware, RAM, timeouts, and session duration are not guaranteed.
- Model cards are inconsistent, so missing metadata is shown as `Unknown`.
- Only text-oriented GGUF models are supported in V1; no multimodal/RAG/multiple-model serving.
- PDF extraction handles embedded text, not OCR.
- Quick Tunnel URLs are temporary, have no SLA, and do not support SSE.
- A real Colab T4 is required to validate CUDA load speed and OOM fallback behavior end to end.

## Roadmap

- Read exact GGUF block/KV metadata before planning and add a single bounded safer-load retry
- Add chunked retrieval for large documents instead of truncation
- Add model presets and reproducible T4/L4/A100 benchmark records
- Add multimodal backends and a backend plugin interface
- Add conversation import/export and persistent Drive caching

## Documentation

[Getting started](docs/getting-started.md) · [Model selection](docs/model-selection.md) ·
[Quantization](docs/quantization.md) · [Hardware](docs/hardware.md) ·
[Performance](docs/performance.md) · [API](docs/api.md) ·
[SillyTavern](docs/sillytavern.md) · [OpenCode](docs/opencode.md) ·
[Troubleshooting](docs/troubleshooting.md) · [Security](docs/security.md)

The implementation follows the official [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Hugging Face Hub downloads guide](https://huggingface.co/docs/huggingface_hub/en/guides/download),
[Cloudflare Quick Tunnel documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/),
and [Google Colab FAQ](https://research.google.com/colaboratory/faq.html).

## License

Apache License 2.0. See [LICENSE](LICENSE).
