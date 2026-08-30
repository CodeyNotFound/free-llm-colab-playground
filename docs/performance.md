# Performance and telemetry

The performance panel samples:

- GPU utilization and VRAM use from `nvidia-smi`
- System RAM from psutil
- UI-observed time to first streamed content and total request time
- Approximate prompt/generated tokens and generation rate (`characters / 4` heuristic)

Hardware readings are real snapshots. Token counts and rates are explicitly labeled estimates because
not every llama-server response exposes token-level timing in a stable OpenAI-compatible field. The
server log remains available for llama.cpp's own timing output.

Hybrid speed depends on GPU layer share, CPU memory bandwidth, model architecture, context, prompt
length, batch size, and thermal/runtime contention. This project intentionally does not promise a
tokens-per-second number before measurement.

For comparisons, keep the model, prompt, context, quantization, and settings fixed; run at least three
times after warmup and report median generation speed.

