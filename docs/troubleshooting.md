# Troubleshooting

## No GPU or CUDA unavailable

Choose a GPU runtime and rerun hardware detection. If `nvidia-smi` works but `nvcc` is unavailable,
the installer will build CPU-only llama.cpp; inspect the build cell output.

## CUDA out of memory

Try, in order: reduce context to 4096, lower GPU layers, use Q4 instead of Q5/Q6/Q8, reduce batch and
microbatch, or choose a smaller model. Restarting the runtime can recover VRAM held by stale processes.

## System RAM exhausted

CPU offload still requires RAM for weight data. Choose a smaller GGUF/model or reduce context. MoE
active parameters do not reduce total weight storage.

## Server exits while loading

The beginner message gives likely remediation. Open Nerd Mode → Logs for llama.cpp output. Verify the
selected file exists, all split files downloaded, and `LLAMA_SERVER_PATH` points to the build.

## Model answers strangely

Choose an instruction-tuned model with a compatible embedded chat template. Verify you did not exceed
its trained context. Uploaded files are reference content; ask an explicit question about them.

## Tunnel fails or returns 429

Quick Tunnel is best-effort and rate-limited. Stop it, retry once, or use localhost built-in chat. It
is not production hosting. External streaming is unsupported by Quick Tunnel.

## Gated/private Hugging Face repository

Accept the model terms on Hugging Face and authenticate the runtime with a scoped read token. Never
paste that token into this repository or commit notebook outputs.

