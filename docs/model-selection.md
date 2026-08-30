# Model selection

Start with an instruction/chat model that has a llama.cpp-compatible chat template and a public GGUF
conversion. The UI searches Hugging Face through its API and shows unavailable metadata as `Unknown`.

## Approximate T4 territory

- **Excellent:** 1B, 3B, 7B, 8B, 12B, and 14B, depending on quantization and context.
- **Hybrid:** 20B, 24B, 30B, and 32B, depending heavily on system RAM and CPU.
- **Very large:** 40B+ becomes increasingly RAM- and CPU-limited.
- **70B:** generally impractical on free T4 runtimes unless RAM is unusually high and very poor
  performance is acceptable.

These are not hard cutoffs. Compare the actual GGUF byte size with currently free VRAM/RAM and the
context-dependent reserve.

## Dense versus MoE

A dense model uses most weights for each token. A Mixture-of-Experts model routes each token through
only some experts, so active parameters can reduce compute. It does **not** eliminate storage: all
experts' weights must still fit across VRAM and RAM. The browser labels Dense/MoE only when metadata
supports it and never invents active-parameter counts.

## Choosing a repository

Prefer an author with a clear model card, license, source-model link, chat template, multiple common
quantizations, and intact split files. Review the model's license yourself; discovery is not legal
advice. If the original repo has no GGUF, the app offers candidate conversion repositories rather
than downloading unrelated files.

