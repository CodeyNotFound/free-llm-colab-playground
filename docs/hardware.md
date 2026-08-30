# Hardware and fitting

The planner uses `HardwareProfile`, `ModelProfile`, `MemoryEstimate`, and `OffloadPlan` dataclasses.
It detects the first NVIDIA GPU with `nvidia-smi` and reads host memory/CPU data with psutil.

## Budget

```text
safe GPU weight budget
  = currently free VRAM
  - runtime/CUDA reserve
  - estimated KV cache
  - workspace reserve
```

GPU layers are proportional to the weight budget and clamped to the estimated total layer count.
Weights not assigned to GPU remain in RAM. At least 2 GiB of available RAM is reserved for the OS;
additional runtime overhead is included separately.

The pre-download layer count and KV cache are conservative heuristics when a model card does not
publish exact tensor metadata. Therefore the display says **estimated**, not exact. Actual llama.cpp
allocation is authoritative.

## Context

KV memory grows approximately linearly with context. The UI offers 4K, 8K, 16K, 32K, and 64K, but a
value appearing in the menu is not a promise the current model/runtime can support it. Values above
the model's trained context are not useful without model-specific scaling support.

Quantized KV (`q8_0` or `q4_0`) can save memory but may affect quality or compatibility. `f16` is the
simple default. Larger batches improve prompt throughput but consume more workspace.

