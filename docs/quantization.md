# Quantization

GGUF quantization stores model weights with fewer bits to reduce memory and often improve throughput.
Names are llama.cpp conventions, not universal quality scores.

| Family | Beginner interpretation |
|---|---|
| Q4 / Q4_K_M | Small, practical, usually the best T4 starting point |
| Q5 / Q5_K_M | More memory, potentially better quality |
| Q6_K | Higher fidelity and memory use |
| Q8_0 | Large; often unnecessary on a T4 |
| IQ variants | Importance-weighted formats with model/runtime-specific trade-offs |

Exact quality depends on the source model and quantization method. File size—not only the filename—is
used for fitting. Split GGUF repositories are treated as one logical model and all pieces are summed
and downloaded.

The automatic recommendation considers actual size, current VRAM/RAM, selected context, runtime
reserve, KV estimate, and safety margin. `Insufficient memory` means the estimated total exceeds the
current combined budget; it is a warning, not mathematical proof.

