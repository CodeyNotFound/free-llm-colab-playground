# Application architecture

`frontend/ui.py` assembles the Gradio interface. It talks only to backend services:

- `backend/catalog.py`: Hugging Face search, metadata, and GGUF discovery
- `backend/downloads.py`: cached, allow-listed model downloads
- `backend/files.py`: bounded document extraction and untrusted-content wrapping
- `backend/tunnel.py`: random API keys and opt-in Quick Tunnel lifecycle
- `backend/inference/hardware.py`: hardware detection
- `backend/inference/planner.py`: explainable hybrid offload estimates
- `backend/inference/llamacpp.py`: authenticated llama-server process and streaming API client
- `backend/inference/telemetry.py`: actual psutil/nvidia-smi snapshots

`InferenceBackend` is deliberately small so another backend can implement the same lifecycle later.

