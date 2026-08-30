from pathlib import Path

from app.backend.inference.llamacpp import LlamaCppBackend


def test_command_contains_hybrid_and_auth_settings(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")
    backend = LlamaCppBackend(binary, port=9999)
    command = backend.build_command(
        model_path="model.gguf",
        gpu_layers=33,
        context_size=8192,
        api_key="secret",
        model_alias="my-model",
        threads=4,
    )
    assert command[command.index("--n-gpu-layers") + 1] == "33"
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert command[command.index("--api-key") + 1] == "secret"
    assert "--metrics" in command
    assert "--flash-attn" in command
    assert command[command.index("--flash-attn") + 1] == "on"
