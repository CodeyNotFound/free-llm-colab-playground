from pathlib import Path
from unittest.mock import Mock

import pytest

from app.backend.inference.models import GGUFFile, HardwareProfile, ModelMetadata
from app.frontend.ui import RuntimeState, _default_gguf, create_app


def gguf(name="model-Q4_K_M.gguf", quant="Q4_K_M", size=2 * 1024**3):
    return GGUFFile("test/model", name, size, quant)


@pytest.fixture
def ui():
    rt = RuntimeState(
        catalog=Mock(),
        backend=Mock(),
        tunnel=Mock(),
        hardware=HardwareProfile(
            gpu_name="Test GPU",
            vram_mb=16384,
            free_vram_mb=15000,
            available_ram_mb=12000,
            system_ram_mb=16000,
            cuda_available=True,
        ),
    )
    rt.ggufs = [gguf()]
    rt.metadata = ModelMetadata("test/model", "Model", "test", parameter_count="3B")
    demo = create_app(rt)
    functions = {item.fn.__name__: item.fn for item in demo.fns.values() if item.fn}
    return demo, rt, functions


def test_default_prefers_balanced_file():
    assert _default_gguf([gguf("huge.gguf", "F16"), gguf()]) == "model-Q4_K_M.gguf"
    assert _default_gguf([]) is None


def test_guided_ui_has_safe_defaults(ui):
    demo, _, _ = ui
    components = demo.config["components"]
    tabs = [item["props"]["label"] for item in components if item["type"] == "tabitem"]
    assert tabs == ["Setup", "Chat", "Monitor", "Connect apps", "Help"]
    by_label = {item["props"].get("label"): item["props"] for item in components}
    assert by_label["Conversation memory (context)"]["value"] == 4096
    assert by_label["Advanced loading settings · optional"]["open"] is False
    buttons = {item["props"]["value"]: item["props"] for item in components if item["type"] == "button"}
    assert buttons["Start model"]["interactive"] is False
    assert buttons["Download selected file"]["interactive"] is False


def test_unknown_file_never_silently_selects_first(ui):
    _, rt, functions = ui
    functions["estimate"]("missing.gguf", 4096, "f16", -1)
    assert rt.selected is None
    assert rt.plan is None


def test_download_uses_current_context_and_enables_start(ui, monkeypatch, tmp_path):
    _, rt, functions = ui
    path = tmp_path / rt.ggufs[0].filename
    path.touch()
    monkeypatch.setattr("app.frontend.ui.download_gguf", lambda *args, **kwargs: path)
    status, result = functions["download"](rt.ggufs[0].filename, 4096, "q8_0", -1)
    assert "Download complete" in status
    assert result == str(path)
    assert rt.plan.context_size == 4096
    assert functions["setup_state"]()[1].interactive is True
    rt.ggufs.append(gguf("other.gguf"))
    functions["estimate"]("other.gguf", 4096, "f16", -1)
    assert functions["setup_state"]()[1].interactive is False
    status = functions["start_model"](str(path), 4096, 1, 512, 128, True, "f16", -1)[0]
    assert "selection changed" in status
    rt.backend.start.assert_not_called()


def test_insufficient_memory_blocks_download(ui, monkeypatch):
    _, rt, functions = ui
    rt.ggufs = [gguf(size=100 * 1024**3)]
    downloader = Mock()
    monkeypatch.setattr("app.frontend.ui.download_gguf", downloader)
    status, path = functions["download"](rt.ggufs[0].filename, 4096, "f16", -1)
    assert "exceed available memory" in status
    assert path == ""
    downloader.assert_not_called()


def test_failed_discovery_clears_previous_selection(ui):
    _, rt, functions = ui
    functions["estimate"](rt.ggufs[0].filename, 4096, "f16", -1)
    rt.model_path = Path("old.gguf")
    rt.downloaded_selection = (rt.selected.repo_id, rt.selected.filename)
    rt.catalog.details.side_effect = RuntimeError("Unavailable")
    functions["discover"]("test/other")
    assert rt.selected is None
    assert rt.model_path is None
    assert rt.plan is None
    assert rt.downloaded_selection is None


def test_successful_start_shows_next_action(ui):
    _, rt, functions = ui
    functions["estimate"](rt.ggufs[0].filename, 4096, "f16", -1)
    rt.downloaded_selection = (rt.selected.repo_id, rt.selected.filename)
    rt.model_path = Path("test.gguf")
    status = functions["start_model"]("test.gguf", 4096, 1, 512, 128, True, "f16", -1)[0]
    assert "Ready to chat" in status
    assert "Chat" in status
    rt.backend.start.assert_called_once()


def test_chat_without_model_explains_setup(ui):
    _, rt, functions = ui
    rt.backend.health.return_value = False
    result = list(functions["stream_message"]("Hello", [], "", 0.7, 0.95, 512, ""))
    assert "Open Setup" in result[0][0][-1]["content"]
    rt.backend.stream_chat.assert_not_called()


def test_empty_stop_field_and_gradio_text_blocks(ui):
    _, rt, functions = ui
    rt.backend.stream_chat.return_value = iter(["Reply"])
    history = [{"role": "user", "content": [{"type": "text", "text": "Earlier question"}]}]
    result = list(functions["stream_message"]("Follow-up", history, "", 0.7, 0.95, 512, None))
    assert result[-1][0][-1]["content"] == "Reply"
    messages = rt.backend.stream_chat.call_args.args[0]
    assert messages[0]["content"] == "Earlier question"
    assert rt.backend.stream_chat.call_args.kwargs["stop"] is None
