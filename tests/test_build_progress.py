from app.backend.build_progress import estimate_eta, format_duration, parse_build_line


def test_parse_stage_marker() -> None:
    assert parse_build_line("::playground-stage::Compiling llama-server") == ("Compiling llama-server", None)


def test_parse_cmake_progress() -> None:
    assert parse_build_line("[ 42%] Building CUDA object foo.cu.o") == (None, 42)


def test_eta_and_duration() -> None:
    assert estimate_eta(120, 25) == 360
    assert estimate_eta(10, 1) is None
    assert format_duration(3661) == "1h 01m"
