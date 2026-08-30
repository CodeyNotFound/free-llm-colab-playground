from app.backend.inference.hardware import parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv() -> None:
    assert parse_nvidia_smi_csv("Tesla T4, 15360, 14002\n") == ("Tesla T4", 15360, 14002)


def test_parse_nvidia_smi_csv_handles_bad_output() -> None:
    assert parse_nvidia_smi_csv("") is None
    assert parse_nvidia_smi_csv("name, nope, 123") is None
