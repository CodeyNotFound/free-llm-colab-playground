from app.backend.tunnel import generate_api_key


def test_api_keys_are_random_and_session_scoped() -> None:
    first = generate_api_key()
    second = generate_api_key()
    assert first.startswith("colab-")
    assert first != second
    assert len(first) > 30
