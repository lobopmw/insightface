from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_round_trip() -> None:
    hashed = hash_password("senha-segura")

    assert hashed != "senha-segura"
    assert verify_password("senha-segura", hashed)
    assert not verify_password("senha-errada", hashed)


def test_access_token_round_trip() -> None:
    token = create_access_token("12345678901", {"role": "admin"})

    payload = decode_access_token(token)

    assert payload is not None
    assert payload["sub"] == "12345678901"
    assert payload["role"] == "admin"
