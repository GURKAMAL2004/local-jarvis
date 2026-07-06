from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deskbot.config import load_config
from deskbot.webui import licensing
from deskbot.webui.generate_license import generate_code
from deskbot.webui.licensing import PUBLIC_KEY_B64, verify_license_code


def _fresh_keypair() -> tuple[str, str]:
    """Generates a brand-new keypair at test time — never the real
    production private key, which must never appear in this repo. Returns
    (private_key_b64, public_key_b64)."""
    private_key = Ed25519PrivateKey.generate()
    private_b64 = base64.b64encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode()
    public_b64 = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    ).decode()
    return private_b64, public_b64


@pytest.fixture
def signing_key(monkeypatch) -> str:
    """Swaps licensing.PUBLIC_KEY_B64 for a fresh, throwaway test keypair for
    the duration of the test, and returns the matching private key — so
    tests can fully exercise sign+verify without ever touching (or risking
    exposure of) the real production private key."""
    private_b64, public_b64 = _fresh_keypair()
    monkeypatch.setattr(licensing, "PUBLIC_KEY_B64", public_b64)
    return private_b64


def test_generate_and_verify_round_trip(signing_key):
    code = generate_code("buyer@example.com", signing_key)
    assert verify_license_code("buyer@example.com", code) is True


def test_verify_rejects_wrong_email(signing_key):
    code = generate_code("buyer@example.com", signing_key)
    assert verify_license_code("someone-else@example.com", code) is False


def test_verify_rejects_tampered_code(signing_key):
    code = generate_code("buyer@example.com", signing_key)
    tampered = code[:-4] + ("AAAA" if not code.endswith("AAAA") else "BBBB")
    assert verify_license_code("buyer@example.com", tampered) is False


def test_verify_rejects_garbage_code():
    assert verify_license_code("buyer@example.com", "not-a-real-code") is False


def test_verify_rejects_empty_inputs():
    assert verify_license_code("", "") is False
    assert verify_license_code("buyer@example.com", "") is False
    assert verify_license_code("", "somecode") is False


def test_verify_is_case_and_whitespace_insensitive_on_email(signing_key):
    code = generate_code("buyer@example.com", signing_key)
    assert verify_license_code("  Buyer@Example.com  ", code) is True


def test_verify_rejects_code_signed_by_a_different_key(signing_key):
    """A code signed by some other keypair must never verify against
    whatever public key is actually configured — proves codes can't be
    forged with an unrelated key."""
    other_private_b64, _other_public_b64 = _fresh_keypair()
    code_from_wrong_key = generate_code("buyer@example.com", other_private_b64)
    assert verify_license_code("buyer@example.com", code_from_wrong_key) is False


def test_public_key_constant_is_valid_base64_32_bytes():
    raw = base64.b64decode(PUBLIC_KEY_B64)
    assert len(raw) == 32  # Ed25519 public keys are always 32 bytes


@pytest.fixture
def client(monkeypatch, tmp_path):
    from deskbot import paths
    from deskbot.webui import server

    monkeypatch.setattr(paths, "HOME_DIR", tmp_path)
    monkeypatch.setattr(server, "PREMIUM_PATH", tmp_path / "premium.json")

    from fastapi.testclient import TestClient

    app = server.create_app(load_config())
    return TestClient(app)


def test_index_page_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "deskbot" in r.text.lower()


def test_personas_endpoint(client):
    r = client.get("/api/personas")
    assert r.status_code == 200
    data = r.json()
    assert "personas" in data
    assert isinstance(data["personas"], list)


def test_routines_endpoint_empty_by_default(client):
    r = client.get("/api/routines")
    assert r.status_code == 200
    assert r.json() == {"routines": []}


def test_premium_wallet_endpoint_returns_configured_address(client):
    r = client.get("/api/premium/wallet")
    assert r.status_code == 200
    data = r.json()
    assert data["address"] == "0x2Fe6cB50E5a3515a16266bcBB8A8a5f5936E1829"
    assert data["email"] == "gkmander7@gmail.com"


def test_premium_qrcode_endpoint_returns_png(client):
    r = client.get("/api/premium/qrcode.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_premium_status_defaults_to_locked(client):
    r = client.get("/api/premium/status")
    assert r.status_code == 200
    assert r.json() == {"unlocked": False}


def test_premium_verify_and_status_round_trip(client, signing_key):
    code = generate_code("buyer@example.com", signing_key)
    r = client.post("/api/premium/verify", json={"email": "buyer@example.com", "code": code})
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    r = client.get("/api/premium/status")
    assert r.json() == {"unlocked": True, "email": "buyer@example.com"}


def test_premium_verify_rejects_invalid_code(client):
    r = client.post("/api/premium/verify", json={"email": "buyer@example.com", "code": "garbage"})
    assert r.status_code == 400


def test_chess_new_and_move(client, monkeypatch):
    from deskbot.llm import OllamaClient

    # The model's reply move is chosen via choose_model_move, which is
    # grounded in the real legal-move list (see chess_game.py) — a canned
    # reply here just avoids a real Ollama round trip in a unit test.
    monkeypatch.setattr(OllamaClient, "chat", lambda self, model, messages, temperature=0.4, tools=None: "Nc6")

    r = client.post("/api/chess/new", json={"color": "white"})
    assert r.status_code == 200
    data = r.json()
    assert data["turn"] == "white"
    assert len(data["legal_moves"]) == 20
    session_id = data["session_id"]

    r = client.post("/api/chess/move", json={"session_id": session_id, "move": "e2e4"})
    assert r.status_code == 200
    data = r.json()
    assert data["moves_played"] == ["e4", "Nc6"]


def test_chess_move_rejects_illegal_move(client):
    r = client.post("/api/chess/new", json={"color": "white"})
    session_id = r.json()["session_id"]

    r = client.post("/api/chess/move", json={"session_id": session_id, "move": "e2e5"})
    assert r.status_code == 400


def test_chess_move_rejects_unknown_session(client):
    r = client.post("/api/chess/move", json={"session_id": "does-not-exist", "move": "e2e4"})
    assert r.status_code == 404


def test_chat_rejects_unknown_persona(client):
    r = client.post("/api/chat", json={"persona": "not-a-real-persona", "message": "hi"})
    assert r.status_code == 404
