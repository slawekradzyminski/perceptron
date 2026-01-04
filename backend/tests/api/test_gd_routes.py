from fastapi.testclient import TestClient

from backend.api.deps import gd_service

from backend.api_app import app


def test_gd_loss_curves():
    client = TestClient(app)
    resp = client.get("/gd/loss-curves")
    assert resp.status_code == 200
    data = resp.json()
    assert "points" in data
    assert len(data["points"]) == 50
    assert {"p", "l1", "ce"}.issubset(data["points"][0].keys())


def test_gd_token_losses():
    client = TestClient(app)
    resp = client.get("/gd/token-losses")
    assert resp.status_code == 200
    data = resp.json()
    assert "examples" in data
    assert len(data["examples"]) >= 3
    example = data["examples"][0]
    assert "rows" in example
    assert "average" in example
    row = example["rows"][0]
    assert {"context", "correct_token", "p_correct", "l1_loss", "ce_loss"}.issubset(row.keys())


def test_gd_next_token_logprobs(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(
        gd_service,
        "next_token_logprobs",
        lambda prompt, limit=None: {
            "prompt": prompt,
            "tokens": [{"token": " Paris", "prob": 0.7, "logprob": -0.356, "rank": 1}],
        },
    )
    resp = client.get("/gd/next-token-logprobs?prompt=The%20capital")
    assert resp.status_code == 200
    data = resp.json()
    assert data["prompt"] == "The capital"
    assert data["tokens"][0]["token"] == " Paris"


def test_gd_ollama_status(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(gd_service, "ollama_status", lambda: {"ok": True, "model": "llama3.2:1b"})
    resp = client.get("/gd/ollama-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "ok" in data
