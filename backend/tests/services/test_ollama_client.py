"""Tests for OllamaClient."""

import json
import math
from typing import Any

import pytest

from backend.services.ollama_client import OllamaClient


class DummyResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_generate_logprobs_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=0):
        captured["req"] = req
        captured["timeout"] = timeout
        return DummyResponse(b"{\"logprobs\": [], \"response\": \"ok\"}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient("http://localhost:11434", "llama3.2")
    result = client.generate_logprobs("Hello")

    assert result["response"] == "ok"
    req = captured["req"]
    payload = json.loads(req.data.decode("utf-8"))
    assert req.full_url == "http://localhost:11434/api/generate"
    assert payload["model"] == "llama3.2"
    assert payload["prompt"] == "Hello"
    assert payload["stream"] is False
    assert payload["logprobs"] is True
    assert payload["raw"] is True
    assert payload["top_logprobs"] == 20
    assert payload["options"] == {"temperature": 0.0, "num_predict": 1}


def test_extract_top_probs_from_list() -> None:
    client = OllamaClient("http://localhost:11434", "llama3.2")
    data = {
        "logprobs": [
            {
                "top_logprobs": [
                    {"token": " hello", "logprob": -0.1},
                    {"token": " world", "logprob": -2.0},
                ]
            }
        ]
    }
    probs = client._extract_top_probs(data)
    assert probs[" hello"] == pytest.approx(math.exp(-0.1))
    assert probs[" world"] == pytest.approx(math.exp(-2.0))


def test_extract_top_probs_from_dict() -> None:
    client = OllamaClient("http://localhost:11434", "llama3.2")
    data = {"logprobs": {"top_logprobs": [{" a": -0.2, "b": -1.0}]}}
    probs = client._extract_top_probs(data)
    assert probs[" a"] == pytest.approx(math.exp(-0.2))
    assert probs["b"] == pytest.approx(math.exp(-1.0))


def test_next_token_stats_fallback_to_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://localhost:11434", "llama3.2")

    def fake_generate_logprobs(prompt: str) -> dict[str, Any]:
        return {"logprobs": {}, "response": "Z"}

    monkeypatch.setattr(client, "generate_logprobs", fake_generate_logprobs)
    stats = client.next_token_stats("prompt")
    assert stats.top_token == "Z"
    assert stats.top_prob == 0.0
    assert stats.latency_ms is not None
    assert client._last_latency_ms is not None


def test_status_aggregates_models(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://localhost:11434", "llama3.2")

    def fake_get_json(path: str) -> dict[str, Any]:
        if path == "/api/version":
            return {"version": "1.2.3"}
        if path == "/api/tags":
            return {"models": [{"name": "m1"}, {"name": "m2"}, "bad", {"name": None}]}
        return {}

    monkeypatch.setattr(client, "_get_json", fake_get_json)
    status = client.status()
    assert status["ok"] is True
    assert status["version"] == "1.2.3"
    assert status["models"] == ["m1", "m2", None]


def test_pick_top_token_uses_probs() -> None:
    client = OllamaClient("http://localhost:11434", "llama3.2")
    token, prob = client._pick_top_token({"a": 0.1, "b": 0.2}, "fallback")
    assert token == "b"
    assert prob == 0.2


def test_extract_top_probs_uses_token_logprob() -> None:
    client = OllamaClient("http://localhost:11434", "llama3.2")
    data = {"logprobs": [{"token": "x", "logprob": -0.5}]}
    probs = client._extract_top_probs(data)
    assert probs == {"x": pytest.approx(math.exp(-0.5))}
