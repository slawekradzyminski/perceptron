from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NextTokenStats:
    top_token: str
    top_prob: float
    probs: dict[str, float]
    latency_ms: float | None = None


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 30.0,
        temperature: float = 0.0,
        top_logprobs: int = 20,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._top_logprobs = top_logprobs
        self._last_latency_ms: float | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate_logprobs(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "raw": True,
            "top_logprobs": self._top_logprobs,
            "options": {
                "temperature": self._temperature,
                "num_predict": 1,
            },
        }
        return self._post_json("/api/generate", payload)

    def next_token_stats(self, prompt: str) -> NextTokenStats:
        start = time.perf_counter()
        data = self.generate_logprobs(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = latency_ms
        probs = self._extract_top_probs(data)
        top_token, top_prob = self._pick_top_token(probs, data.get("response", ""))
        return NextTokenStats(top_token=top_token, top_prob=top_prob, probs=probs, latency_ms=latency_ms)

    def status(self) -> dict[str, Any]:
        try:
            version = self._get_json("/api/version")
            tags = self._get_json("/api/tags")
            models = [model.get("name") for model in tags.get("models", []) if isinstance(model, dict)]
            return {
                "ok": True,
                "model": self._model,
                "base_url": self._base_url,
                "version": version.get("version"),
                "models": models,
                "last_latency_ms": self._last_latency_ms,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            return {
                "ok": False,
                "model": self._model,
                "base_url": self._base_url,
                "error": str(exc),
                "last_latency_ms": self._last_latency_ms,
            }

    def _pick_top_token(self, probs: dict[str, float], fallback: str) -> tuple[str, float]:
        if probs:
            token, prob = max(probs.items(), key=lambda item: item[1])
            return token, prob
        return fallback, 0.0

    def _extract_top_probs(self, data: dict[str, Any]) -> dict[str, float]:
        logprobs = data.get("logprobs") or {}
        if isinstance(logprobs, list):
            first = logprobs[0] if logprobs else None
            if isinstance(first, dict):
                top_logprobs = first.get("top_logprobs")
                if isinstance(top_logprobs, list):
                    probs: dict[str, float] = {}
                    for item in top_logprobs:
                        if not isinstance(item, dict):
                            continue
                        token = item.get("token")
                        logprob = item.get("logprob")
                        if isinstance(token, str) and isinstance(logprob, (int, float)):
                            probs[token] = math.exp(logprob)
                    if probs:
                        return probs
                token = first.get("token")
                logprob = first.get("logprob")
                if isinstance(token, str) and isinstance(logprob, (int, float)):
                    return {token: math.exp(logprob)}
            return {}
        top_logprobs = logprobs.get("top_logprobs") if isinstance(logprobs, dict) else None
        top_map: dict[str, Any] = {}
        if isinstance(top_logprobs, list):
            if top_logprobs:
                top_map = top_logprobs[0] or {}
        elif isinstance(top_logprobs, dict):
            top_map = top_logprobs
        result: dict[str, float] = {}
        for token, logprob in top_map.items():
            if isinstance(logprob, (int, float)):
                result[token] = math.exp(logprob)
        return result

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                body = resp.read()
            result: dict[str, Any] = json.loads(body.decode("utf-8"))
            return result
        except urllib.error.HTTPError as e:
            # Read error body if available for better error messages
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama API error {e.code}: {e.reason}. "
                f"Is Ollama running at {self._base_url}? "
                f"Details: {error_body[:200] if error_body else 'No details'}"
            ) from e

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout_s) as resp:
                body = resp.read()
            result: dict[str, Any] = json.loads(body.decode("utf-8"))
            return result
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama API error {e.code}: {e.reason}. "
                f"Is Ollama running at {self._base_url}?"
            ) from e
