"""Gradient descent helper data for Chapter 2 exercises."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, TypedDict

from backend.services.ollama_client import NextTokenStats, OllamaClient


class ExampleRow(TypedDict):
    """Type for example row data."""

    context: str
    prompt: str
    correct_token: str
    p_correct: float
    top_token: str
    top_prob: float


class Example(TypedDict):
    """Type for example data."""

    id: str
    title: str
    description: str
    rows: list[ExampleRow]


@dataclass(frozen=True)
class GdConfig:
    token_source: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_s: float
    ollama_temperature: float
    ollama_top_logprobs: int


def _read_config() -> GdConfig:
    return GdConfig(
        token_source=os.getenv("GD_TOKEN_SOURCE", "static"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        ollama_timeout_s=float(os.getenv("OLLAMA_TIMEOUT_S", "30")),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        ollama_top_logprobs=int(os.getenv("OLLAMA_TOP_LOGPROBS", "20")),
    )


class GdService:
    def __init__(self) -> None:
        self._config = _read_config()
        self._client: OllamaClient | None = None
        self._examples: list[Example] = [
            {
                "id": "france-paris",
                "title": "The capital of France is Paris",
                "description": "Chapter 2 exercises 2.7-2.11 (single short phrase).",
                "rows": [
                    {
                        "context": "<begin_of_text>",
                        "prompt": "",
                        "correct_token": "The",
                        "p_correct": 0.026724,
                        "top_token": "Question",
                        "top_prob": 0.301258,
                    },
                    {
                        "context": "<begin_of_text> The",
                        "prompt": "The",
                        "correct_token": "capital",
                        "p_correct": 0.000169,
                        "top_token": "",
                        "top_prob": 0.0244,
                    },
                    {
                        "context": "<begin_of_text> The capital",
                        "prompt": "The capital",
                        "correct_token": "of",
                        "p_correct": 0.568659,
                        "top_token": "of",
                        "top_prob": 0.568659,
                    },
                    {
                        "context": "<begin_of_text> The capital of",
                        "prompt": "The capital of",
                        "correct_token": "France",
                        "p_correct": 0.012172,
                        "top_token": "the",
                        "top_prob": 0.204712,
                    },
                    {
                        "context": "<begin_of_text> The capital of France",
                        "prompt": "The capital of France",
                        "correct_token": "is",
                        "p_correct": 0.141121,
                        "top_token": ",",
                        "top_prob": 0.508131,
                    },
                    {
                        "context": "<begin_of_text> The capital of France is",
                        "prompt": "The capital of France is",
                        "correct_token": "Paris",
                        "p_correct": 0.391531,
                        "top_token": "Paris",
                        "top_prob": 0.391531,
                    },
                ],
            },
            {
                "id": "apple-day",
                "title": "An apple a day keeps the doctor away",
                "description": "Chapter 2 exercises 2.12-2.16 (short proverb).",
                "rows": [
                    {
                        "context": "<begin_of_text>",
                        "prompt": "",
                        "correct_token": "An",
                        "p_correct": 0.001692,
                        "top_token": "Question",
                        "top_prob": 0.301258,
                    },
                    {
                        "context": "<begin_of_text> An",
                        "prompt": "An",
                        "correct_token": "apple",
                        "p_correct": 0.000615,
                        "top_token": "",
                        "top_prob": 0.022775,
                    },
                    {
                        "context": "<begin_of_text> An apple",
                        "prompt": "An apple",
                        "correct_token": "a",
                        "p_correct": 0.649742,
                        "top_token": "a",
                        "top_prob": 0.649742,
                    },
                    {
                        "context": "<begin_of_text> An apple a",
                        "prompt": "An apple a",
                        "correct_token": "day",
                        "p_correct": 0.986583,
                        "top_token": "day",
                        "top_prob": 0.986583,
                    },
                    {
                        "context": "<begin_of_text> An apple a day",
                        "prompt": "An apple a day",
                        "correct_token": "keeps",
                        "p_correct": 0.483511,
                        "top_token": "keeps",
                        "top_prob": 0.483511,
                    },
                    {
                        "context": "<begin_of_text> An apple a day keeps",
                        "prompt": "An apple a day keeps",
                        "correct_token": "the",
                        "p_correct": 0.850819,
                        "top_token": "the",
                        "top_prob": 0.850819,
                    },
                    {
                        "context": "<begin_of_text> An apple a day keeps the",
                        "prompt": "An apple a day keeps the",
                        "correct_token": "doctor",
                        "p_correct": 0.738021,
                        "top_token": "doctor",
                        "top_prob": 0.738021,
                    },
                    {
                        "context": "<begin_of_text> An apple a day keeps the doctor",
                        "prompt": "An apple a day keeps the doctor",
                        "correct_token": "away",
                        "p_correct": 0.954824,
                        "top_token": "away",
                        "top_prob": 0.954824,
                    },
                ],
            },
            {
                "id": "wonderful-evening",
                "title": "I had a perfectly wonderful evening, but this wasn't it",
                "description": "Chapter 2 exercise 2.17 (mixed-confidence sentence).",
                "rows": [
                    {
                        "context": "<begin_of_text>",
                        "prompt": "",
                        "correct_token": "I",
                        "p_correct": math.exp(-4.9784),
                        "top_token": "Question",
                        "top_prob": 0.301258,
                    },
                    {
                        "context": "<begin_of_text> I",
                        "prompt": "I",
                        "correct_token": "had",
                        "p_correct": math.exp(-3.515),
                        "top_token": "have",
                        "top_prob": 0.093446,
                    },
                    {
                        "context": "<begin_of_text> I had",
                        "prompt": "I had",
                        "correct_token": "a",
                        "p_correct": math.exp(-2.0396),
                        "top_token": "a",
                        "top_prob": 0.130298,
                    },
                    {
                        "context": "<begin_of_text> I had a",
                        "prompt": "I had a",
                        "correct_token": "perfectly",
                        "p_correct": math.exp(-9.252),
                        "top_token": "few",
                        "top_prob": 0.143885,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly",
                        "prompt": "I had a perfectly",
                        "correct_token": "wonderful",
                        "p_correct": math.exp(-4.6985),
                        "top_token": "good",
                        "top_prob": 0.134728,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful",
                        "prompt": "I had a perfectly wonderful",
                        "correct_token": "evening",
                        "p_correct": math.exp(-2.2093),
                        "top_token": "life",
                        "top_prob": 0.158798,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening",
                        "prompt": "I had a perfectly wonderful evening",
                        "correct_token": ",",
                        "p_correct": math.exp(-2.2038),
                        "top_token": "with",
                        "top_prob": 0.086383,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening ,",
                        "prompt": "I had a perfectly wonderful evening,",
                        "correct_token": "but",
                        "p_correct": math.exp(-4.0289),
                        "top_token": "and",
                        "top_prob": 0.13039,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening , but",
                        "prompt": "I had a perfectly wonderful evening, but",
                        "correct_token": "this",
                        "p_correct": math.exp(-4.074),
                        "top_token": "I",
                        "top_prob": 0.327177,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening , but this",
                        "prompt": "I had a perfectly wonderful evening, but this",
                        "correct_token": "wasn't",
                        "p_correct": math.exp(-0.003),
                        "top_token": "wasn't",
                        "top_prob": 0.997032,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening , but this wasn't",
                        "prompt": "I had a perfectly wonderful evening, but this wasn't",
                        "correct_token": "it",
                        "p_correct": math.exp(-1.56),
                        "top_token": "it",
                        "top_prob": 0.210132,
                    },
                ],
            },
        ]

    def loss_curves(self, points: int = 50, eps: float = 1e-3) -> dict[str, Any]:
        if points < 2:
            raise ValueError("points must be >= 2")
        values: list[dict[str, float]] = []
        for idx in range(points):
            t = idx / (points - 1)
            p = eps + t * (1.0 - eps)
            values.append(
                {
                    "p": p,
                    "l1": 1.0 - p,
                    "ce": -math.log(p),
                }
            )
        return {"points": values}

    def token_loss_examples(
        self,
        source_override: str | None = None,
        example_id: str | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        source = source_override or self._config.token_source
        if source == "ollama":
            if self._client is None:
                self._client = OllamaClient(
                    self._config.ollama_base_url,
                    self._config.ollama_model,
                    timeout_s=self._config.ollama_timeout_s,
                    temperature=self._config.ollama_temperature,
                    top_logprobs=self._config.ollama_top_logprobs,
                )
            try:
                return self._token_loss_examples_ollama(example_id=example_id, prompt=prompt)
            except Exception as exc:  # pragma: no cover - fallback path
                payload = self._token_loss_examples_static()
                payload["source"] = "static"
                payload["warning"] = f"ollama_failed: {exc}"
                return payload
        return self._token_loss_examples_static(example_id=example_id)

    def _token_loss_examples_static(self, example_id: str | None = None) -> dict[str, Any]:
        examples = []
        for example in self._examples:
            if example_id and example["id"] != example_id:
                continue
            rows = []
            l1_losses = []
            ce_losses = []
            for row in example["rows"]:
                p_correct = float(row["p_correct"])
                l1 = 1.0 - p_correct
                ce = -math.log(max(p_correct, 1e-12))
                l1_losses.append(l1)
                ce_losses.append(ce)
                rows.append(
                    {
                        "context": row["context"],
                        "correct_token": row["correct_token"],
                        "p_correct": p_correct,
                        "top_token": row["top_token"],
                        "top_prob": row["top_prob"],
                        "l1_loss": l1,
                        "ce_loss": ce,
                    }
                )
            l1_avg = sum(l1_losses) / len(l1_losses)
            ce_avg = sum(ce_losses) / len(ce_losses)
            examples.append(
                {
                    "id": example["id"],
                    "title": example["title"],
                    "description": example["description"],
                    "average": {"l1": l1_avg, "ce": ce_avg},
                    "rows": rows,
                }
            )
        return {"source": "static", "examples": examples}

    def _token_loss_examples_ollama(
        self,
        example_id: str | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        examples: list[dict[str, Any]] = []
        warning: str | None = None
        if example_id == "custom":
            return self._token_loss_custom_ollama(prompt or "")
        selected_ids = ["france-paris", "apple-day"]
        if example_id:
            selected_ids = [example_id]
        selected = [ex for ex in self._examples if ex["id"] in selected_ids]
        for example in selected:
            rows = []
            l1_losses = []
            ce_losses = []
            missing = 0
            for row in example["rows"]:
                prompt_val: str = row["prompt"]
                correct_token: str = row["correct_token"]
                prompt_for_stats = prompt_val if prompt_val else " "
                assert self._client is not None
                stats = self._client.next_token_stats(prompt_for_stats)
                top_token = stats.top_token
                top_prob = stats.top_prob
                p_correct = self._match_prob(stats, correct_token)
                if p_correct == 0.0:
                    missing += 1
                l1 = 1.0 - p_correct
                ce = -math.log(max(p_correct, 1e-12))
                l1_losses.append(l1)
                ce_losses.append(ce)
                rows.append(
                    {
                        "context": row["context"],
                        "correct_token": correct_token,
                        "p_correct": p_correct,
                        "top_token": top_token,
                        "top_prob": top_prob,
                        "l1_loss": l1,
                        "ce_loss": ce,
                    }
                )
            warnings = []
            if missing:
                warnings.append(f"correct_token_missing_in_top_logprobs: {missing}")
            if warnings:
                warning = "; ".join(warnings)
            l1_avg = sum(l1_losses) / len(l1_losses)
            ce_avg = sum(ce_losses) / len(ce_losses)
            examples.append(
                {
                    "id": example["id"],
                    "title": example["title"],
                    "description": example["description"],
                    "average": {"l1": l1_avg, "ce": ce_avg},
                    "rows": rows,
                }
            )
        payload = {"source": "ollama", "examples": examples}
        if warning:
            payload["warning"] = warning
        if example_id is None:
            existing_warning = payload.get("warning")
            prefix = f"{existing_warning}; " if existing_warning else ""
            payload["warning"] = prefix + "ollama_default_example: france-paris"
        return payload

    def _token_loss_custom_ollama(self, prompt: str) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            return {
                "source": "ollama",
                "examples": [],
                "warning": "custom_prompt_empty",
            }
        tokens = prompt.split()
        rows = []
        l1_losses = []
        ce_losses = []
        missing = 0
        for idx, token in enumerate(tokens):
            prefix = " ".join(tokens[:idx]).strip()
            prompt_for_stats = prefix if prefix else " "
            assert self._client is not None
            stats = self._client.next_token_stats(prompt_for_stats)
            p_correct = self._match_prob(stats, token)
            if p_correct == 0.0:
                missing += 1
            l1 = 1.0 - p_correct
            ce = -math.log(max(p_correct, 1e-12))
            l1_losses.append(l1)
            ce_losses.append(ce)
            context = "<begin_of_text>" if not prefix else f"<begin_of_text> {prefix}"
            rows.append(
                {
                    "context": context,
                    "correct_token": token,
                    "p_correct": p_correct,
                    "top_token": stats.top_token,
                    "top_prob": stats.top_prob,
                    "l1_loss": l1,
                    "ce_loss": ce,
                }
            )
        l1_avg = sum(l1_losses) / len(l1_losses)
        ce_avg = sum(ce_losses) / len(ce_losses)
        payload = {
            "source": "ollama",
            "examples": [
                {
                    "id": "custom",
                    "title": prompt,
                    "description": "Custom prompt",
                    "average": {"l1": l1_avg, "ce": ce_avg},
                    "rows": rows,
                }
            ],
        }
        warnings = []
        if missing:
            warnings.append(f"correct_token_missing_in_top_logprobs: {missing}")
        warnings.append("tokenization_simple_split")
        payload["warning"] = "; ".join(warnings)
        return payload

    def ollama_status(self) -> dict[str, Any]:
        if self._client is None:
            self._client = OllamaClient(
                self._config.ollama_base_url,
                self._config.ollama_model,
                timeout_s=self._config.ollama_timeout_s,
                temperature=self._config.ollama_temperature,
                top_logprobs=self._config.ollama_top_logprobs,
            )
        result: dict[str, Any] = self._client.status()
        return result

    def next_token_logprobs(self, prompt: str, limit: int | None = None) -> dict[str, Any]:
        if self._client is None:
            self._client = OllamaClient(
                self._config.ollama_base_url,
                self._config.ollama_model,
                timeout_s=self._config.ollama_timeout_s,
                temperature=self._config.ollama_temperature,
                top_logprobs=self._config.ollama_top_logprobs,
            )
        assert self._client is not None
        stats = self._client.next_token_stats(prompt)
        probs = stats.probs or {}
        if not probs:
            probs = {stats.top_token: stats.top_prob} if stats.top_token else {}
            warning = "top_logprobs_unavailable"
        else:
            warning = None
        items = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        limit_val = limit if limit is not None else self._config.ollama_top_logprobs
        tokens = [
            {
                "token": token,
                "prob": prob,
                "logprob": math.log(max(prob, 1e-12)),
                "rank": idx + 1,
            }
            for idx, (token, prob) in enumerate(items[: max(1, limit_val)])
        ]
        payload = {"prompt": prompt, "tokens": tokens}
        if warning:
            payload["warning"] = warning
        return payload

    def _match_prob(self, stats: NextTokenStats, token: str) -> float:
        candidates = {token, token.lstrip(), token.rstrip()}
        if not token.startswith(" "):
            candidates.add(f" {token}")
        probs = stats.probs
        return max((probs.get(candidate, 0.0) for candidate in candidates), default=0.0)
