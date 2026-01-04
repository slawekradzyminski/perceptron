"""Gradient descent helper data for Chapter 2 exercises."""

from __future__ import annotations

import math
from typing import Any, Dict, List


class GdService:
    def __init__(self) -> None:
        self._examples = [
            {
                "id": "france-paris",
                "title": "The capital of France is Paris",
                "description": "Chapter 2 exercises 2.7-2.11 (single short phrase).",
                "rows": [
                    {
                        "context": "<begin_of_text>",
                        "correct_token": "The",
                        "p_correct": 0.026724,
                        "top_token": "Question",
                        "top_prob": 0.301258,
                    },
                    {
                        "context": "<begin_of_text> The",
                        "correct_token": "capital",
                        "p_correct": 0.000169,
                        "top_token": "",
                        "top_prob": 0.0244,
                    },
                    {
                        "context": "<begin_of_text> The capital",
                        "correct_token": "of",
                        "p_correct": 0.568659,
                        "top_token": "of",
                        "top_prob": 0.568659,
                    },
                    {
                        "context": "<begin_of_text> The capital of",
                        "correct_token": "France",
                        "p_correct": 0.012172,
                        "top_token": "the",
                        "top_prob": 0.204712,
                    },
                    {
                        "context": "<begin_of_text> The capital of France",
                        "correct_token": "is",
                        "p_correct": 0.141121,
                        "top_token": ",",
                        "top_prob": 0.508131,
                    },
                    {
                        "context": "<begin_of_text> The capital of France is",
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
                        "correct_token": "An",
                        "p_correct": 0.001692,
                        "top_token": "Question",
                        "top_prob": 0.301258,
                    },
                    {
                        "context": "<begin_of_text> An",
                        "correct_token": "apple",
                        "p_correct": 0.000615,
                        "top_token": "",
                        "top_prob": 0.022775,
                    },
                    {
                        "context": "<begin_of_text> An apple",
                        "correct_token": "a",
                        "p_correct": 0.649742,
                        "top_token": "a",
                        "top_prob": 0.649742,
                    },
                    {
                        "context": "<begin_of_text> An apple a",
                        "correct_token": "day",
                        "p_correct": 0.986583,
                        "top_token": "day",
                        "top_prob": 0.986583,
                    },
                    {
                        "context": "<begin_of_text> An apple a day",
                        "correct_token": "keeps",
                        "p_correct": 0.483511,
                        "top_token": "keeps",
                        "top_prob": 0.483511,
                    },
                    {
                        "context": "<begin_of_text> An apple a day keeps",
                        "correct_token": "the",
                        "p_correct": 0.850819,
                        "top_token": "the",
                        "top_prob": 0.850819,
                    },
                    {
                        "context": "<begin_of_text> An apple a day keeps the",
                        "correct_token": "doctor",
                        "p_correct": 0.738021,
                        "top_token": "doctor",
                        "top_prob": 0.738021,
                    },
                    {
                        "context": "<begin_of_text> An apple a day keeps the doctor",
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
                        "correct_token": "I",
                        "p_correct": math.exp(-4.9784),
                        "top_token": "Question",
                        "top_prob": 0.301258,
                    },
                    {
                        "context": "<begin_of_text> I",
                        "correct_token": "had",
                        "p_correct": math.exp(-3.515),
                        "top_token": "have",
                        "top_prob": 0.093446,
                    },
                    {
                        "context": "<begin_of_text> I had",
                        "correct_token": "a",
                        "p_correct": math.exp(-2.0396),
                        "top_token": "a",
                        "top_prob": 0.130298,
                    },
                    {
                        "context": "<begin_of_text> I had a",
                        "correct_token": "perfectly",
                        "p_correct": math.exp(-9.252),
                        "top_token": "few",
                        "top_prob": 0.143885,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly",
                        "correct_token": "wonderful",
                        "p_correct": math.exp(-4.6985),
                        "top_token": "good",
                        "top_prob": 0.134728,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful",
                        "correct_token": "evening",
                        "p_correct": math.exp(-2.2093),
                        "top_token": "life",
                        "top_prob": 0.158798,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening",
                        "correct_token": ",",
                        "p_correct": math.exp(-2.2038),
                        "top_token": "with",
                        "top_prob": 0.086383,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening ,",
                        "correct_token": "but",
                        "p_correct": math.exp(-4.0289),
                        "top_token": "and",
                        "top_prob": 0.13039,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening , but",
                        "correct_token": "this",
                        "p_correct": math.exp(-4.074),
                        "top_token": "I",
                        "top_prob": 0.327177,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening , but this",
                        "correct_token": "wasn't",
                        "p_correct": math.exp(-0.003),
                        "top_token": "wasn't",
                        "top_prob": 0.997032,
                    },
                    {
                        "context": "<begin_of_text> I had a perfectly wonderful evening , but this wasn't",
                        "correct_token": "it",
                        "p_correct": math.exp(-1.56),
                        "top_token": "it",
                        "top_prob": 0.210132,
                    },
                ],
            },
        ]

    def loss_curves(self, points: int = 50, eps: float = 1e-3) -> Dict[str, Any]:
        if points < 2:
            raise ValueError("points must be >= 2")
        values: List[Dict[str, float]] = []
        for idx in range(points):
            t = idx / (points - 1)
            p = eps + t * (1.0 - eps)
            values.append({
                "p": p,
                "l1": 1.0 - p,
                "ce": -math.log(p),
            })
        return {"points": values}

    def token_loss_examples(self) -> Dict[str, Any]:
        examples = []
        for example in self._examples:
            rows = []
            l1_losses = []
            ce_losses = []
            for row in example["rows"]:
                p_correct = float(row["p_correct"])
                l1 = 1.0 - p_correct
                ce = -math.log(max(p_correct, 1e-12))
                l1_losses.append(l1)
                ce_losses.append(ce)
                rows.append({
                    **row,
                    "l1_loss": l1,
                    "ce_loss": ce,
                })
            l1_avg = sum(l1_losses) / len(l1_losses)
            ce_avg = sum(ce_losses) / len(ce_losses)
            examples.append({
                "id": example["id"],
                "title": example["title"],
                "description": example["description"],
                "average": {"l1": l1_avg, "ce": ce_avg},
                "rows": rows,
            })
        return {"examples": examples}
