"""Metrics helpers for perceptron training."""

from __future__ import annotations

from typing import Iterable, Sequence


def count_mistakes(results: Iterable[dict]) -> int:
    return sum(1 for r in results if r.get("mistake"))


def accuracy(samples: Iterable[dict], predict_fn) -> float:
    samples_list = list(samples)
    if not samples_list:
        return 0.0
    correct = 0
    for s in samples_list:
        pred = predict_fn(s["x"])
        if pred == s["y"]:
            correct += 1
    return correct / len(samples_list)


def margin(y: int, score: float) -> float:
    return y * score


def margins(samples: Iterable[dict], score_fn) -> list[float]:
    return [margin(s["y"], score_fn(s["x"])) for s in samples]
