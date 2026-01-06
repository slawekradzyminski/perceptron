"""Softmax utilities and cross-entropy helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable


def softmax(logits: Iterable[float]) -> list[float]:
    """Stable softmax over a 1D iterable."""
    values = list(logits)
    if not values:
        raise ValueError("logits must be non-empty")
    max_logit = max(values)
    exps = [math.exp(val - max_logit) for val in values]
    denom = sum(exps)
    if denom == 0.0:
        raise ValueError("softmax denominator is zero")
    return [val / denom for val in exps]


def cross_entropy_from_logits(logits: Iterable[float], target_index: int, eps: float = 1e-12) -> float:
    """Cross-entropy loss for a single target index given logits."""
    probs = softmax(logits)
    if target_index < 0 or target_index >= len(probs):
        raise IndexError("target_index out of range")
    p = min(max(probs[target_index], eps), 1.0 - eps)
    return -math.log(p)


def grad_logits_softmax_ce(logits: Iterable[float], target_index: int) -> list[float]:
    """Gradient of softmax+cross-entropy with respect to logits."""
    probs = softmax(logits)
    if target_index < 0 or target_index >= len(probs):
        raise IndexError("target_index out of range")
    grads = list(probs)
    grads[target_index] -= 1.0
    return grads
