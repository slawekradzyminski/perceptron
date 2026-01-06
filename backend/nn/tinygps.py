"""TinyGPS softmax classifier helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from backend.nn.losses import cross_entropy_prob
from backend.nn.softmax import softmax


@dataclass
class TinyGpsForward:
    logits: list[float]
    probs: list[float]
    loss: float


@dataclass
class TinyGpsGrads1D:
    dL_dm: list[float]
    dL_db: list[float]


@dataclass
class TinyGpsGrads2D:
    dL_dM: list[list[float]]
    dL_db: list[float]


def linear_logits_1d(x: float, m: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(m) != len(b):
        raise ValueError("m and b must have same length")
    return [mk * x + bk for mk, bk in zip(m, b)]


def linear_logits_2d(x: Sequence[float], M: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    if len(M) != len(b):
        raise ValueError("M and b must have same length")
    if len(x) != 2:
        raise ValueError("x must be 2D (lat, lon)")
    logits = []
    for row, bk in zip(M, b):
        if len(row) != 2:
            raise ValueError("each row of M must be length 2")
        logits.append(row[0] * x[0] + row[1] * x[1] + bk)
    return logits


def softmax_probs(logits: Iterable[float]) -> list[float]:
    return softmax(logits)


def cross_entropy_from_probs(probs: Sequence[float], target_index: int, eps: float = 1e-12) -> float:
    if target_index < 0 or target_index >= len(probs):
        raise IndexError("target_index out of range")
    p = min(max(probs[target_index], eps), 1.0 - eps)
    return -math.log(p)


def forward_1d(x: float, m: Sequence[float], b: Sequence[float], target_index: int) -> TinyGpsForward:
    logits = linear_logits_1d(x, m, b)
    probs = softmax_probs(logits)
    loss = cross_entropy_from_probs(probs, target_index)
    return TinyGpsForward(logits=logits, probs=probs, loss=loss)


def forward_2d(
    x: Sequence[float], M: Sequence[Sequence[float]], b: Sequence[float], target_index: int
) -> TinyGpsForward:
    logits = linear_logits_2d(x, M, b)
    probs = softmax_probs(logits)
    loss = cross_entropy_from_probs(probs, target_index)
    return TinyGpsForward(logits=logits, probs=probs, loss=loss)


def softmax_ce_backward(probs: Sequence[float], target_index: int) -> list[float]:
    if target_index < 0 or target_index >= len(probs):
        raise IndexError("target_index out of range")
    grads = [float(p) for p in probs]
    grads[target_index] -= 1.0
    return grads


def backward_1d(x: float, probs: Sequence[float], target_index: int) -> TinyGpsGrads1D:
    dL_dh = softmax_ce_backward(probs, target_index)
    dL_dm = [x * grad for grad in dL_dh]
    dL_db = dL_dh
    return TinyGpsGrads1D(dL_dm=dL_dm, dL_db=dL_db)


def backward_2d(x: Sequence[float], probs: Sequence[float], target_index: int) -> TinyGpsGrads2D:
    if len(x) != 2:
        raise ValueError("x must be 2D (lat, lon)")
    dL_dh = softmax_ce_backward(probs, target_index)
    dL_dM: list[list[float]] = []
    for grad in dL_dh:
        dL_dM.append([x[0] * grad, x[1] * grad])
    dL_db = dL_dh
    return TinyGpsGrads2D(dL_dM=dL_dM, dL_db=dL_db)


def accuracy_from_probs(probs: Sequence[float], target_index: int) -> float:
    if target_index < 0 or target_index >= len(probs):
        raise IndexError("target_index out of range")
    pred = max(range(len(probs)), key=lambda idx: probs[idx])
    return 1.0 if pred == target_index else 0.0


def mean_accuracy(
    samples: Sequence[dict],
    logits_fn,
) -> float:
    if not samples:
        raise ValueError("samples must be non-empty")
    correct = 0
    for sample in samples:
        logits = logits_fn(sample)
        probs = softmax_probs(logits)
        pred = max(range(len(probs)), key=lambda idx: probs[idx])
        if pred == sample["y"]:
            correct += 1
    return correct / float(len(samples))


def mean_ce_loss(samples: Sequence[dict], logits_fn) -> float:
    if not samples:
        raise ValueError("samples must be non-empty")
    total = 0.0
    for sample in samples:
        logits = logits_fn(sample)
        probs = softmax_probs(logits)
        total += cross_entropy_prob(probs[sample["y"]])
    return total / float(len(samples))
