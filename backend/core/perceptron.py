"""Classic Rosenblatt perceptron with mistake-driven updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import random


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class StepResult:
    score: float
    pred: int
    mistake: bool
    delta_w: List[float]
    delta_b: float


class Perceptron:
    def __init__(self, dim: int, lr: float = 1.0, seed: int | None = None, init: str = "zeros") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.lr = lr
        self._rng = random.Random(seed)
        if init == "zeros":
            self.w = [0.0 for _ in range(dim)]
            self.b = 0.0
        elif init == "random":
            self.w = [self._rng.uniform(-0.5, 0.5) for _ in range(dim)]
            self.b = self._rng.uniform(-0.5, 0.5)
        else:
            raise ValueError("init must be 'zeros' or 'random'")

    def predict_score(self, x: Sequence[float]) -> float:
        if len(x) != self.dim:
            raise ValueError("x has wrong dimension")
        return _dot(self.w, x) + self.b

    def predict_label(self, x: Sequence[float]) -> int:
        score = self.predict_score(x)
        return 1 if score >= 0 else -1

    def train_step(self, x: Sequence[float], y: int, lr: float | None = None) -> StepResult:
        if y not in (-1, 1):
            raise ValueError("y must be -1 or +1")
        score = self.predict_score(x)
        pred = 1 if score >= 0 else -1
        mistake = (y * score) <= 0
        step_lr = self.lr if lr is None else lr
        delta_w = [0.0 for _ in range(self.dim)]
        delta_b = 0.0
        if mistake:
            for i in range(self.dim):
                delta_w[i] = step_lr * y * x[i]
                self.w[i] += delta_w[i]
            delta_b = step_lr * y
            self.b += delta_b
        return StepResult(score=score, pred=pred, mistake=mistake, delta_w=delta_w, delta_b=delta_b)

    def train_epoch(self, samples: Iterable[dict], lr: float | None = None, shuffle: bool = True) -> List[StepResult]:
        data = list(samples)
        if shuffle:
            self._rng.shuffle(data)
        results: List[StepResult] = []
        for sample in data:
            results.append(self.train_step(sample["x"], sample["y"], lr=lr))
        return results
