"""Mini runner helpers for differentiable neurons."""

from __future__ import annotations

from typing import List, Sequence

from backend.nn.models import Adaline, LogisticNeuron


def train_adaline(samples: Sequence[dict], epochs: int, lr: float) -> List[float]:
    model = Adaline(dim=len(samples[0]["x"]), lr=lr)
    losses: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for sample in samples:
            step = model.step(sample["x"], sample["y"])
            epoch_loss += step.loss
        losses.append(epoch_loss / len(samples))
    return losses


def train_logistic(samples: Sequence[dict], epochs: int, lr: float) -> List[float]:
    model = LogisticNeuron(dim=len(samples[0]["x"]), lr=lr)
    losses: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for sample in samples:
            step = model.step(sample["x"], sample["y"])
            epoch_loss += step.loss
        losses.append(epoch_loss / len(samples))
    return losses
