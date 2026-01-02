"""Differentiable neuron models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from backend.nn.activations import sigmoid
from backend.nn.losses import bce_grad_wrt_logit, bce_loss, mse_grad, mse_loss, pm1_to_01


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class AdalineStep:
    y_hat: float
    loss: float
    grad_w: List[float]
    grad_b: float


class Adaline:
    """Linear neuron trained with MSE (Adaline)."""

    def __init__(self, dim: int, lr: float = 1.0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.lr = lr
        self.w = [0.0 for _ in range(dim)]
        self.b = 0.0

    def forward(self, x: Sequence[float]) -> float:
        if len(x) != self.dim:
            raise ValueError("x has wrong dimension")
        return _dot(self.w, x) + self.b

    def step(self, x: Sequence[float], y: float) -> AdalineStep:
        y_hat = self.forward(x)
        loss = mse_loss(y, y_hat)
        grad_y_hat = mse_grad(y, y_hat)
        grad_w = [grad_y_hat * xi for xi in x]
        grad_b = grad_y_hat
        for i in range(self.dim):
            self.w[i] -= self.lr * grad_w[i]
        self.b -= self.lr * grad_b
        return AdalineStep(y_hat=y_hat, loss=loss, grad_w=grad_w, grad_b=grad_b)


@dataclass
class LogisticStep:
    p_hat: float
    loss: float
    grad_w: List[float]
    grad_b: float


class LogisticNeuron:
    """Logistic neuron with BCE loss."""

    def __init__(self, dim: int, lr: float = 1.0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.lr = lr
        self.w = [0.0 for _ in range(dim)]
        self.b = 0.0

    def logit(self, x: Sequence[float]) -> float:
        if len(x) != self.dim:
            raise ValueError("x has wrong dimension")
        return _dot(self.w, x) + self.b

    def forward(self, x: Sequence[float]) -> float:
        return sigmoid(self.logit(x))

    def step(self, x: Sequence[float], y_pm1: int) -> LogisticStep:
        y01 = pm1_to_01(y_pm1)
        z = self.logit(x)
        p_hat = sigmoid(z)
        loss = bce_loss(float(y01), p_hat)
        grad_z = bce_grad_wrt_logit(p_hat, float(y01))
        grad_w = [grad_z * xi for xi in x]
        grad_b = grad_z
        for i in range(self.dim):
            self.w[i] -= self.lr * grad_w[i]
        self.b -= self.lr * grad_b
        return LogisticStep(p_hat=p_hat, loss=loss, grad_w=grad_w, grad_b=grad_b)
