"""Loss functions and gradients."""

from __future__ import annotations

import math


def perceptron_loss(y: int, score: float) -> float:
    """Perceptron hinge-like loss for y in {-1, +1}."""
    if y not in (-1, 1):
        raise ValueError("y must be -1 or +1")
    return max(0.0, -y * score)


def mse_loss(y: float, y_hat: float) -> float:
    """Mean squared error (single sample)."""
    diff = y - y_hat
    return 0.5 * diff * diff


def mse_grad(y: float, y_hat: float) -> float:
    """Gradient of MSE with respect to y_hat."""
    return y_hat - y


def bce_loss(y: float, p_hat: float, eps: float = 1e-12) -> float:
    """Binary cross-entropy for y in {0, 1} and probability p_hat."""
    if y not in (0.0, 1.0):
        raise ValueError("y must be 0 or 1 for BCE")
    p = min(max(p_hat, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def bce_grad_wrt_logit(p_hat: float, y: float) -> float:
    """Gradient of BCE with sigmoid output w.r.t. logit (z)."""
    if y not in (0.0, 1.0):
        raise ValueError("y must be 0 or 1 for BCE")
    return p_hat - y


def pm1_to_01(y: int) -> int:
    """Map labels from {-1, +1} to {0, 1}."""
    if y not in (-1, 1):
        raise ValueError("y must be -1 or +1")
    return 1 if y == 1 else 0
