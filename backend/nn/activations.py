"""Activation functions and derivatives."""

from __future__ import annotations

import math


def step_pm1(x: float) -> int:
    """Hard step returning -1 or +1."""
    return 1 if x >= 0 else -1


def sigmoid(x: float) -> float:
    """Logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_prime_from_output(p: float) -> float:
    """Derivative of sigmoid given output p."""
    return p * (1.0 - p)


def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)


def tanh_prime_from_output(t: float) -> float:
    """Derivative of tanh given output t."""
    return 1.0 - t * t


def relu(x: float) -> float:
    """ReLU activation."""
    return x if x > 0 else 0.0


def relu_prime(x: float) -> float:
    """Derivative of ReLU with respect to input."""
    return 1.0 if x > 0 else 0.0
