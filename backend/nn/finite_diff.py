"""Finite difference utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List


def finite_diff_grad(func: Callable[[List[float]], float], theta: Iterable[float], eps: float = 1e-5) -> List[float]:
    """Central-difference gradient estimate for a scalar function."""
    values = list(theta)
    if not values:
        raise ValueError("theta must be non-empty")
    grads: List[float] = []
    for idx in range(len(values)):
        theta_plus = values.copy()
        theta_minus = values.copy()
        theta_plus[idx] += eps
        theta_minus[idx] -= eps
        f_plus = func(theta_plus)
        f_minus = func(theta_minus)
        grads.append((f_plus - f_minus) / (2.0 * eps))
    return grads
