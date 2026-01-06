"""Finite difference utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def finite_diff_grad(func: Callable[[list[float]], float], theta: Iterable[float], eps: float = 1e-5) -> list[float]:
    """Central-difference gradient estimate for a scalar function."""
    values = list(theta)
    if not values:
        raise ValueError("theta must be non-empty")
    grads: list[float] = []
    for idx in range(len(values)):
        theta_plus = values.copy()
        theta_minus = values.copy()
        theta_plus[idx] += eps
        theta_minus[idx] -= eps
        f_plus = func(theta_plus)
        f_minus = func(theta_minus)
        grads.append((f_plus - f_minus) / (2.0 * eps))
    return grads
