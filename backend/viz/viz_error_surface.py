"""Error surface helpers for 2D weights."""

from __future__ import annotations

from typing import Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def mse_surface(
    samples: Iterable[dict],
    w_range: Tuple[float, float],
    steps: int = 25,
    b: float = 0.0,
) -> List[List[float]]:
    """Compute MSE surface for 2D weights over a grid."""
    if steps <= 1:
        raise ValueError("steps must be > 1")
    w_min, w_max = w_range
    if w_min >= w_max:
        raise ValueError("w_range must be increasing")
    data = list(samples)
    grid: List[List[float]] = []
    for i in range(steps):
        w1 = w_min + (w_max - w_min) * i / (steps - 1)
        row: List[float] = []
        for j in range(steps):
            w2 = w_min + (w_max - w_min) * j / (steps - 1)
            loss = 0.0
            for sample in data:
                x1, x2 = sample["x"]
                y = sample["y"]
                y_hat = w1 * x1 + w2 * x2 + b
                diff = y - y_hat
                loss += 0.5 * diff * diff
            row.append(loss / len(data))
        grid.append(row)
    return grid


def plot_surface(grid: List[List[float]], title: str = "MSE surface") -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    plt.imshow(grid, cmap="viridis")
    plt.colorbar()
    plt.title(title)
