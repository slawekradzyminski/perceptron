"""2D visualization helpers (pure math + optional plotting)."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def decision_boundary_points(
    w: Sequence[float],
    b: float,
    x_min: float = -1.5,
    x_max: float = 1.5,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return two points for the decision boundary line w1*x + w2*y + b = 0.

    If w2 == 0, returns a vertical line x = -b / w1.
    """
    if len(w) != 2:
        raise ValueError("w must be length 2 for 2D boundary")
    w1, w2 = w
    if w1 == 0 and w2 == 0:
        raise ValueError("cannot compute boundary for zero vector")

    if w2 == 0:
        x = -b / w1
        return (x, x_min), (x, x_max)

    y1 = -(w1 * x_min + b) / w2
    y2 = -(w1 * x_max + b) / w2
    return (x_min, y1), (x_max, y2)


# Optional matplotlib rendering, imported lazily to avoid hard dependency.

def plot_points_and_boundary(ax, X: List[Sequence[float]], y: List[int], w, b) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    colors = ["#1f77b4" if label == 1 else "#d62728" for label in y]
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    ax.scatter(xs, ys, c=colors, s=80, edgecolors="#111")
    (x1, y1), (x2, y2) = decision_boundary_points(w, b)
    ax.plot([x1, x2], [y1, y2], color="#222", linewidth=2)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="#ccc", linewidth=1)
    ax.axvline(0, color="#ccc", linewidth=1)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
