"""Grid visualization helpers (pure math + optional plotting)."""

from __future__ import annotations

from typing import List, Sequence


def contribution_grid(x_grid: Sequence[Sequence[float]], w_grid: Sequence[Sequence[float]]) -> List[List[float]]:
    if len(x_grid) != len(w_grid) or len(x_grid[0]) != len(w_grid[0]):
        raise ValueError("grid sizes must match")
    out: List[List[float]] = []
    for r in range(len(x_grid)):
        row: List[float] = []
        for c in range(len(x_grid[0])):
            row.append(x_grid[r][c] * w_grid[r][c])
        out.append(row)
    return out


def flatten(grid: Sequence[Sequence[float]]) -> List[float]:
    return [cell for row in grid for cell in row]


def score_from_grid(x_grid: Sequence[Sequence[float]], w_grid: Sequence[Sequence[float]], b: float = 0.0) -> float:
    contrib = contribution_grid(x_grid, w_grid)
    return sum(flatten(contrib)) + b


# Optional matplotlib rendering, imported lazily to avoid hard dependency.

def render_grid(ax, grid: Sequence[Sequence[float]], title: str = "") -> None:
    import matplotlib.pyplot as plt  # type: ignore

    ax.imshow(grid, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
