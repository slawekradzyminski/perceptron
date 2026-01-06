"""Visualization helpers for MLP internals."""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def plot_heatmap(grid: list[list[float]], title: str = "") -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    plt.imshow(grid, cmap="coolwarm")
    plt.colorbar()
    if title:
        plt.title(title)


def plot_templates(templates: list[list[list[float]]], title_prefix: str = "Template") -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    count = len(templates)
    cols = min(4, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_list = axes if isinstance(axes, list) else axes.flatten()
    for i, template in enumerate(templates):
        ax = axes_list[i]
        ax.imshow(template, cmap="coolwarm")
        ax.set_title(f"{title_prefix} {i + 1}")
        ax.axis("off")
    for j in range(count, len(axes_list)):
        axes_list[j].axis("off")
    fig.tight_layout()
