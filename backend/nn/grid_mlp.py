"""Grid-shaped MLP helpers for visualizing templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from backend.nn.mlp import MlpTwoLayer


def reshape_template(weights: Sequence[float], rows: int, cols: int) -> List[List[float]]:
    if len(weights) != rows * cols:
        raise ValueError("weights size does not match grid shape")
    grid: List[List[float]] = []
    for r in range(rows):
        start = r * cols
        grid.append(list(weights[start:start + cols]))
    return grid


@dataclass
class GridMlpStep:
    loss: float
    p_hat: float


class GridMlp:
    """MLP specialized for grid inputs with template visualization."""

    def __init__(self, rows: int, cols: int, hidden_dim: int = 4, lr: float = 0.5, seed: int | None = 0) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        self.rows = rows
        self.cols = cols
        self.model = MlpTwoLayer(input_dim=rows * cols, hidden_dim=hidden_dim, lr=lr, seed=seed)

    def step(self, x: Sequence[float], y: int) -> GridMlpStep:
        step = self.model.step(x, y)
        return GridMlpStep(loss=step.loss, p_hat=step.p_hat)

    def train(self, samples: Sequence[dict], epochs: int) -> List[float]:
        return self.model.train(samples, epochs)

    def weight_templates(self) -> List[List[List[float]]]:
        return [reshape_template(w_row, self.rows, self.cols) for w_row in self.model.hidden.W]

    def gradient_templates(self) -> List[List[List[float]]] | None:
        if self.model.last_grad_hidden is None:
            return None
        return [reshape_template(row, self.rows, self.cols) for row in self.model.last_grad_hidden]
