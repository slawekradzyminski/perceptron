from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.datasets import make_or_dataset_pm1, make_xor_dataset_pm1
from backend.nn.grid_mlp import reshape_template
from backend.nn.mlp import MlpInternals, MlpTwoLayer


def _pred_from_prob(p_hat: float) -> int:
    return 1 if p_hat >= 0.5 else -1


class MlpService:
    def __init__(self, dataset: str = "xor", hidden_dim: int = 2, lr: float = 0.5, seed: int | None = 0) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.custom_samples: Optional[List[Dict[str, Any]]] = None
        self.custom_shape: Optional[Tuple[int, int]] = None
        self.set_dataset(dataset)

    def set_hyperparams(self, hidden_dim: int | None = None, lr: float | None = None, seed: int | None = None) -> None:
        if hidden_dim is not None:
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            self.hidden_dim = hidden_dim
        if lr is not None:
            self.lr = lr
            if hasattr(self, "model"):
                self.model.lr = lr
        if seed is not None:
            self.seed = seed

    def set_dataset(self, name: str, custom: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None) -> None:
        if name == "or":
            self.samples = make_or_dataset_pm1()
            grid_shape = (1, 2)
        elif name == "xor":
            self.samples = make_xor_dataset_pm1()
            grid_shape = (1, 2)
        elif name == "custom":
            if custom is not None:
                self.custom_samples, self.custom_shape = custom
            if self.custom_samples is None or self.custom_shape is None:
                raise ValueError("custom dataset requires samples and grid size")
            self.samples = self.custom_samples
            grid_shape = self.custom_shape
        else:
            raise ValueError("dataset must be 'or', 'xor', or 'custom'")
        self.dataset = name
        self.grid_rows, self.grid_cols = grid_shape
        self.reset_model()

    def reset_model(self) -> None:
        self.idx = 0
        self.model = MlpTwoLayer(
            input_dim=self.grid_rows * self.grid_cols,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            seed=self.seed,
        )

    def snapshot(self) -> Dict[str, Any]:
        templates = [reshape_template(row, self.grid_rows, self.grid_cols) for row in self.model.hidden.W]
        evals = []
        for sample in self.samples:
            p_hat = self.model.forward(sample["x"])
            evals.append(
                {
                    "x": sample["x"],
                    "y": sample["y"],
                    "p_hat": p_hat,
                    "pred": _pred_from_prob(p_hat),
                }
            )
        next_sample = self.samples[self.idx]
        return {
            "dataset": self.dataset,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "seed": self.seed,
            "idx": self.idx,
            "sample_count": len(self.samples),
            "next_x": next_sample["x"],
            "next_y": next_sample["y"],
            "hidden": {
                "weights": self.model.hidden.W,
                "bias": self.model.hidden.b,
                "templates": templates,
            },
            "output": {
                "weights": self.model.output.W,
                "bias": self.model.output.b,
            },
            "evals": evals,
        }

    def step(self) -> Tuple[Dict[str, Any], MlpInternals]:
        sample = self.samples[self.idx]
        internals = self.model.inspect_step(sample["x"], sample["y"])
        self.idx = (self.idx + 1) % len(self.samples)
        return self.snapshot(), internals
