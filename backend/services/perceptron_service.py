from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.core.datasets import make_or_dataset_pm1, make_xor_dataset_pm1
from backend.core.perceptron import Perceptron


class PerceptronService:
    def __init__(self, dataset: str = "or", lr: float = 1.0, seed: int | None = 0) -> None:
        self.lr = lr
        self.seed = seed
        self.custom_samples: List[Dict[str, Any]] | None = None
        self.custom_shape: Tuple[int, int] | None = None
        self.set_dataset(dataset)

    def set_lr(self, lr: float) -> None:
        self.lr = lr
        self.perceptron.lr = lr

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
        self.idx = 0
        self.grid_rows, self.grid_cols = grid_shape
        self.perceptron = Perceptron(dim=self.grid_rows * self.grid_cols, lr=self.lr, seed=self.seed, init="zeros")

    def step(self) -> Dict[str, Any]:
        sample = self.samples[self.idx]
        result = self.perceptron.train_step(sample["x"], sample["y"], lr=self.lr)
        self.idx = (self.idx + 1) % len(self.samples)
        next_sample = self.samples[self.idx]
        return {
            "w": self.perceptron.w,
            "b": self.perceptron.b,
            "x": sample["x"],
            "y": sample["y"],
            "score": result.score,
            "pred": result.pred,
            "mistake": result.mistake,
            "delta_w": result.delta_w,
            "delta_b": result.delta_b,
            "idx": self.idx,
            "dataset": self.dataset,
            "lr": self.lr,
            "next_x": next_sample["x"],
            "next_y": next_sample["y"],
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "sample_count": len(self.samples),
        }

    def reset(self) -> Dict[str, Any]:
        self.set_dataset(self.dataset)
        return self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "w": self.perceptron.w,
            "b": self.perceptron.b,
            "idx": self.idx,
            "dataset": self.dataset,
            "lr": self.lr,
            "next_x": self.samples[self.idx]["x"],
            "next_y": self.samples[self.idx]["y"],
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "sample_count": len(self.samples),
        }
