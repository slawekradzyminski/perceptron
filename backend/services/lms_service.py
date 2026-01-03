from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.datasets import make_or_dataset_pm1, make_xor_dataset_pm1


class LmsService:
    def __init__(self, lr: float = 0.1, dataset: str = "or") -> None:
        self.lr = lr
        self.custom_samples: Optional[List[Dict[str, Any]]] = None
        self.set_dataset(dataset)

    def set_dataset(self, name: str, custom_samples: Optional[List[Dict[str, Any]]] = None) -> None:
        if name == "or":
            samples = make_or_dataset_pm1()
        elif name == "xor":
            samples = make_xor_dataset_pm1()
        elif name == "custom":
            if custom_samples is not None:
                self.custom_samples = custom_samples
            if self.custom_samples is None:
                raise ValueError("custom dataset requires samples")
            samples = self.custom_samples
        else:
            raise ValueError("dataset must be 'or', 'xor', or 'custom'")
        if any(len(sample.get("x", [])) != 2 for sample in samples):
            raise ValueError("LMS requires 2D inputs")
        self.dataset = name
        self.samples = samples
        self.reset()

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def reset(self) -> Dict[str, Any]:
        self.idx = 0
        self.w = [0.0, 0.0]
        self.b = 0.0
        return self.state()

    def state(self) -> Dict[str, Any]:
        sample = self.samples[self.idx]
        return {
            "w": self.w,
            "b": self.b,
            "idx": self.idx,
            "lr": self.lr,
            "x": sample["x"],
            "y": sample["y"],
            "sample_count": len(self.samples),
            "dataset": self.dataset,
        }

    def step(self) -> Dict[str, Any]:
        sample = self.samples[self.idx]
        x1, x2 = sample["x"]
        y = sample["y"]
        y_hat = self.w[0] * x1 + self.w[1] * x2 + self.b
        error = y - y_hat
        grad_common = -2 * error
        grad_w1 = grad_common * x1
        grad_w2 = grad_common * x2
        grad_b = grad_common
        w_before = self.w[:]
        b_before = self.b
        self.w[0] += self.lr * error * x1
        self.w[1] += self.lr * error * x2
        self.b += self.lr * error
        self.idx = (self.idx + 1) % len(self.samples)
        return {
            "x": [x1, x2],
            "y": y,
            "w_before": w_before,
            "b_before": b_before,
            "y_hat": y_hat,
            "error": error,
            "grad_w1": grad_w1,
            "grad_w2": grad_w2,
            "grad_b": grad_b,
            "w_after": self.w,
            "b_after": self.b,
            "idx": self.idx,
            "lr": self.lr,
        }
