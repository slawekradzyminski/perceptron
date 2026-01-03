from __future__ import annotations

from typing import Any, Dict, List


class LmsService:
    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr
        self.samples: List[Dict[str, Any]] = [
            {"x": [-1.0, -1.0], "y": -1},
            {"x": [-1.0, 1.0], "y": -1},
            {"x": [1.0, -1.0], "y": 1},
            {"x": [1.0, 1.0], "y": 1},
        ]
        self.reset()

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
