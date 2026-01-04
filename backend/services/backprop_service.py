"""Chapter 3 backprop services (TinyGPS + regression)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

from backend.core.datasets import (
    make_tinygps_dataset_1d,
    make_tinygps_dataset_1d_from_coords,
    make_tinygps_dataset_1d_multi,
    make_tinygps_dataset_2d,
    tinygps_city_coords,
    tinygps_city_coords_exercise,
)
from backend.nn.regression import LossType, backward as reg_backward, forward as reg_forward
from backend.nn.tinygps import (
    backward_1d,
    backward_2d,
    forward_1d,
    forward_2d,
    mean_accuracy,
    mean_ce_loss,
)

TinyGpsMode = Literal["1d", "2d"]

TINYGPS_DATASETS: Dict[str, Dict[str, Any]] = {
    "paris-berlin": {"mode": "1d", "cities": ["paris", "berlin"]},
    "paris-madrid": {"mode": "1d", "cities": ["paris", "madrid"]},
    "paris-berlin-4": {"mode": "1d", "cities": ["paris", "berlin"], "exercise": True},
    "paris-madrid-4": {"mode": "1d", "cities": ["paris", "madrid"], "exercise": True},
    "madrid-paris-berlin": {"mode": "1d", "cities": ["madrid", "paris", "berlin"]},
    "four-cities": {"mode": "2d", "cities": ["madrid", "paris", "berlin", "barcelona"]},
}

DEFAULT_REGRESSION_SAMPLES = [
    {"x": 1.0, "y": 3.0},
    {"x": 2.0, "y": 5.0},
    {"x": 3.0, "y": 7.0},
    {"x": 4.0, "y": 9.0},
]


@dataclass
class TinyGpsState:
    dataset: str
    mode: TinyGpsMode
    samples: List[Dict[str, Any]]
    sample_order: List[int]
    idx: int
    lr: float
    m: List[float]
    b: List[float]
    M: List[List[float]]


@dataclass
class RegressionState:
    loss: LossType
    samples: List[Dict[str, float]]
    sample_order: List[int]
    idx: int
    lr: float
    m: float
    b: float


class BackpropService:
    def __init__(self) -> None:
        self._city_coords = tinygps_city_coords()
        self._tinygps = self._init_tinygps("madrid-paris-berlin", lr=0.1)
        self._regression = self._init_regression(DEFAULT_REGRESSION_SAMPLES, loss="mse", lr=0.1)

    def _init_tinygps(self, dataset: str, lr: float) -> TinyGpsState:
        if dataset not in TINYGPS_DATASETS:
            raise ValueError("unknown TinyGPS dataset")
        config = TINYGPS_DATASETS[dataset]
        mode = config["mode"]
        cities = config["cities"]
        if mode == "1d":
            if config.get("exercise"):
                samples = make_tinygps_dataset_1d_from_coords(tinygps_city_coords_exercise(), cities)
                k = len(cities)
                m = [0.0 for _ in range(k)]
                b = [0.0 for _ in range(k)]
                M: List[List[float]] = []
                sample_order = list(range(len(samples)))
                return TinyGpsState(
                    dataset=dataset,
                    mode=mode,
                    samples=samples,
                    sample_order=sample_order,
                    idx=0,
                    lr=lr,
                    m=m,
                    b=b,
                    M=M,
                )
            if len(cities) == 2:
                samples = make_tinygps_dataset_1d(cities[0], cities[1])
            else:
                samples = make_tinygps_dataset_1d_multi(cities)
            k = len(cities)
            m = [0.0 for _ in range(k)]
            b = [0.0 for _ in range(k)]
            M: List[List[float]] = []
        else:
            samples = make_tinygps_dataset_2d(cities)
            k = len(cities)
            m = []
            b = [0.0 for _ in range(k)]
            M = [[0.0, 0.0] for _ in range(k)]
        sample_order = list(range(len(samples)))
        return TinyGpsState(
            dataset=dataset,
            mode=mode,
            samples=samples,
            sample_order=sample_order,
            idx=0,
            lr=lr,
            m=m,
            b=b,
            M=M,
        )

    def _init_regression(self, samples: List[Dict[str, float]], loss: LossType, lr: float) -> RegressionState:
        if not samples:
            raise ValueError("regression samples must be non-empty")
        if loss not in ("mse", "l1"):
            raise ValueError("loss must be 'mse' or 'l1'")
        sample_order = list(range(len(samples)))
        return RegressionState(loss=loss, samples=samples, sample_order=sample_order, idx=0, lr=lr, m=0.0, b=0.0)

    def reset_tinygps(
        self,
        dataset: str | None = None,
        lr: float | None = None,
        params: Dict[str, Any] | None = None,
        order: List[int] | None = None,
    ) -> Dict[str, Any]:
        dataset_name = dataset or self._tinygps.dataset
        lr_value = self._tinygps.lr if lr is None else lr
        self._tinygps = self._init_tinygps(dataset_name, lr_value)
        if params:
            if "m" in params:
                m = list(params["m"])
                if len(m) != len(self._tinygps.b):
                    raise ValueError("m length must match number of classes")
                self._tinygps.m = m
            if "b" in params:
                b = list(params["b"])
                if len(b) != len(self._tinygps.b):
                    raise ValueError("b length must match number of classes")
                self._tinygps.b = b
            if "M" in params:
                M = [list(row) for row in params["M"]]
                if len(M) != len(self._tinygps.b):
                    raise ValueError("M length must match number of classes")
                self._tinygps.M = M
        if order is not None:
            if not order:
                raise ValueError("order must be non-empty")
            if any(idx < 0 or idx >= len(self._tinygps.samples) for idx in order):
                raise ValueError("order indices must be within sample range")
            self._tinygps.samples = [self._tinygps.samples[i] for i in order]
            self._tinygps.sample_order = order
        return self.tinygps_state()

    def reset_regression(
        self,
        loss: LossType | None = None,
        lr: float | None = None,
        samples: List[Dict[str, float]] | None = None,
        order: List[int] | None = None,
        params: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        loss_name = loss or self._regression.loss
        lr_value = self._regression.lr if lr is None else lr
        sample_list = samples if samples is not None else self._regression.samples
        self._regression = self._init_regression(sample_list, loss=loss_name, lr=lr_value)
        if params:
            if "m" in params:
                self._regression.m = float(params["m"])
            if "b" in params:
                self._regression.b = float(params["b"])
        if order is not None:
            if not order:
                raise ValueError("order must be non-empty")
            if any(idx < 0 or idx >= len(self._regression.samples) for idx in order):
                raise ValueError("order indices must be within sample range")
            self._regression.samples = [self._regression.samples[i] for i in order]
            self._regression.sample_order = order
        return self.regression_state()

    def tinygps_state(self) -> Dict[str, Any]:
        state = self._tinygps
        sample = state.samples[state.idx]
        return {
            "dataset": state.dataset,
            "mode": state.mode,
            "cities": TINYGPS_DATASETS[state.dataset]["cities"],
            "idx": state.idx,
            "lr": state.lr,
            "sample_count": len(state.samples),
            "feature_order": ["lon"] if state.mode == "1d" else ["lat", "lon"],
            "sample_order": state.sample_order,
            "samples": state.samples,
            "params": {
                "m": state.m,
                "b": state.b,
                "M": state.M,
            },
            "next_sample": sample,
        }

    def regression_state(self) -> Dict[str, Any]:
        state = self._regression
        sample = state.samples[state.idx]
        return {
            "loss": state.loss,
            "idx": state.idx,
            "lr": state.lr,
            "sample_count": len(state.samples),
            "params": {"m": state.m, "b": state.b},
            "next_sample": sample,
            "samples": state.samples,
            "sample_order": state.sample_order,
        }

    def state(self) -> Dict[str, Any]:
        return {
            "tinygps": self.tinygps_state(),
            "regression": self.regression_state(),
            "tinygps_datasets": list(TINYGPS_DATASETS.keys()),
            "tinygps_city_coords": self._city_coords,
        }

    def step_tinygps(self) -> Dict[str, Any]:
        state = self._tinygps
        sample_idx = state.idx
        sample = state.samples[sample_idx]
        params_before = {
            "m": list(state.m),
            "b": list(state.b),
            "M": [row[:] for row in state.M],
        }

        if state.mode == "1d":
            forward = forward_1d(sample["x"][0], state.m, state.b, sample["y"])
            grads = backward_1d(sample["x"][0], forward.probs, sample["y"])
            for i in range(len(state.m)):
                state.m[i] -= state.lr * grads.dL_dm[i]
                state.b[i] -= state.lr * grads.dL_db[i]
            grads_payload = {"m": grads.dL_dm, "b": grads.dL_db}
        else:
            forward = forward_2d(sample["x"], state.M, state.b, sample["y"])
            grads = backward_2d(sample["x"], forward.probs, sample["y"])
            for i in range(len(state.M)):
                state.M[i][0] -= state.lr * grads.dL_dM[i][0]
                state.M[i][1] -= state.lr * grads.dL_dM[i][1]
                state.b[i] -= state.lr * grads.dL_db[i]
            grads_payload = {"M": grads.dL_dM, "b": grads.dL_db}

        pred = max(range(len(forward.probs)), key=lambda idx: forward.probs[idx])
        correct = pred == sample["y"]

        def logits_fn(s: Dict[str, Any]) -> List[float]:
            if state.mode == "1d":
                return [mk * s["x"][0] + bk for mk, bk in zip(state.m, state.b)]
            return [row[0] * s["x"][0] + row[1] * s["x"][1] + bk for row, bk in zip(state.M, state.b)]

        metrics_after = {
            "loss": mean_ce_loss(state.samples, logits_fn),
            "accuracy": mean_accuracy(state.samples, logits_fn),
        }

        state.idx = (state.idx + 1) % len(state.samples)

        return {
            "dataset": state.dataset,
            "mode": state.mode,
            "cities": TINYGPS_DATASETS[state.dataset]["cities"],
            "idx": state.idx,
            "sample_idx": sample_idx,
            "lr": state.lr,
            "feature_order": ["lon"] if state.mode == "1d" else ["lat", "lon"],
            "sample": sample,
            "logits": forward.logits,
            "probs": forward.probs,
            "loss": forward.loss,
            "pred": pred,
            "correct": correct,
            "grads": grads_payload,
            "params_before": params_before,
            "params_after": {
                "m": state.m,
                "b": state.b,
                "M": state.M,
            },
            "metrics_after": metrics_after,
            "sample_count": len(state.samples),
        }

    def step_regression(self) -> Dict[str, Any]:
        state = self._regression
        sample_idx = state.idx
        sample = state.samples[sample_idx]
        params_before = {"m": state.m, "b": state.b}
        forward = reg_forward(sample["x"], sample["y"], state.m, state.b, state.loss)
        grads = reg_backward(sample["x"], sample["y"], forward.y_hat, state.loss)

        state.m -= state.lr * grads.dL_dm
        state.b -= state.lr * grads.dL_db

        def mean_loss() -> float:
            total = 0.0
            for row in state.samples:
                step = reg_forward(row["x"], row["y"], state.m, state.b, state.loss)
                total += step.loss
            return total / float(len(state.samples))

        state.idx = (state.idx + 1) % len(state.samples)

        return {
            "loss": state.loss,
            "idx": state.idx,
            "sample_idx": sample_idx,
            "lr": state.lr,
            "sample": sample,
            "y_hat": forward.y_hat,
            "loss_value": forward.loss,
            "grads": {"m": grads.dL_dm, "b": grads.dL_db},
            "params_before": params_before,
            "params_after": {"m": state.m, "b": state.b},
            "mean_loss": mean_loss(),
            "sample_count": len(state.samples),
        }
