"""Chapter 4 Deep Learning service (Geometry of Depth)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from backend.core.datasets import make_baarle_hertog_dataset, make_xor_dataset_pm1
from backend.nn.deep_mlp import DeepMlp, DeepMlpStep

# Available datasets for deep learning experiments
DEEP_DATASETS = {
    "baarle": {
        "name": "Baarle-Hertog",
        "description": "Complex enclave borders (Belgium/Netherlands)",
        "n_classes": 2,
    },
    "xor": {
        "name": "XOR",
        "description": "Classic XOR problem",
        "n_classes": 2,
    },
    "circles": {
        "name": "Concentric Circles",
        "description": "Two concentric circles",
        "n_classes": 2,
    },
    "spiral": {
        "name": "Spiral",
        "description": "Two interleaved spirals",
        "n_classes": 2,
    },
}


def _make_circles_dataset(n_samples: int = 400, seed: int = 42) -> list[dict[str, Any]]:
    """Generate concentric circles dataset."""
    import math
    import random

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    n_per_class = n_samples // 2

    for _ in range(n_per_class):
        # Inner circle (class 0)
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0, 0.4)
        x1 = r * math.cos(angle)
        x2 = r * math.sin(angle)
        samples.append({"x": [x1, x2], "y": 0})

    for _ in range(n_per_class):
        # Outer ring (class 1)
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.6, 1.0)
        x1 = r * math.cos(angle)
        x2 = r * math.sin(angle)
        samples.append({"x": [x1, x2], "y": 1})

    rng.shuffle(samples)
    return samples


def _make_spiral_dataset(n_samples: int = 400, seed: int = 42) -> list[dict[str, Any]]:
    """Generate two interleaved spirals dataset."""
    import math
    import random

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    n_per_class = n_samples // 2

    # Both spirals use same t range [0, 2.5π], just rotated by π
    # This ensures symmetric spirals that fit in [-1, 1]
    max_t = 2.5 * math.pi
    noise = 0.03  # Reduced noise for cleaner separation

    for i in range(n_per_class):
        # First spiral (class 0)
        t = i / n_per_class * max_t + rng.uniform(-0.05, 0.05)
        r = 0.1 + 0.85 * (t / max_t)  # radius from 0.1 to 0.95
        x1 = r * math.cos(t) + rng.uniform(-noise, noise)
        x2 = r * math.sin(t) + rng.uniform(-noise, noise)
        samples.append({"x": [x1, x2], "y": 0})

    for i in range(n_per_class):
        # Second spiral (class 1, rotated by π)
        t = i / n_per_class * max_t + rng.uniform(-0.05, 0.05)
        r = 0.1 + 0.85 * (t / max_t)  # same radius progression
        x1 = r * math.cos(t + math.pi) + rng.uniform(-noise, noise)
        x2 = r * math.sin(t + math.pi) + rng.uniform(-noise, noise)
        samples.append({"x": [x1, x2], "y": 1})

    rng.shuffle(samples)
    return samples


@dataclass
class DeepState:
    """State for deep learning experiments."""

    dataset: str
    samples: list[dict[str, Any]]
    idx: int
    epoch: int
    total_steps: int
    lr: float
    model: DeepMlp


class DeepService:
    """Manages state for deep MLP experiments."""

    def __init__(self) -> None:
        self._state = self._init_state(
            dataset="circles",
            hidden_dims=[8, 8],
            lr=0.1,
            seed=42,
        )
        self._step_history: list[dict[str, Any]] = []
        self._comparison_table: list[dict[str, Any]] = []

    def _load_dataset(self, dataset: str, seed: int) -> list[dict[str, Any]]:
        """Load samples for a dataset."""
        if dataset == "baarle":
            # Try to find the image file
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            image_path = os.path.join(project_root, "data", "Baarle-Nassau_-_Baarle-Hertog-en no legend.png")
            if not os.path.exists(image_path):
                image_path = None
            return make_baarle_hertog_dataset(n_samples=500, seed=seed, image_path=image_path)
        elif dataset == "xor":
            # Convert XOR to 0/1 labels and scale inputs
            xor_raw = make_xor_dataset_pm1()
            samples = []
            for s in xor_raw:
                # Scale from {-1, 1} to [-0.8, 0.8] for better visualization
                x = [v * 0.8 for v in s["x"]]
                y = 1 if s["y"] == 1 else 0
                samples.append({"x": x, "y": y})
            return samples
        elif dataset == "circles":
            return _make_circles_dataset(n_samples=400, seed=seed)
        elif dataset == "spiral":
            return _make_spiral_dataset(n_samples=400, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _init_state(
        self,
        dataset: str,
        hidden_dims: list[int],
        lr: float,
        seed: int,
    ) -> DeepState:
        """Initialize a fresh state."""
        samples = self._load_dataset(dataset, seed)
        model = DeepMlp(
            input_dim=2,
            hidden_dims=hidden_dims,
            output_dim=2,
            lr=lr,
            seed=seed,
        )
        return DeepState(
            dataset=dataset,
            samples=samples,
            idx=0,
            epoch=0,
            total_steps=0,
            lr=lr,
            model=model,
        )

    def reset(
        self,
        dataset: str | None = None,
        hidden_dims: list[int] | None = None,
        lr: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Reset the model with new configuration."""
        dataset = dataset or self._state.dataset
        hidden_dims = hidden_dims or self._state.model.hidden_dims
        lr = lr if lr is not None else self._state.lr
        seed = seed if seed is not None else self._state.model.seed

        if dataset not in DEEP_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")

        self._state = self._init_state(dataset, hidden_dims, lr, seed)
        self._step_history = []
        return self.state()

    def state(self) -> dict[str, Any]:
        """Get current state as JSON-serializable dict."""
        s = self._state
        model_state = s.model.get_state()

        # Compute current metrics
        correct = 0
        total_loss = 0.0
        for sample in s.samples:
            fwd = s.model.forward(sample["x"])
            if fwd.prediction == sample["y"]:
                correct += 1
            total_loss += -__import__("math").log(max(fwd.output_probs[sample["y"]], 1e-15))

        accuracy = correct / len(s.samples) if s.samples else 0
        mean_loss = total_loss / len(s.samples) if s.samples else 0

        return {
            "dataset": s.dataset,
            "dataset_info": DEEP_DATASETS[s.dataset],
            "sample_count": len(s.samples),
            "idx": s.idx,
            "epoch": s.epoch,
            "total_steps": s.total_steps,
            "lr": s.lr,
            "architecture": {
                "input_dim": model_state.input_dim,
                "hidden_dims": model_state.hidden_dims,
                "output_dim": model_state.output_dim,
                "depth": s.model.depth,
                "width": s.model.width,
                "param_count": model_state.param_count,
            },
            "metrics": {
                "loss": mean_loss,
                "accuracy": accuracy,
            },
            "samples": s.samples,
        }

    def step(self, batch_size: int = 1) -> dict[str, Any]:
        """Take one or more training steps."""
        s = self._state
        steps: list[DeepMlpStep] = []

        for _ in range(batch_size):
            sample = s.samples[s.idx]
            step_result = s.model.step(sample["x"], sample["y"])
            steps.append(step_result)

            s.idx = (s.idx + 1) % len(s.samples)
            s.total_steps += 1
            if s.idx == 0:
                s.epoch += 1

        # Compute metrics after batch
        state = self.state()
        last_step = steps[-1]

        step_info = {
            "batch_size": batch_size,
            "last_loss": last_step.loss,
            "last_correct": last_step.correct,
            "last_prediction": last_step.forward.prediction,
            "grad_norm": last_step.grad_norm,
        }

        # Add to history
        self._step_history.append({
            "step": s.total_steps,
            "epoch": s.epoch,
            "loss": state["metrics"]["loss"],
            "accuracy": state["metrics"]["accuracy"],
        })

        return {**state, "step_info": step_info}

    def train_epoch(self) -> dict[str, Any]:
        """Train for one full epoch."""
        return self.step(batch_size=len(self._state.samples))

    def get_regions(self, resolution: int = 50) -> dict[str, Any]:
        """Get region counting information."""
        regions = self._state.model.count_regions(resolution=resolution)
        return {
            "count": regions.count,
            "theoretical_max": regions.theoretical_max,
            "efficiency": regions.count / regions.theoretical_max if regions.theoretical_max > 0 else 0,
        }

    def get_boundary(self, resolution: int = 50) -> dict[str, Any]:
        """Get boundary and tiling data for visualization."""
        predictions, region_ids = self._state.model.predict_batch(resolution=resolution)
        regions = self._state.model.count_regions(resolution=resolution)

        return {
            "resolution": resolution,
            "predictions": predictions,
            "region_ids": region_ids,
            "region_count": regions.count,
            "theoretical_max": regions.theoretical_max,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Get training history."""
        return self._step_history

    def add_to_comparison(self) -> dict[str, Any]:
        """Add current architecture to comparison table."""
        state = self.state()
        regions = self.get_regions(resolution=50)

        entry = {
            "depth": state["architecture"]["depth"],
            "width": state["architecture"]["width"],
            "hidden_dims": state["architecture"]["hidden_dims"],
            "param_count": state["architecture"]["param_count"],
            "actual_regions": regions["count"],
            "theoretical_max": regions["theoretical_max"],
            "accuracy": state["metrics"]["accuracy"],
            "loss": state["metrics"]["loss"],
            "total_steps": state["total_steps"],
        }
        self._comparison_table.append(entry)
        return entry

    def get_comparison_table(self) -> list[dict[str, Any]]:
        """Get the architecture comparison table."""
        return self._comparison_table

    def clear_comparison_table(self) -> None:
        """Clear the comparison table."""
        self._comparison_table = []

