"""CLI runner for perceptron training over simple datasets."""

from __future__ import annotations

import argparse
from typing import Dict, List

from backend.core.datasets import make_and_dataset_pm1, make_or_dataset_pm1, make_xor_dataset_pm1
from backend.core.metrics import accuracy, count_mistakes
from backend.core.perceptron import Perceptron


DATASETS = {
    "or": make_or_dataset_pm1,
    "and": make_and_dataset_pm1,
    "xor": make_xor_dataset_pm1,
}


def run_training(dataset: str, epochs: int, lr: float, seed: int | None = 0) -> Dict[str, List[float]]:
    if dataset not in DATASETS:
        raise ValueError("dataset must be one of: or, and, xor")
    samples = DATASETS[dataset]()
    p = Perceptron(dim=2, lr=lr, seed=seed, init="zeros")

    mistake_history: List[float] = []
    accuracy_history: List[float] = []

    for _ in range(epochs):
        results = p.train_epoch(samples, lr=lr, shuffle=True)
        mistake_history.append(count_mistakes([r.__dict__ for r in results]))
        accuracy_history.append(accuracy(samples, p.predict_label))

    return {"mistakes": mistake_history, "accuracy": accuracy_history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptron training runner")
    parser.add_argument("--dataset", default="or", choices=DATASETS.keys())
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    args = parser.parse_args()

    stats = run_training(args.dataset, args.epochs, args.lr)
    for i, (m, acc) in enumerate(zip(stats["mistakes"], stats["accuracy"])):
        print(f"epoch {i + 1:02d}: mistakes={int(m)} accuracy={acc:.2f}")


if __name__ == "__main__":
    main()
