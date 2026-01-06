"""Deep MLP with variable depth for Chapter 4 (Geometry of Depth)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field


def relu(x: float) -> float:
    """ReLU activation."""
    return x if x > 0 else 0.0


def relu_prime(x: float) -> float:
    """Derivative of ReLU."""
    return 1.0 if x > 0 else 0.0


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax."""
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(probs: Sequence[float], target: int) -> float:
    """Cross-entropy loss for classification."""
    p = probs[target]
    return -math.log(max(p, 1e-15))


@dataclass
class DeepMlpForward:
    """Forward pass result with all intermediate activations."""

    hidden_z: list[list[float]]  # Pre-activation for each hidden layer
    hidden_a: list[list[float]]  # Post-activation (ReLU) for each hidden layer
    output_logits: list[float]  # Final logits (pre-softmax)
    output_probs: list[float]  # Softmax probabilities
    activation_signature: tuple[int, ...]  # Binary signature of ReLU activations

    @property
    def prediction(self) -> int:
        """Predicted class (argmax of probabilities)."""
        return max(range(len(self.output_probs)), key=lambda i: self.output_probs[i])


@dataclass
class DeepMlpStep:
    """Result of a training step."""

    x: list[float]
    y: int
    forward: DeepMlpForward
    loss: float
    correct: bool
    grad_norm: float


@dataclass
class DeepMlpState:
    """Serializable state of the model."""

    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    lr: float
    hidden_weights: list[list[list[float]]]
    hidden_biases: list[list[float]]
    output_weights: list[list[float]]
    output_bias: list[float]
    param_count: int


@dataclass
class RegionInfo:
    """Region counting results."""

    count: int
    theoretical_max: int
    signatures: dict[tuple[int, ...], list[tuple[float, float]]] = field(default_factory=dict)


class DeepMlp:
    """Generic K-layer MLP with ReLU hidden activations."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        output_dim: int = 2,
        lr: float = 0.01,
        seed: int = 0,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [16]
        self.output_dim = output_dim
        self.lr = lr
        self.seed = seed

        # Validate hidden dims
        for i, dim in enumerate(self.hidden_dims):
            if dim <= 0:
                raise ValueError(f"hidden_dims[{i}] must be positive")

        self._init_weights(seed)

    def _init_weights(self, seed: int) -> None:
        """Initialize weights using Xavier/He initialization."""
        rng = random.Random(seed)

        self.hidden_W: list[list[list[float]]] = []
        self.hidden_b: list[list[float]] = []

        prev_dim = self.input_dim
        for dim in self.hidden_dims:
            # He initialization for ReLU: std = sqrt(2/fan_in)
            std = (2.0 / prev_dim) ** 0.5
            W = [[rng.gauss(0, std) for _ in range(prev_dim)] for _ in range(dim)]
            b = [0.0 for _ in range(dim)]
            self.hidden_W.append(W)
            self.hidden_b.append(b)
            prev_dim = dim

        # Output layer
        std = (1.0 / prev_dim) ** 0.5
        self.output_W = [[rng.gauss(0, std) for _ in range(prev_dim)] for _ in range(self.output_dim)]
        self.output_b = [0.0 for _ in range(self.output_dim)]

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        count = 0
        for W, b in zip(self.hidden_W, self.hidden_b):
            count += sum(len(row) for row in W) + len(b)
        count += sum(len(row) for row in self.output_W) + len(self.output_b)
        return count

    @property
    def depth(self) -> int:
        """Number of hidden layers."""
        return len(self.hidden_dims)

    @property
    def width(self) -> int:
        """Width of first hidden layer (representative)."""
        return self.hidden_dims[0] if self.hidden_dims else 0

    def forward(self, x: Sequence[float]) -> DeepMlpForward:
        """Forward pass retaining all activations."""
        if len(x) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {len(x)}")

        hidden_z: list[list[float]] = []
        hidden_a: list[list[float]] = []
        signature_bits: list[int] = []

        current = list(x)
        for W, b in zip(self.hidden_W, self.hidden_b):
            z = [sum(W[i][j] * current[j] for j in range(len(current))) + b[i] for i in range(len(W))]
            a = [relu(zi) for zi in z]
            hidden_z.append(z)
            hidden_a.append(a)
            # Record activation signature (1 if active, 0 if not)
            signature_bits.extend(1 if zi > 0 else 0 for zi in z)
            current = a

        # Output layer (no activation before softmax)
        logits = [
            sum(self.output_W[i][j] * current[j] for j in range(len(current))) + self.output_b[i]
            for i in range(self.output_dim)
        ]
        probs = softmax(logits)

        return DeepMlpForward(
            hidden_z=hidden_z,
            hidden_a=hidden_a,
            output_logits=logits,
            output_probs=probs,
            activation_signature=tuple(signature_bits),
        )

    def step(self, x: Sequence[float], y: int) -> DeepMlpStep:
        """Single training step with backprop."""
        if y < 0 or y >= self.output_dim:
            raise ValueError(f"Target y must be in [0, {self.output_dim}), got {y}")

        fwd = self.forward(x)
        loss = cross_entropy_loss(fwd.output_probs, y)
        correct = fwd.prediction == y

        # Backprop
        grad_norm = self._backward(list(x), y, fwd)

        return DeepMlpStep(
            x=list(x),
            y=y,
            forward=fwd,
            loss=loss,
            correct=correct,
            grad_norm=grad_norm,
        )

    def _backward(self, x: list[float], y: int, fwd: DeepMlpForward) -> float:
        """Backpropagation. Returns gradient norm."""
        # Gradient of cross-entropy + softmax: p - one_hot(y)
        grad_logits = list(fwd.output_probs)
        grad_logits[y] -= 1.0

        # Gradient for output layer
        last_hidden = fwd.hidden_a[-1] if fwd.hidden_a else x
        grad_W_out = [[grad_logits[i] * last_hidden[j] for j in range(len(last_hidden))] for i in range(self.output_dim)]
        grad_b_out = list(grad_logits)

        # Gradient flowing back to last hidden layer
        grad_hidden = [sum(self.output_W[i][j] * grad_logits[i] for i in range(self.output_dim)) for j in range(len(last_hidden))]

        # Apply output gradients
        for i in range(self.output_dim):
            for j in range(len(self.output_W[i])):
                self.output_W[i][j] -= self.lr * grad_W_out[i][j]
            self.output_b[i] -= self.lr * grad_b_out[i]

        grad_norm = sum(abs(g) for row in grad_W_out for g in row) + sum(abs(g) for g in grad_b_out)

        # Backprop through hidden layers (reverse order)
        for layer_idx in range(len(self.hidden_dims) - 1, -1, -1):
            W = self.hidden_W[layer_idx]
            b = self.hidden_b[layer_idx]
            z = fwd.hidden_z[layer_idx]
            prev_a = fwd.hidden_a[layer_idx - 1] if layer_idx > 0 else x

            # Apply ReLU derivative
            grad_z = [grad_hidden[i] * relu_prime(z[i]) for i in range(len(z))]

            # Compute gradients
            grad_W = [[grad_z[i] * prev_a[j] for j in range(len(prev_a))] for i in range(len(W))]
            grad_b = list(grad_z)

            # Gradient for previous layer
            if layer_idx > 0:
                grad_hidden = [sum(W[i][j] * grad_z[i] for i in range(len(W))) for j in range(len(prev_a))]

            # Apply gradients
            for i in range(len(W)):
                for j in range(len(W[i])):
                    W[i][j] -= self.lr * grad_W[i][j]
                b[i] -= self.lr * grad_b[i]

            grad_norm += sum(abs(g) for row in grad_W for g in row) + sum(abs(g) for g in grad_b)

        return grad_norm

    def get_state(self) -> DeepMlpState:
        """Get serializable state."""
        return DeepMlpState(
            input_dim=self.input_dim,
            hidden_dims=list(self.hidden_dims),
            output_dim=self.output_dim,
            lr=self.lr,
            hidden_weights=[[row[:] for row in W] for W in self.hidden_W],
            hidden_biases=[b[:] for b in self.hidden_b],
            output_weights=[row[:] for row in self.output_W],
            output_bias=self.output_b[:],
            param_count=self.param_count,
        )

    def count_regions(
        self,
        resolution: int = 50,
        x_range: tuple[float, float] = (-1, 1),
        y_range: tuple[float, float] = (-1, 1),
    ) -> RegionInfo:
        """Count unique linear regions by sampling a grid."""
        signatures: dict[tuple[int, ...], list[tuple[float, float]]] = {}

        x_min, x_max = x_range
        y_min, y_max = y_range

        for i in range(resolution):
            for j in range(resolution):
                x1 = x_min + (x_max - x_min) * i / (resolution - 1)
                x2 = y_min + (y_max - y_min) * j / (resolution - 1)
                fwd = self.forward([x1, x2])
                sig = fwd.activation_signature
                if sig not in signatures:
                    signatures[sig] = []
                signatures[sig].append((x1, x2))

        # Theoretical maximum regions (Montufar et al. formula)
        # For input_dim=2, K hidden layers of width D:
        # N_r = (D/D_in + 1)^(D_in*(K-1)) * (D^2 + D + 2) / 2
        # Simplified for uniform width
        D_in = self.input_dim
        K = len(self.hidden_dims)
        D = self.hidden_dims[0] if self.hidden_dims else 1

        if K == 1:
            # Single layer: at most D+1 regions for 2D input (actually more complex)
            # Using simpler bound: sum of products
            theoretical_max = (D * D + D + 2) // 2
        else:
            base = (D / D_in + 1) ** (D_in * (K - 1))
            factor = (D * D + D + 2) / 2
            theoretical_max = int(base * factor)

        return RegionInfo(
            count=len(signatures),
            theoretical_max=theoretical_max,
            signatures=signatures,
        )

    def predict_batch(
        self,
        resolution: int = 50,
        x_range: tuple[float, float] = (-1, 1),
        y_range: tuple[float, float] = (-1, 1),
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Predict classes and region IDs for a grid.

        Returns:
            - predictions: 2D grid of predicted class labels
            - region_ids: 2D grid of region IDs (for tiling visualization)
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        predictions: list[list[int]] = []
        region_ids: list[list[int]] = []
        signature_to_id: dict[tuple[int, ...], int] = {}
        next_id = 0

        for j in range(resolution):
            pred_row: list[int] = []
            region_row: list[int] = []
            for i in range(resolution):
                x1 = x_min + (x_max - x_min) * i / (resolution - 1)
                x2 = y_min + (y_max - y_min) * j / (resolution - 1)
                fwd = self.forward([x1, x2])
                pred_row.append(fwd.prediction)

                sig = fwd.activation_signature
                if sig not in signature_to_id:
                    signature_to_id[sig] = next_id
                    next_id += 1
                region_row.append(signature_to_id[sig])

            predictions.append(pred_row)
            region_ids.append(region_row)

        return predictions, region_ids

