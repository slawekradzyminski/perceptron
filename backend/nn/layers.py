"""Dense layers with activation support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence
import random


@dataclass
class DenseStep:
    z: List[float]
    a: List[float]


class DenseLayer:
    """Simple dense layer with activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable[[float], float],
        activation_prime_from_output: Callable[[float], float],
        seed: int | None = 0,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        rng = random.Random(seed)
        self.W = [[rng.uniform(-0.5, 0.5) for _ in range(input_dim)] for _ in range(output_dim)]
        self.b = [0.0 for _ in range(output_dim)]
        self.activation = activation
        self.activation_prime_from_output = activation_prime_from_output
        self.last_input: List[float] | None = None
        self.last_output: List[float] | None = None

    def forward(self, x: Sequence[float]) -> DenseStep:
        if len(x) != self.input_dim:
            raise ValueError("x has wrong dimension")
        self.last_input = list(x)
        z: List[float] = []
        a: List[float] = []
        for i in range(self.output_dim):
            z_i = sum(self.W[i][j] * x[j] for j in range(self.input_dim)) + self.b[i]
            a_i = self.activation(z_i)
            z.append(z_i)
            a.append(a_i)
        self.last_output = a
        return DenseStep(z=z, a=a)

    def backward(self, grad_out: Sequence[float]) -> tuple[List[List[float]], List[float], List[float]]:
        if self.last_input is None or self.last_output is None:
            raise ValueError("forward must be called before backward")
        if len(grad_out) != self.output_dim:
            raise ValueError("grad_out has wrong dimension")
        grad_z: List[float] = []
        for i in range(self.output_dim):
            grad_z.append(grad_out[i] * self.activation_prime_from_output(self.last_output[i]))
        grad_W: List[List[float]] = [
            [grad_z[i] * self.last_input[j] for j in range(self.input_dim)]
            for i in range(self.output_dim)
        ]
        grad_b: List[float] = grad_z[:]
        grad_x: List[float] = [0.0 for _ in range(self.input_dim)]
        for j in range(self.input_dim):
            grad_x[j] = sum(self.W[i][j] * grad_z[i] for i in range(self.output_dim))
        return grad_W, grad_b, grad_x

    def apply_gradients(self, grad_W: Sequence[Sequence[float]], grad_b: Sequence[float], lr: float) -> None:
        if len(grad_W) != self.output_dim:
            raise ValueError("grad_W has wrong dimension")
        if len(grad_b) != self.output_dim:
            raise ValueError("grad_b has wrong dimension")
        for i in range(self.output_dim):
            if len(grad_W[i]) != self.input_dim:
                raise ValueError("grad_W has wrong dimension")
            for j in range(self.input_dim):
                self.W[i][j] -= lr * grad_W[i][j]
            self.b[i] -= lr * grad_b[i]
