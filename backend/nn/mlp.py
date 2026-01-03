"""Two-layer MLP for XOR demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from backend.nn.activations import sigmoid, sigmoid_prime_from_output, tanh, tanh_prime_from_output
from backend.nn.layers import DenseLayer
from backend.nn.losses import bce_grad_wrt_logit, bce_loss, pm1_to_01


@dataclass
class MlpStep:
    loss: float
    p_hat: float
    grad_norm: float
    grad_hidden: List[List[float]] | None = None


@dataclass
class MlpInternals:
    x: List[float]
    y: int
    y01: int
    hidden_z: List[float]
    hidden_a: List[float]
    output_z: float
    output_a: float
    loss: float
    grad_hidden_W: List[List[float]]
    grad_hidden_b: List[float]
    grad_out_W: List[List[float]]
    grad_out_b: List[float]
    hidden_W_before: List[List[float]]
    hidden_b_before: List[float]
    out_W_before: List[List[float]]
    out_b_before: List[float]
    hidden_W_after: List[List[float]]
    hidden_b_after: List[float]
    out_W_after: List[List[float]]
    out_b_after: List[float]


class MlpTwoLayer:
    """Minimal 2-layer MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 2, lr: float = 0.5, seed: int | None = 0) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hidden = DenseLayer(
            input_dim,
            hidden_dim,
            activation=tanh,
            activation_prime_from_output=tanh_prime_from_output,
            seed=seed,
        )
        self.output = DenseLayer(
            hidden_dim,
            1,
            activation=sigmoid,
            activation_prime_from_output=sigmoid_prime_from_output,
            seed=seed,
        )
        self.last_grad_hidden: List[List[float]] | None = None

    def forward(self, x: Sequence[float]) -> float:
        hidden = self.hidden.forward(x).a
        out = self.output.forward(hidden).a[0]
        return out

    def step(self, x: Sequence[float], y_pm1: int) -> MlpStep:
        internals = self.inspect_step(x, y_pm1)
        grad_norm = sum(abs(val) for row in internals.grad_hidden_W for val in row) + sum(
            abs(val) for val in internals.grad_hidden_b
        )
        return MlpStep(
            loss=internals.loss,
            p_hat=internals.output_a,
            grad_norm=grad_norm,
            grad_hidden=internals.grad_hidden_W,
        )

    def inspect_step(self, x: Sequence[float], y_pm1: int) -> MlpInternals:
        y01 = pm1_to_01(y_pm1)
        hidden_W_before = [row[:] for row in self.hidden.W]
        hidden_b_before = self.hidden.b[:]
        out_W_before = [row[:] for row in self.output.W]
        out_b_before = self.output.b[:]

        hidden_step = self.hidden.forward(x)
        output_step = self.output.forward(hidden_step.a)
        p_hat = output_step.a[0]
        loss = bce_loss(float(y01), p_hat)

        grad_z_out = bce_grad_wrt_logit(p_hat, float(y01))
        grad_W_out, grad_b_out, grad_hidden = self.output.backward([grad_z_out])
        grad_W_hidden, grad_b_hidden, _ = self.hidden.backward(grad_hidden)

        self.output.apply_gradients(grad_W_out, grad_b_out, self.lr)
        self.hidden.apply_gradients(grad_W_hidden, grad_b_hidden, self.lr)
        self.last_grad_hidden = grad_W_hidden

        return MlpInternals(
            x=list(x),
            y=y_pm1,
            y01=y01,
            hidden_z=hidden_step.z,
            hidden_a=hidden_step.a,
            output_z=output_step.z[0],
            output_a=p_hat,
            loss=loss,
            grad_hidden_W=grad_W_hidden,
            grad_hidden_b=grad_b_hidden,
            grad_out_W=grad_W_out,
            grad_out_b=grad_b_out,
            hidden_W_before=hidden_W_before,
            hidden_b_before=hidden_b_before,
            out_W_before=out_W_before,
            out_b_before=out_b_before,
            hidden_W_after=[row[:] for row in self.hidden.W],
            hidden_b_after=self.hidden.b[:],
            out_W_after=[row[:] for row in self.output.W],
            out_b_after=self.output.b[:],
        )

    def train(self, samples: Sequence[dict], epochs: int) -> List[float]:
        losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for sample in samples:
                step = self.step(sample["x"], sample["y"])
                epoch_loss += step.loss
            losses.append(epoch_loss / len(samples))
        return losses
