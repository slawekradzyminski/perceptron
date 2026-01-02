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
        y01 = pm1_to_01(y_pm1)
        hidden_out = self.hidden.forward(x).a
        p_hat = self.output.forward(hidden_out).a[0]
        loss = bce_loss(float(y01), p_hat)
        grad_z_out = bce_grad_wrt_logit(p_hat, float(y01))
        grad_W_out, grad_b_out, grad_hidden = self.output.backward([grad_z_out])
        grad_W_hidden, grad_b_hidden, _ = self.hidden.backward(grad_hidden)
        self.output.apply_gradients(grad_W_out, grad_b_out, self.lr)
        self.hidden.apply_gradients(grad_W_hidden, grad_b_hidden, self.lr)
        self.last_grad_hidden = grad_W_hidden
        grad_norm = sum(abs(val) for row in grad_W_hidden for val in row) + sum(abs(val) for val in grad_b_hidden)
        return MlpStep(loss=loss, p_hat=p_hat, grad_norm=grad_norm, grad_hidden=grad_W_hidden)

    def train(self, samples: Sequence[dict], epochs: int) -> List[float]:
        losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for sample in samples:
                step = self.step(sample["x"], sample["y"])
                epoch_loss += step.loss
            losses.append(epoch_loss / len(samples))
        return losses
