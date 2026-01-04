"""Single-neuron regression helpers for MSE and L1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.nn.losses import l1_grad, l1_loss

LossType = Literal["mse", "l1"]


@dataclass
class RegressionForward:
    y_hat: float
    loss: float


@dataclass
class RegressionGrads:
    dL_dm: float
    dL_db: float


def forward(x: float, y: float, m: float, b: float, loss_type: LossType) -> RegressionForward:
    y_hat = m * x + b
    if loss_type == "mse":
        diff = y_hat - y
        loss = diff * diff
    elif loss_type == "l1":
        loss = l1_loss(y, y_hat)
    else:
        raise ValueError("loss_type must be 'mse' or 'l1'")
    return RegressionForward(y_hat=y_hat, loss=loss)


def backward(x: float, y: float, y_hat: float, loss_type: LossType) -> RegressionGrads:
    if loss_type == "mse":
        grad_y_hat = 2.0 * (y_hat - y)
    elif loss_type == "l1":
        grad_y_hat = l1_grad(y, y_hat)
    else:
        raise ValueError("loss_type must be 'mse' or 'l1'")
    return RegressionGrads(dL_dm=grad_y_hat * x, dL_db=grad_y_hat)
