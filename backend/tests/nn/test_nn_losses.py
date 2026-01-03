import pytest

from backend.nn.losses import bce_grad_wrt_logit, bce_loss, mse_grad, mse_loss, perceptron_loss, pm1_to_01


def test_perceptron_loss():
    assert perceptron_loss(1, 2.0) == 0.0
    assert perceptron_loss(-1, -0.5) == 0.0
    assert perceptron_loss(1, -0.5) == pytest.approx(0.5)


def test_mse():
    assert mse_loss(1.0, 0.0) == pytest.approx(0.5)
    assert mse_grad(1.0, 0.0) == pytest.approx(-1.0)


def test_bce_ordering():
    assert bce_loss(1.0, 0.9) < bce_loss(1.0, 0.1)
    assert bce_loss(0.0, 0.1) < bce_loss(0.0, 0.9)


def test_bce_grad_logit():
    assert bce_grad_wrt_logit(0.7, 1.0) == pytest.approx(-0.3)
    assert bce_grad_wrt_logit(0.2, 0.0) == pytest.approx(0.2)


def test_pm1_to_01():
    assert pm1_to_01(1) == 1
    assert pm1_to_01(-1) == 0
    with pytest.raises(ValueError):
        pm1_to_01(0)
