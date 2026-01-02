import pytest

from backend.nn.activations import sigmoid, sigmoid_prime_from_output
from backend.nn.layers import DenseLayer


def test_dense_forward_backward_shapes():
    layer = DenseLayer(3, 2, activation=sigmoid, activation_prime_from_output=sigmoid_prime_from_output)
    out = layer.forward([1.0, 0.0, -1.0])
    assert len(out.a) == 2
    grad_W, grad_b, grad_x = layer.backward([0.1, -0.2])
    assert len(grad_W) == 2
    assert len(grad_W[0]) == 3
    assert len(grad_b) == 2
    assert len(grad_x) == 3


def test_dense_apply_gradients():
    layer = DenseLayer(2, 1, activation=sigmoid, activation_prime_from_output=sigmoid_prime_from_output)
    out = layer.forward([1.0, 1.0])
    grad_W, grad_b, _ = layer.backward([0.5])
    w_before = [row[:] for row in layer.W]
    layer.apply_gradients(grad_W, grad_b, lr=0.1)
    assert layer.W != w_before


def test_dense_requires_forward():
    layer = DenseLayer(2, 1, activation=sigmoid, activation_prime_from_output=sigmoid_prime_from_output)
    with pytest.raises(ValueError):
        layer.backward([0.1])
