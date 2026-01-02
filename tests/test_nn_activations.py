import math

import pytest

from backend.nn.activations import relu, relu_prime, sigmoid, sigmoid_prime_from_output, step_pm1, tanh, tanh_prime_from_output


def test_step_pm1():
    assert step_pm1(0.0) == 1
    assert step_pm1(-0.1) == -1


def test_sigmoid_basic():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_sigmoid_prime_from_output():
    p = sigmoid(0.0)
    assert sigmoid_prime_from_output(p) == pytest.approx(0.25)


def test_tanh_basic():
    assert tanh(0.0) == pytest.approx(0.0)
    assert tanh(3.0) == pytest.approx(math.tanh(3.0))


def test_tanh_prime_from_output():
    t = tanh(0.4)
    assert tanh_prime_from_output(t) == pytest.approx(1.0 - t * t)


def test_relu():
    assert relu(-1.0) == 0.0
    assert relu(2.0) == 2.0
    assert relu_prime(-1.0) == 0.0
    assert relu_prime(0.0) == 0.0
    assert relu_prime(1.0) == 1.0
