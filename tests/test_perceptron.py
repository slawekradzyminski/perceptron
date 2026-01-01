import math
import pytest

from backend.perceptron import Perceptron


def test_predict_score_and_label_zero_init():
    p = Perceptron(dim=2, lr=1.0, seed=0, init="zeros")
    assert p.predict_score([0.0, 0.0]) == 0.0
    assert p.predict_label([0.0, 0.0]) == 1


def test_train_step_updates_on_mistake():
    p = Perceptron(dim=2, lr=1.0, seed=0, init="zeros")
    # With zero weights, any negative label is a mistake (score=0 => y*score<=0)
    result = p.train_step([1.0, -1.0], y=-1)
    assert result.mistake is True
    assert result.delta_w == [-1.0, 1.0]
    assert result.delta_b == -1.0
    assert p.w == [-1.0, 1.0]
    assert p.b == -1.0


def test_train_step_no_update_when_correct():
    p = Perceptron(dim=2, lr=1.0, seed=0, init="zeros")
    # Make it predict +1 correctly
    p.w = [1.0, 1.0]
    p.b = 0.5
    result = p.train_step([1.0, 1.0], y=1)
    assert result.mistake is False
    assert result.delta_w == [0.0, 0.0]
    assert result.delta_b == 0.0
    assert p.w == [1.0, 1.0]
    assert p.b == 0.5


def test_random_init_is_deterministic_with_seed():
    p1 = Perceptron(dim=3, lr=1.0, seed=123, init="random")
    p2 = Perceptron(dim=3, lr=1.0, seed=123, init="random")
    assert p1.w == p2.w
    assert math.isclose(p1.b, p2.b)


def test_invalid_label_raises():
    p = Perceptron(dim=1)
    with pytest.raises(ValueError):
        p.train_step([0.0], y=0)
