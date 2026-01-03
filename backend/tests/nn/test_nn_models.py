import pytest

from backend.nn.models import Adaline, LogisticNeuron


def test_adaline_step_updates_weights():
    model = Adaline(dim=2, lr=0.5)
    step = model.step([1.0, -1.0], 1.0)
    assert step.loss > 0
    assert model.w != [0.0, 0.0]
    assert model.b != 0.0


def test_adaline_dimension_guard():
    model = Adaline(dim=2)
    with pytest.raises(ValueError):
        model.forward([1.0])


def test_logistic_step_updates_weights():
    model = LogisticNeuron(dim=2, lr=0.5)
    step = model.step([1.0, 1.0], 1)
    assert 0.0 <= step.p_hat <= 1.0
    assert step.loss >= 0.0
    assert model.w != [0.0, 0.0]


def test_logistic_dimension_guard():
    model = LogisticNeuron(dim=2)
    with pytest.raises(ValueError):
        model.logit([1.0])
