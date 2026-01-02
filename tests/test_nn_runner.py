from backend.datasets import make_or_dataset_pm1
from backend.nn.runner import train_adaline, train_logistic


def test_train_adaline_returns_losses():
    samples = make_or_dataset_pm1()
    losses = train_adaline(samples, epochs=3, lr=0.1)
    assert len(losses) == 3


def test_train_logistic_returns_losses():
    samples = make_or_dataset_pm1()
    losses = train_logistic(samples, epochs=2, lr=0.2)
    assert len(losses) == 2
