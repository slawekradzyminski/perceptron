import pytest

from backend.runner import run_training


def test_run_training_or_converges():
    stats = run_training("or", epochs=5, lr=1.0, seed=0)
    assert len(stats["mistakes"]) == 5
    assert len(stats["accuracy"]) == 5
    assert stats["accuracy"][-1] >= 0.75


def test_run_training_invalid_dataset():
    with pytest.raises(ValueError):
        run_training("bad", epochs=1, lr=1.0)
