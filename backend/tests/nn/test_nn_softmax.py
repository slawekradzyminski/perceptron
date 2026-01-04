import math

import pytest

from backend.nn.softmax import cross_entropy_from_logits, grad_logits_softmax_ce, softmax


def test_softmax_sums_to_one():
    probs = softmax([1.0, 2.0, 3.0])
    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 < p < 1.0 for p in probs)


def test_cross_entropy_from_logits():
    logits = [0.0, 0.0]
    loss = cross_entropy_from_logits(logits, 0)
    assert loss == pytest.approx(math.log(2.0))


def test_grad_logits_softmax_ce_matches_probs_minus_onehot():
    logits = [1.0, 0.0, -1.0]
    probs = softmax(logits)
    grads = grad_logits_softmax_ce(logits, 1)
    for idx, grad in enumerate(grads):
        expected = probs[idx] - (1.0 if idx == 1 else 0.0)
        assert grad == pytest.approx(expected)
