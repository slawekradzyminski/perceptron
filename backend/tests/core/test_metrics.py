from backend.core.metrics import accuracy, count_mistakes, margin, margins


def test_count_mistakes():
    results = [{"mistake": True}, {"mistake": False}, {"mistake": True}]
    assert count_mistakes(results) == 2


def test_accuracy():
    samples = [
        {"x": [1], "y": 1},
        {"x": [-1], "y": -1},
        {"x": [1], "y": 1},
    ]
    predict = lambda x: 1 if x[0] > 0 else -1
    assert accuracy(samples, predict) == 1.0


def test_margin_helpers():
    assert margin(1, 2.5) == 2.5
    assert margin(-1, 2.5) == -2.5
    samples = [{"x": [1], "y": 1}, {"x": [1], "y": -1}]
    score = lambda x: 0.5
    assert margins(samples, score) == [0.5, -0.5]
