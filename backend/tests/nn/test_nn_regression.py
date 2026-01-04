import math

from backend.nn.regression import backward, forward


def test_regression_mse_grads():
    x = 2.0
    y = 5.0
    m = 0.5
    b = -1.0
    step = forward(x, y, m, b, "mse")
    grads = backward(x, y, step.y_hat, "mse")
    expected_grad = 2 * (step.y_hat - y)
    assert math.isclose(grads.dL_db, expected_grad, rel_tol=1e-6)
    assert math.isclose(grads.dL_dm, expected_grad * x, rel_tol=1e-6)


def test_regression_l1_grads():
    x = 2.0
    y = 5.0
    m = 0.5
    b = -1.0
    step = forward(x, y, m, b, "l1")
    grads = backward(x, y, step.y_hat, "l1")
    assert grads.dL_db in (-1.0, 0.0, 1.0)
    assert math.isclose(grads.dL_dm, grads.dL_db * x, rel_tol=1e-6)
