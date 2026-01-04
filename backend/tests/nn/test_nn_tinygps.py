import math

from backend.nn.finite_diff import finite_diff_grad
from backend.nn.tinygps import backward_1d, backward_2d, forward_1d, forward_2d


def test_tinygps_backward_1d_matches_finite_diff():
    x = 2.0
    m = [0.3, -0.7]
    b = [0.1, -0.2]
    target = 1

    forward = forward_1d(x, m, b, target)
    grads = backward_1d(x, forward.probs, target)
    analytic = grads.dL_dm + grads.dL_db

    def loss_fn(theta):
        m_vec = theta[:2]
        b_vec = theta[2:]
        return forward_1d(x, m_vec, b_vec, target).loss

    numeric = finite_diff_grad(loss_fn, m + b)
    for a, n in zip(analytic, numeric):
        assert math.isclose(a, n, rel_tol=1e-4, abs_tol=1e-4)


def test_tinygps_backward_2d_matches_finite_diff():
    x = [2.0, -1.0]
    M = [[0.2, -0.1], [0.5, 0.3], [-0.4, 0.25]]
    b = [0.05, -0.2, 0.1]
    target = 0

    forward = forward_2d(x, M, b, target)
    grads = backward_2d(x, forward.probs, target)
    analytic = [val for row in grads.dL_dM for val in row] + grads.dL_db

    def loss_fn(theta):
        m_rows = [theta[0:2], theta[2:4], theta[4:6]]
        b_vec = theta[6:]
        return forward_2d(x, m_rows, b_vec, target).loss

    flat_params = [val for row in M for val in row] + b
    numeric = finite_diff_grad(loss_fn, flat_params)
    for a, n in zip(analytic, numeric):
        assert math.isclose(a, n, rel_tol=1e-4, abs_tol=1e-4)
