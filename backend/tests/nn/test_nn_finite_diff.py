import pytest

from backend.nn.finite_diff import finite_diff_grad


def test_finite_diff_matches_quadratic():
    def quad(theta):
        return (theta[0] - 2.0) ** 2 + 3.0 * (theta[1] + 1.0) ** 2

    grads = finite_diff_grad(quad, [0.0, 0.0])
    # analytic gradient: [2*(x-2), 6*(y+1)]
    assert grads[0] == pytest.approx(-4.0, rel=1e-4)
    assert grads[1] == pytest.approx(6.0, rel=1e-4)
