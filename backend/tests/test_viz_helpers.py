import pytest

from backend.viz_2d import decision_boundary_points
from backend.viz_grid import contribution_grid, score_from_grid


def test_decision_boundary_points_basic():
    (x1, y1), (x2, y2) = decision_boundary_points([1.0, 1.0], b=0.0, x_min=-1, x_max=1)
    assert x1 == -1 and x2 == 1
    assert y1 == 1
    assert y2 == -1


def test_decision_boundary_vertical():
    (x1, y1), (x2, y2) = decision_boundary_points([2.0, 0.0], b=-4.0, x_min=-1, x_max=1)
    assert x1 == x2 == 2.0
    assert y1 == -1 and y2 == 1


def test_contribution_grid_and_score():
    x = [[1, -1], [-1, 1]]
    w = [[2, 3], [4, 5]]
    contrib = contribution_grid(x, w)
    assert contrib == [[2, -3], [-4, 5]]
    assert score_from_grid(x, w, b=1.0) == 1.0


def test_decision_boundary_errors():
    with pytest.raises(ValueError):
        decision_boundary_points([0.0, 0.0], b=0.0)
    with pytest.raises(ValueError):
        decision_boundary_points([1.0], b=0.0)
