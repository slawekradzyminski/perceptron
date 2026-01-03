import pytest

from backend.core.datasets import make_or_dataset_pm1
from backend.viz.viz_error_surface import mse_surface, plot_surface


def test_mse_surface_shape():
    samples = make_or_dataset_pm1()
    grid = mse_surface(samples, (-1.0, 1.0), steps=5, b=0.0)
    assert len(grid) == 5
    assert len(grid[0]) == 5


def test_mse_surface_validates_inputs():
    samples = make_or_dataset_pm1()
    with pytest.raises(ValueError):
        mse_surface(samples, (1.0, -1.0), steps=5)
    with pytest.raises(ValueError):
        mse_surface(samples, (-1.0, 1.0), steps=1)


def test_plot_surface_requires_matplotlib(monkeypatch):
    import backend.viz.viz_error_surface as mod

    monkeypatch.setattr(mod, "plt", None)
    with pytest.raises(RuntimeError):
        plot_surface([[0.0]])
