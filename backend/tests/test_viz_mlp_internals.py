import pytest

from backend.viz import viz_mlp_internals


def test_plot_heatmap_requires_matplotlib(monkeypatch):
    monkeypatch.setattr(viz_mlp_internals, "plt", None)
    with pytest.raises(RuntimeError):
        viz_mlp_internals.plot_heatmap([[1.0]])


def test_plot_templates_requires_matplotlib(monkeypatch):
    monkeypatch.setattr(viz_mlp_internals, "plt", None)
    with pytest.raises(RuntimeError):
        viz_mlp_internals.plot_templates([[[1.0]]])
