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


class _FakeAxis:
    def __init__(self):
        self.imshow_calls = 0
        self.title = None
        self.axis_calls = []

    def imshow(self, _grid, cmap=None):
        self.imshow_calls += 1

    def set_title(self, title):
        self.title = title

    def axis(self, value):
        self.axis_calls.append(value)


class _FakeAxesArray:
    def __init__(self, axes):
        self._axes = axes

    def flatten(self):
        return self._axes


class _FakeFig:
    def __init__(self):
        self.tight_layout_called = False

    def tight_layout(self):
        self.tight_layout_called = True


class _FakePlt:
    def __init__(self):
        self.imshow_calls = 0
        self.colorbar_calls = 0
        self.title_calls = []
        self.subplots_args = None
        self._axes = []
        self._fig = _FakeFig()

    def imshow(self, _grid, cmap=None):
        self.imshow_calls += 1

    def colorbar(self):
        self.colorbar_calls += 1

    def title(self, title):
        self.title_calls.append(title)

    def subplots(self, rows, cols, figsize=None):
        self.subplots_args = (rows, cols, figsize)
        count = rows * cols
        self._axes = [_FakeAxis() for _ in range(count)]
        return self._fig, _FakeAxesArray(self._axes)


def test_plot_heatmap_draws_with_title(monkeypatch):
    fake = _FakePlt()
    monkeypatch.setattr(viz_mlp_internals, "plt", fake)
    viz_mlp_internals.plot_heatmap([[1.0, -1.0]], title="Example")
    assert fake.imshow_calls == 1
    assert fake.colorbar_calls == 1
    assert fake.title_calls == ["Example"]


def test_plot_templates_draws_and_hides_unused_axes(monkeypatch):
    fake = _FakePlt()
    monkeypatch.setattr(viz_mlp_internals, "plt", fake)
    templates = [
        [[1.0, -1.0]],
        [[-1.0, 1.0]],
        [[1.0, 1.0]],
        [[-1.0, -1.0]],
        [[0.5, -0.5]],
    ]
    viz_mlp_internals.plot_templates(templates, title_prefix="Hidden")
    rows, cols, _ = fake.subplots_args
    assert cols == 4
    assert rows == 2
    assert fake._fig.tight_layout_called
    used = fake._axes[: len(templates)]
    unused = fake._axes[len(templates) :]
    assert all(axis.imshow_calls == 1 for axis in used)
    assert all("off" in axis.axis_calls for axis in used)
    assert all("off" in axis.axis_calls for axis in unused)
