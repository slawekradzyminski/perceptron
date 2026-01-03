from backend.datasets import make_shape_dataset
from backend.nn.grid_mlp import GridMlp, reshape_template


def test_reshape_template():
    weights = [1, 2, 3, 4]
    grid = reshape_template(weights, 2, 2)
    assert grid == [[1, 2], [3, 4]]


def test_grid_mlp_templates_shape():
    model = GridMlp(rows=2, cols=2, hidden_dim=3, lr=0.1, seed=1)
    templates = model.weight_templates()
    assert len(templates) == 3
    assert len(templates[0]) == 2
    assert len(templates[0][0]) == 2


def test_grid_mlp_trains():
    good = [[1, 0], [0, 0]]
    bad = [[0, 0], [0, 1]]
    samples = make_shape_dataset(good, bad, board_size=(2, 2), translations=False)
    model = GridMlp(rows=2, cols=2, hidden_dim=2, lr=0.3, seed=2)
    losses = model.train(samples, epochs=5)
    assert len(losses) == 5


def test_grid_mlp_gradient_templates():
    good = [[1, 0], [0, 0]]
    bad = [[0, 0], [0, 1]]
    samples = make_shape_dataset(good, bad, board_size=(2, 2), translations=False)
    model = GridMlp(rows=2, cols=2, hidden_dim=2, lr=0.3, seed=2)
    model.step(samples[0]["x"], samples[0]["y"])
    grads = model.gradient_templates()
    assert grads is not None
    assert len(grads) == 2
    assert len(grads[0]) == 2
