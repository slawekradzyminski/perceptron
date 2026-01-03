from backend.core.datasets import make_xor_dataset_pm1
from backend.nn.mlp import MlpTwoLayer


def test_mlp_trains_on_xor():
    samples = make_xor_dataset_pm1()
    model = MlpTwoLayer(input_dim=2, hidden_dim=2, lr=0.5, seed=1)
    losses = model.train(samples, epochs=200)
    assert losses[-1] < losses[0]


def test_mlp_inspect_step_shapes_and_gradients():
    model = MlpTwoLayer(input_dim=2, hidden_dim=2, lr=0.1, seed=0)
    internals = model.inspect_step([1.0, -1.0], 1)
    assert len(internals.hidden_W_before) == 2
    assert len(internals.hidden_W_before[0]) == 2
    assert len(internals.out_W_before) == 1
    assert len(internals.out_W_before[0]) == 2
    assert len(internals.grad_hidden_W) == 2
    assert len(internals.grad_hidden_W[0]) == 2
    grad_sum = sum(abs(val) for row in internals.grad_hidden_W for val in row)
    assert grad_sum > 0.0
