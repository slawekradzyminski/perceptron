from backend.datasets import make_xor_dataset_pm1
from backend.nn.mlp import MlpTwoLayer


def test_mlp_trains_on_xor():
    samples = make_xor_dataset_pm1()
    model = MlpTwoLayer(input_dim=2, hidden_dim=2, lr=0.5, seed=1)
    losses = model.train(samples, epochs=200)
    assert losses[-1] < losses[0]
