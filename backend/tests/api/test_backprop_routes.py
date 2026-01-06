def test_backprop_state(client):
    state = client.get("/backprop/state").json()
    assert "tinygps" in state
    assert "regression" in state
    assert "tinygps_datasets" in state
    assert "tinygps_city_coords" in state


def test_backprop_tinygps_step_and_reset(client):
    reset = client.post("/backprop/tinygps/reset", json={"dataset": "madrid-paris-berlin", "lr": 0.1}).json()
    assert reset["dataset"] == "madrid-paris-berlin"
    assert reset["mode"] == "1d"
    step = client.post("/backprop/tinygps/step", json={}).json()
    assert "logits" in step and "probs" in step
    assert "grads" in step and "params_after" in step


def test_backprop_regression_step_and_reset(client):
    reset = client.post(
        "/backprop/regression/reset",
        json={"loss": "l1", "lr": 0.05, "samples": [{"x": 1.0, "y": 2.0}]},
    ).json()
    assert reset["loss"] == "l1"
    step = client.post("/backprop/regression/step", json={}).json()
    assert "y_hat" in step and "loss_value" in step
    assert "grads" in step and "params_after" in step
