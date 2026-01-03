def test_mlp_state_step_reset(client):
    state = client.get("/mlp/state").json()
    assert state["dataset"] in ("xor", "or")
    assert state["hidden_dim"] == 2
    assert state["sample_count"] == 4

    reset = client.post("/mlp/reset", json={"dataset": "or", "hidden_dim": 3, "lr": 0.4, "seed": 1}).json()
    assert reset["dataset"] == "or"
    assert reset["hidden_dim"] == 3
    assert reset["lr"] == 0.4
    assert len(reset["hidden"]["templates"]) == 3

    step = client.post("/mlp/step", json={}).json()
    assert "step" in step
    assert step["step"]["dataset"] == "or"
    assert step["step"]["hidden_dim"] == 3
