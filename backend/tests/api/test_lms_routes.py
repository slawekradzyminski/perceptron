def test_lms_state_step_reset(client):
    state = client.get("/lms/state").json()
    assert state["w"] == [0.0, 0.0]
    assert state["b"] == 0.0
    assert state["sample_count"] == 4
    assert state["dataset"] == "or"

    step = client.post("/lms/step", json={}).json()
    assert "w_before" in step and "w_after" in step
    assert "y_hat" in step and "error" in step
    assert "grad_w1" in step and "grad_w2" in step and "grad_b" in step

    reset = client.post("/lms/reset", json={"dataset": "xor", "lr": 0.2}).json()
    assert reset["w"] == [0.0, 0.0]
    assert reset["dataset"] == "xor"
    assert reset["lr"] == 0.2


def test_lms_custom_reset(client):
    custom = client.post(
        "/lms/reset",
        json={
            "dataset": "custom",
            "grid_rows": 1,
            "grid_cols": 2,
            "samples": [
                {"grid": [[-1, 1]], "y": -1},
                {"grid": [[1, 1]], "y": 1},
            ],
        },
    ).json()
    assert custom["dataset"] == "custom"
    assert custom["sample_count"] == 2
