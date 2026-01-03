def test_perceptron_state_step_reset(client):
    state = client.get("/state").json()
    assert state["dataset"] == "or"
    assert "next_x" in state and "next_y" in state
    assert state["grid_rows"] == 1
    assert state["grid_cols"] == 2

    step = client.post("/step", json={"lr": 0.5}).json()
    assert "w" in step and "b" in step
    assert step["x"] in ([-1, -1], [-1, 1], [1, -1], [1, 1])
    assert step["y"] in (-1, 1)
    assert "delta_w" in step and "delta_b" in step
    assert step["lr"] == 0.5
    assert "next_x" in step and "next_y" in step
    assert step["grid_rows"] == 1
    assert step["grid_cols"] == 2

    reset = client.post("/reset", json={"dataset": "xor", "lr": 1.2}).json()
    assert reset["dataset"] == "xor"
    assert reset["lr"] == 1.2
    assert "next_x" in reset and "next_y" in reset
    assert reset["grid_rows"] == 1
    assert reset["grid_cols"] == 2


def test_perceptron_custom_reset(client):
    custom = client.post(
        "/reset",
        json={
            "dataset": "custom",
            "grid_rows": 2,
            "grid_cols": 2,
            "samples": [
                {"grid": [[-1, 1], [1, -1]], "y": 1},
                {"grid": [[1, 1], [1, 1]], "y": -1},
            ],
        },
    ).json()
    assert custom["dataset"] == "custom"
    assert custom["grid_rows"] == 2
    assert custom["grid_cols"] == 2
    assert custom["sample_count"] == 2
