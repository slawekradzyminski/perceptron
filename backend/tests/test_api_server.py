from fastapi.testclient import TestClient

from backend.api_app import app


client = TestClient(app)


def test_api_state_and_step():
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


def test_api_error_surface_and_mlp_internals():
    surface = client.post(
        "/error-surface",
        json={"dataset": "or", "steps": 5, "w_min": -1.0, "w_max": 1.0, "b": 0.0},
    ).json()
    assert surface["dataset"] == "or"
    assert surface["steps"] == 5
    assert surface["grid_rows"] == 1
    assert surface["grid_cols"] == 2
    assert surface["sample_count"] == 4
    assert len(surface["grid"]) == 5
    assert all(len(row) == 5 for row in surface["grid"])

    internals = client.post(
        "/mlp-internals",
        json={"dataset": "xor", "hidden_dim": 3, "sample_index": 2, "lr": 0.5, "seed": 1},
    ).json()
    assert internals["dataset"] == "xor"
    assert internals["grid_rows"] == 1
    assert internals["grid_cols"] == 2
    assert internals["hidden_dim"] == 3
    assert internals["sample_count"] == 4
    hidden = internals["hidden"]
    output = internals["output"]
    assert len(hidden["weights_before"]) == 3
    assert len(hidden["weights_before"][0]) == 2
    assert len(hidden["templates_before"]) == 3
    assert len(hidden["templates_before"][0]) == 1
    assert len(hidden["templates_before"][0][0]) == 2
    assert len(output["weights_before"]) == 1
    assert len(output["weights_before"][0]) == 3
    gradients = internals["gradients"]
    assert len(gradients["hidden_W"]) == 3
    assert len(gradients["hidden_W"][0]) == 2
    assert len(gradients["templates"]) == 3



def test_api_lms_step_and_state():
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
