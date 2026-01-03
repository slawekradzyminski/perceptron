def test_error_surface(client):
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


def test_mlp_internals(client):
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
