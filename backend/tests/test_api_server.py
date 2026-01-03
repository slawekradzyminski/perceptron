import json
import threading
import urllib.request

from backend.api_server import create_server


def _request_json(url: str, method: str = "GET", data: dict | None = None):
    payload = None
    headers = {}
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_api_state_and_step():
    server = create_server("127.0.0.1", 0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f"http://{host}:{port}"
    state = _request_json(f"{base}/state")
    assert state["dataset"] == "or"
    assert "next_x" in state and "next_y" in state
    assert state["grid_rows"] == 1
    assert state["grid_cols"] == 2

    step = _request_json(f"{base}/step", method="POST", data={"lr": 0.5})
    assert "w" in step and "b" in step
    assert step["x"] in ([-1, -1], [-1, 1], [1, -1], [1, 1])
    assert step["y"] in (-1, 1)
    assert "delta_w" in step and "delta_b" in step
    assert step["lr"] == 0.5
    assert "next_x" in step and "next_y" in step
    assert step["grid_rows"] == 1
    assert step["grid_cols"] == 2

    reset = _request_json(f"{base}/reset", method="POST", data={"dataset": "xor", "lr": 1.2})
    assert reset["dataset"] == "xor"
    assert reset["lr"] == 1.2
    assert "next_x" in reset and "next_y" in reset
    assert reset["grid_rows"] == 1
    assert reset["grid_cols"] == 2

    custom = _request_json(
        f"{base}/reset",
        method="POST",
        data={
            "dataset": "custom",
            "grid_rows": 2,
            "grid_cols": 2,
            "samples": [
                {"grid": [[-1, 1], [1, -1]], "y": 1},
                {"grid": [[1, 1], [1, 1]], "y": -1},
            ],
        },
    )
    assert custom["dataset"] == "custom"
    assert custom["grid_rows"] == 2
    assert custom["grid_cols"] == 2
    assert custom["sample_count"] == 2

    server.shutdown()
    server.server_close()


def test_api_error_surface_and_mlp_internals():
    server = create_server("127.0.0.1", 0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f"http://{host}:{port}"
    surface = _request_json(
        f"{base}/error-surface",
        method="POST",
        data={"dataset": "or", "steps": 5, "w_min": -1.0, "w_max": 1.0, "b": 0.0},
    )
    assert surface["dataset"] == "or"
    assert surface["steps"] == 5
    assert surface["grid_rows"] == 1
    assert surface["grid_cols"] == 2
    assert surface["sample_count"] == 4
    assert len(surface["grid"]) == 5
    assert all(len(row) == 5 for row in surface["grid"])

    internals = _request_json(
        f"{base}/mlp-internals",
        method="POST",
        data={"dataset": "xor", "hidden_dim": 3, "sample_index": 2, "lr": 0.5, "seed": 1},
    )
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

    server.shutdown()
    server.server_close()
