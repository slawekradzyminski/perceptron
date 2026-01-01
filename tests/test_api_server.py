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

    step = _request_json(f"{base}/step", method="POST", data={"lr": 0.5})
    assert "w" in step and "b" in step
    assert step["x"] in ([-1, -1], [-1, 1], [1, -1], [1, 1])
    assert step["y"] in (-1, 1)
    assert "delta_w" in step and "delta_b" in step
    assert step["lr"] == 0.5
    assert "next_x" in step and "next_y" in step

    reset = _request_json(f"{base}/reset", method="POST", data={"dataset": "xor", "lr": 1.2})
    assert reset["dataset"] == "xor"
    assert reset["lr"] == 1.2
    assert "next_x" in reset and "next_y" in reset

    server.shutdown()
    server.server_close()
