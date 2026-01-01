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

    step = _request_json(f"{base}/step", method="POST", data={})
    assert "w" in step and "b" in step

    reset = _request_json(f"{base}/reset", method="POST", data={"dataset": "xor"})
    assert reset["dataset"] == "xor"

    server.shutdown()
    server.server_close()
