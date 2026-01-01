"""Minimal JSON API server for perceptron stepping (stdlib only)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from backend.datasets import make_or_dataset_pm1, make_xor_dataset_pm1
from backend.perceptron import Perceptron


class PerceptronService:
    def __init__(self, dataset: str = "or", lr: float = 1.0, seed: int | None = 0) -> None:
        self.lr = lr
        self.seed = seed
        self.set_dataset(dataset)

    def set_lr(self, lr: float) -> None:
        self.lr = lr
        self.perceptron.lr = lr

    def set_dataset(self, name: str) -> None:
        if name == "or":
            self.samples = make_or_dataset_pm1()
        elif name == "xor":
            self.samples = make_xor_dataset_pm1()
        else:
            raise ValueError("dataset must be 'or' or 'xor'")
        self.dataset = name
        self.idx = 0
        self.perceptron = Perceptron(dim=2, lr=self.lr, seed=self.seed, init="zeros")

    def step(self) -> Dict[str, Any]:
        sample = self.samples[self.idx]
        result = self.perceptron.train_step(sample["x"], sample["y"], lr=self.lr)
        self.idx = (self.idx + 1) % len(self.samples)
        return {
            "w": self.perceptron.w,
            "b": self.perceptron.b,
            "x": sample["x"],
            "y": sample["y"],
            "score": result.score,
            "pred": result.pred,
            "mistake": result.mistake,
            "delta_w": result.delta_w,
            "delta_b": result.delta_b,
            "idx": self.idx,
            "dataset": self.dataset,
            "lr": self.lr,
        }

    def reset(self) -> Dict[str, Any]:
        self.set_dataset(self.dataset)
        return self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "w": self.perceptron.w,
            "b": self.perceptron.b,
            "idx": self.idx,
            "dataset": self.dataset,
            "lr": self.lr,
        }


class RequestHandler(BaseHTTPRequestHandler):
    service = PerceptronService()

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def _parse_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path == "/state":
            self._send_json(self.service.state())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        body = self._parse_body()
        if "lr" in body:
            try:
                self.service.set_lr(float(body["lr"]))
            except (TypeError, ValueError):
                self._send_json({"error": "lr must be a number"}, status=400)
                return
        if self.path == "/reset":
            dataset = body.get("dataset")
            if dataset:
                try:
                    self.service.set_dataset(dataset)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
            self._send_json(self.service.reset())
            return
        if self.path == "/step":
            dataset = body.get("dataset")
            if dataset and dataset != self.service.dataset:
                try:
                    self.service.set_dataset(dataset)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
            self._send_json(self.service.step())
            return
        self._send_json({"error": "not found"}, status=404)


def create_server(host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    return HTTPServer((host, port), RequestHandler)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = create_server(host, port)
    print(f"Perceptron API running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
