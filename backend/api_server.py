"""Minimal JSON API server for perceptron stepping (stdlib only)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterable, List, Tuple

from backend.datasets import make_or_dataset_pm1, make_xor_dataset_pm1
from backend.nn.grid_mlp import GridMlp, reshape_template
from backend.nn.mlp import MlpInternals
from backend.viz.viz_error_surface import mse_surface
from backend.perceptron import Perceptron


def _validate_grid_shape(rows: Any, cols: Any) -> Tuple[int, int]:
    try:
        r = int(rows)
        c = int(cols)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_rows and grid_cols must be integers") from exc
    if r < 1 or c < 1 or r > 5 or c > 5:
        raise ValueError("grid_rows and grid_cols must be between 1 and 5")
    return r, c


def _normalize_samples(
    samples: Iterable[Dict[str, Any]],
    rows: int,
    cols: int,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for sample in samples:
        y = sample.get("y")
        if y not in (-1, 1):
            raise ValueError("each sample y must be -1 or +1")
        grid = sample.get("grid")
        x = sample.get("x")
        if grid is not None:
            if not isinstance(grid, list) or len(grid) != rows:
                raise ValueError("grid rows must match grid_rows")
            flat: List[float] = []
            for row in grid:
                if not isinstance(row, list) or len(row) != cols:
                    raise ValueError("grid cols must match grid_cols")
                for cell in row:
                    if cell not in (-1, 1):
                        raise ValueError("grid values must be -1 or +1")
                    flat.append(float(cell))
            x = flat
        if x is None:
            raise ValueError("each sample must include grid or x")
        if not isinstance(x, list) or len(x) != rows * cols:
            raise ValueError("x length must match grid_rows * grid_cols")
        for cell in x:
            if cell not in (-1, 1):
                raise ValueError("x values must be -1 or +1")
        normalized.append({"x": [float(val) for val in x], "y": int(y)})
    if not normalized:
        raise ValueError("samples must be non-empty")
    return normalized


def _load_samples_from_body(body: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Tuple[int, int]]:
    dataset = body.get("dataset", "or")
    if dataset == "or":
        return "or", make_or_dataset_pm1(), (1, 2)
    if dataset == "xor":
        return "xor", make_xor_dataset_pm1(), (1, 2)
    if dataset == "custom":
        rows, cols = _validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
        samples = _normalize_samples(body.get("samples", []), rows, cols)
        return "custom", samples, (rows, cols)
    raise ValueError("dataset must be 'or', 'xor', or 'custom'")


def _build_mlp_payload(
    internals: MlpInternals,
    rows: int,
    cols: int,
    hidden_dim: int,
    sample_index: int,
    sample_count: int,
    dataset: str,
) -> Dict[str, Any]:
    templates_before = [reshape_template(row, rows, cols) for row in internals.hidden_W_before]
    templates_after = [reshape_template(row, rows, cols) for row in internals.hidden_W_after]
    gradient_templates = [reshape_template(row, rows, cols) for row in internals.grad_hidden_W]
    return {
        "dataset": dataset,
        "grid_rows": rows,
        "grid_cols": cols,
        "hidden_dim": hidden_dim,
        "sample_index": sample_index,
        "sample_count": sample_count,
        "x": internals.x,
        "y": internals.y,
        "y01": internals.y01,
        "loss": internals.loss,
        "p_hat": internals.output_a,
        "hidden": {
            "weights_before": internals.hidden_W_before,
            "bias_before": internals.hidden_b_before,
            "weights_after": internals.hidden_W_after,
            "bias_after": internals.hidden_b_after,
            "z": internals.hidden_z,
            "a": internals.hidden_a,
            "templates_before": templates_before,
            "templates_after": templates_after,
        },
        "output": {
            "weights_before": internals.out_W_before,
            "bias_before": internals.out_b_before,
            "weights_after": internals.out_W_after,
            "bias_after": internals.out_b_after,
            "z": internals.output_z,
            "a": internals.output_a,
        },
        "gradients": {
            "hidden_W": internals.grad_hidden_W,
            "hidden_b": internals.grad_hidden_b,
            "out_W": internals.grad_out_W,
            "out_b": internals.grad_out_b,
            "templates": gradient_templates,
        },
    }


class PerceptronService:
    def __init__(self, dataset: str = "or", lr: float = 1.0, seed: int | None = 0) -> None:
        self.lr = lr
        self.seed = seed
        self.custom_samples: List[Dict[str, Any]] | None = None
        self.custom_shape: Tuple[int, int] | None = None
        self.set_dataset(dataset)

    def set_lr(self, lr: float) -> None:
        self.lr = lr
        self.perceptron.lr = lr

    def set_dataset(self, name: str, custom: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None) -> None:
        if name == "or":
            self.samples = make_or_dataset_pm1()
            grid_shape = (1, 2)
        elif name == "xor":
            self.samples = make_xor_dataset_pm1()
            grid_shape = (1, 2)
        elif name == "custom":
            if custom is not None:
                self.custom_samples, self.custom_shape = custom
            if self.custom_samples is None or self.custom_shape is None:
                raise ValueError("custom dataset requires samples and grid size")
            self.samples = self.custom_samples
            grid_shape = self.custom_shape
        else:
            raise ValueError("dataset must be 'or', 'xor', or 'custom'")
        self.dataset = name
        self.idx = 0
        self.grid_rows, self.grid_cols = grid_shape
        self.perceptron = Perceptron(dim=self.grid_rows * self.grid_cols, lr=self.lr, seed=self.seed, init="zeros")

    def step(self) -> Dict[str, Any]:
        sample = self.samples[self.idx]
        result = self.perceptron.train_step(sample["x"], sample["y"], lr=self.lr)
        self.idx = (self.idx + 1) % len(self.samples)
        next_sample = self.samples[self.idx]
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
            "next_x": next_sample["x"],
            "next_y": next_sample["y"],
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "sample_count": len(self.samples),
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
            "next_x": self.samples[self.idx]["x"],
            "next_y": self.samples[self.idx]["y"],
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "sample_count": len(self.samples),
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
        custom_payload: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None
        if body.get("dataset") == "custom" and "samples" in body:
            try:
                rows, cols = _validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
                samples = _normalize_samples(body.get("samples", []), rows, cols)
                custom_payload = (samples, (rows, cols))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
        if self.path == "/reset":
            dataset = body.get("dataset")
            if dataset:
                try:
                    self.service.set_dataset(dataset, custom=custom_payload)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
            self._send_json(self.service.reset())
            return
        if self.path == "/error-surface":
            try:
                dataset, samples, grid_shape = _load_samples_from_body(body)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            rows, cols = grid_shape
            if rows * cols != 2:
                self._send_json({"error": "error surface requires 2D inputs (grid_rows * grid_cols == 2)"}, status=400)
                return
            try:
                steps = int(body.get("steps", 25))
                w_min = float(body.get("w_min", -2.0))
                w_max = float(body.get("w_max", 2.0))
                b = float(body.get("b", 0.0))
            except (TypeError, ValueError):
                self._send_json({"error": "steps, w_min, w_max, and b must be numeric"}, status=400)
                return
            try:
                grid = mse_surface(samples, (w_min, w_max), steps=steps, b=b)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(
                {
                    "dataset": dataset,
                    "grid_rows": rows,
                    "grid_cols": cols,
                    "steps": steps,
                    "w_range": [w_min, w_max],
                    "bias": b,
                    "sample_count": len(samples),
                    "grid": grid,
                }
            )
            return
        if self.path == "/mlp-internals":
            try:
                dataset, samples, grid_shape = _load_samples_from_body(body)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            rows, cols = grid_shape
            try:
                hidden_dim = int(body.get("hidden_dim", 2))
                lr = float(body.get("lr", 0.5))
                seed = body.get("seed")
                seed_value = int(seed) if seed is not None else 0
                sample_index = int(body.get("sample_index", 0))
            except (TypeError, ValueError):
                self._send_json({"error": "hidden_dim, lr, seed, and sample_index must be numeric"}, status=400)
                return
            if hidden_dim <= 0:
                self._send_json({"error": "hidden_dim must be positive"}, status=400)
                return
            if not samples:
                self._send_json({"error": "samples must be non-empty"}, status=400)
                return
            sample = samples[sample_index % len(samples)]
            model = GridMlp(rows=rows, cols=cols, hidden_dim=hidden_dim, lr=lr, seed=seed_value)
            internals = model.model.inspect_step(sample["x"], sample["y"])
            payload = _build_mlp_payload(
                internals=internals,
                rows=rows,
                cols=cols,
                hidden_dim=hidden_dim,
                sample_index=sample_index % len(samples),
                sample_count=len(samples),
                dataset=dataset,
            )
            self._send_json(payload)
            return
        if self.path == "/step":
            dataset = body.get("dataset")
            if dataset and dataset != self.service.dataset:
                try:
                    self.service.set_dataset(dataset, custom=custom_payload)
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
