"""FastAPI server for perceptron + diagnostics + LMS."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.datasets import make_or_dataset_pm1, make_xor_dataset_pm1
from backend.nn.grid_mlp import GridMlp, reshape_template
from backend.nn.mlp import MlpInternals
from backend.services.lms_service import LmsService
from backend.services.perceptron_service import PerceptronService
from backend.viz.viz_error_surface import mse_surface

app = FastAPI(title="Perceptron Visual Lab API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

perceptron_service = PerceptronService()
lms_service = LmsService()


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


@app.get("/state")
def state() -> Dict[str, Any]:
    return perceptron_service.state()


@app.post("/step")
def step(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "lr" in body:
        try:
            perceptron_service.set_lr(float(body["lr"]))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    custom_payload: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None
    if body.get("dataset") == "custom" and "samples" in body:
        try:
            rows, cols = _validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
            samples = _normalize_samples(body.get("samples", []), rows, cols)
            custom_payload = (samples, (rows, cols))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = body.get("dataset")
    if dataset and dataset != perceptron_service.dataset:
        try:
            perceptron_service.set_dataset(dataset, custom=custom_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return perceptron_service.step()


@app.post("/reset")
def reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "lr" in body:
        try:
            perceptron_service.set_lr(float(body["lr"]))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    custom_payload: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None
    if body.get("dataset") == "custom" and "samples" in body:
        try:
            rows, cols = _validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
            samples = _normalize_samples(body.get("samples", []), rows, cols)
            custom_payload = (samples, (rows, cols))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = body.get("dataset")
    if dataset:
        try:
            perceptron_service.set_dataset(dataset, custom=custom_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return perceptron_service.reset()


@app.post("/error-surface")
def error_surface(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    try:
        dataset, samples, grid_shape = _load_samples_from_body(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rows, cols = grid_shape
    if rows * cols != 2:
        raise HTTPException(status_code=400, detail="error surface requires 2D inputs (grid_rows * grid_cols == 2)")
    try:
        steps = int(body.get("steps", 25))
        w_min = float(body.get("w_min", -2.0))
        w_max = float(body.get("w_max", 2.0))
        b = float(body.get("b", 0.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="steps, w_min, w_max, and b must be numeric") from exc
    try:
        grid = mse_surface(samples, (w_min, w_max), steps=steps, b=b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "dataset": dataset,
        "grid_rows": rows,
        "grid_cols": cols,
        "steps": steps,
        "w_range": [w_min, w_max],
        "bias": b,
        "sample_count": len(samples),
        "grid": grid,
    }


@app.post("/mlp-internals")
def mlp_internals(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    try:
        dataset, samples, grid_shape = _load_samples_from_body(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rows, cols = grid_shape
    try:
        hidden_dim = int(body.get("hidden_dim", 2))
        lr = float(body.get("lr", 0.5))
        seed = body.get("seed")
        seed_value = int(seed) if seed is not None else 0
        sample_index = int(body.get("sample_index", 0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="hidden_dim, lr, seed, and sample_index must be numeric") from exc
    if hidden_dim <= 0:
        raise HTTPException(status_code=400, detail="hidden_dim must be positive")
    if not samples:
        raise HTTPException(status_code=400, detail="samples must be non-empty")
    sample = samples[sample_index % len(samples)]
    model = GridMlp(rows=rows, cols=cols, hidden_dim=hidden_dim, lr=lr, seed=seed_value)
    internals = model.model.inspect_step(sample["x"], sample["y"])
    return _build_mlp_payload(
        internals=internals,
        rows=rows,
        cols=cols,
        hidden_dim=hidden_dim,
        sample_index=sample_index % len(samples),
        sample_count=len(samples),
        dataset=dataset,
    )


@app.get("/lms/state")
def lms_state() -> Dict[str, Any]:
    return lms_service.state()


@app.post("/lms/reset")
def lms_reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "lr" in body:
        try:
            lms_service.set_lr(float(body["lr"]))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    dataset = body.get("dataset", lms_service.dataset)
    if dataset == "custom":
        try:
            rows, cols = _validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
            if rows * cols != 2:
                raise ValueError("LMS requires 2D inputs (grid_rows * grid_cols == 2)")
            samples = _normalize_samples(body.get("samples", []), rows, cols)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        lms_service.set_dataset("custom", custom_samples=samples)
    else:
        try:
            lms_service.set_dataset(dataset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return lms_service.state()


@app.post("/lms/step")
def lms_step() -> Dict[str, Any]:
    return lms_service.step()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("backend.api_app:app", host=host, port=port, log_level="info")
