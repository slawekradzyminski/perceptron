from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from backend.api.utils import build_mlp_payload, load_samples_from_body
from backend.nn.grid_mlp import GridMlp
from backend.viz.viz_error_surface import mse_surface

router = APIRouter()


@router.post("/error-surface")
def error_surface(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    try:
        dataset, samples, grid_shape = load_samples_from_body(body)
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


@router.post("/mlp-internals")
def mlp_internals(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    try:
        dataset, samples, grid_shape = load_samples_from_body(body)
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
    return build_mlp_payload(
        internals=internals,
        rows=rows,
        cols=cols,
        hidden_dim=hidden_dim,
        sample_index=sample_index % len(samples),
        sample_count=len(samples),
        dataset=dataset,
    )
