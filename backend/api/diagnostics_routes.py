"""Diagnostics routes for error surface and MLP internals visualization."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.deps import DiagnosticsServiceDep
from backend.api.utils import load_samples_from_body
from backend.schemas.request import ErrorSurfaceRequest, MlpInternalsRequest
from backend.schemas.response import ErrorSurfaceResponse

router = APIRouter()


@router.post("/error-surface")
def error_surface(
    service: DiagnosticsServiceDep, body: ErrorSurfaceRequest
) -> ErrorSurfaceResponse:
    """Compute MSE error surface over a weight grid."""
    try:
        # Convert request body to samples format
        raw_body = body.model_dump()
        dataset, samples, grid_shape = load_samples_from_body(raw_body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rows, cols = grid_shape
    if rows * cols != 2:
        raise HTTPException(
            status_code=400,
            detail="error surface requires 2D inputs (grid_rows * grid_cols == 2)",
        )

    try:
        grid = service.error_surface(
            samples=samples,
            w_min=body.w_min,
            w_max=body.w_max,
            steps=body.steps,
            b=body.b,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ErrorSurfaceResponse(
        dataset=dataset,
        grid_rows=rows,
        grid_cols=cols,
        steps=body.steps,
        w_range=[body.w_min, body.w_max],
        bias=body.b,
        sample_count=len(samples),
        grid=grid,
    )


@router.post("/mlp-internals")
def mlp_internals(service: DiagnosticsServiceDep, body: MlpInternalsRequest) -> dict[str, Any]:
    """Compute MLP internals for a single forward-backward pass."""
    try:
        raw_body = body.model_dump()
        dataset, samples, grid_shape = load_samples_from_body(raw_body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rows, cols = grid_shape

    if not samples:
        raise HTTPException(status_code=400, detail="samples must be non-empty")

    seed_value = body.seed if body.seed is not None else 0

    result = service.mlp_internals(
        samples=samples,
        rows=rows,
        cols=cols,
        hidden_dim=body.hidden_dim,
        lr=body.lr,
        seed=seed_value,
        sample_index=body.sample_index,
    )
    # Override dataset from load_samples_from_body
    result["dataset"] = dataset
    return result
