from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.deps import MlpServiceDep
from backend.api.utils import build_mlp_payload, normalize_samples
from backend.schemas.request import MlpResetRequest

router = APIRouter(prefix="/mlp")


def _parse_custom_payload(body: MlpResetRequest) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """Parse custom dataset from request body."""
    if body.grid_rows is None or body.grid_cols is None:
        raise ValueError("grid_rows and grid_cols required for custom dataset")
    if body.samples is None:
        raise ValueError("samples required for custom dataset")
    raw_samples = [s.model_dump() for s in body.samples]
    samples = normalize_samples(raw_samples, body.grid_rows, body.grid_cols)
    return samples, (body.grid_rows, body.grid_cols)


@router.get("/state")
def mlp_state(service: MlpServiceDep) -> dict[str, Any]:
    return service.snapshot()


@router.post("/reset")
def mlp_reset(service: MlpServiceDep, body: MlpResetRequest) -> dict[str, Any]:
    # Apply hyperparameters if any were provided
    if body.hidden_dim is not None or body.lr is not None or body.seed is not None:
        try:
            service.set_hyperparams(hidden_dim=body.hidden_dim, lr=body.lr, seed=body.seed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Handle dataset changes
    if body.dataset:
        if body.dataset == "custom":
            try:
                custom = _parse_custom_payload(body)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            service.set_dataset("custom", custom=custom)
        else:
            try:
                service.set_dataset(body.dataset)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        service.reset_model()
    return service.snapshot()


@router.post("/step")
def mlp_step(service: MlpServiceDep) -> dict[str, Any]:
    snapshot, internals = service.step()
    step_payload = build_mlp_payload(
        internals=internals,
        rows=service.grid_rows,
        cols=service.grid_cols,
        hidden_dim=service.hidden_dim,
        sample_index=(service.idx - 1) % len(service.samples),
        sample_count=len(service.samples),
        dataset=service.dataset,
    )
    return {**snapshot, "step": step_payload}
