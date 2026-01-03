from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Body, HTTPException

from backend.api.deps import mlp_service
from backend.api.utils import build_mlp_payload, normalize_samples, validate_grid_shape

router = APIRouter(prefix="/mlp")


def _parse_custom_payload(body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    rows, cols = validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
    samples = normalize_samples(body.get("samples", []), rows, cols)
    return samples, (rows, cols)


@router.get("/state")
def mlp_state() -> Dict[str, Any]:
    return mlp_service.snapshot()


@router.post("/reset")
def mlp_reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "hidden_dim" in body or "lr" in body or "seed" in body:
        try:
            hidden_dim = int(body["hidden_dim"]) if "hidden_dim" in body else None
            lr = float(body["lr"]) if "lr" in body else None
            seed = int(body["seed"]) if "seed" in body else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="hidden_dim, lr, and seed must be numeric") from exc
        try:
            mlp_service.set_hyperparams(hidden_dim=hidden_dim, lr=lr, seed=seed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = body.get("dataset")
    if dataset:
        if dataset == "custom":
            try:
                custom = _parse_custom_payload(body)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            mlp_service.set_dataset("custom", custom=custom)
        else:
            try:
                mlp_service.set_dataset(dataset)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        mlp_service.reset_model()
    return mlp_service.snapshot()


@router.post("/step")
def mlp_step() -> Dict[str, Any]:
    snapshot, internals = mlp_service.step()
    step_payload = build_mlp_payload(
        internals=internals,
        rows=mlp_service.grid_rows,
        cols=mlp_service.grid_cols,
        hidden_dim=mlp_service.hidden_dim,
        sample_index=(mlp_service.idx - 1) % len(mlp_service.samples),
        sample_count=len(mlp_service.samples),
        dataset=mlp_service.dataset,
    )
    return {**snapshot, "step": step_payload}
