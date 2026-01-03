from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from backend.api.deps import lms_service
from backend.api.utils import normalize_samples, validate_grid_shape

router = APIRouter(prefix="/lms")


@router.get("/state")
def lms_state() -> Dict[str, Any]:
    return lms_service.state()


@router.post("/reset")
def lms_reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "lr" in body:
        try:
            lms_service.set_lr(float(body["lr"]))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    dataset = body.get("dataset", lms_service.dataset)
    if dataset == "custom":
        try:
            rows, cols = validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
            if rows * cols != 2:
                raise ValueError("LMS requires 2D inputs (grid_rows * grid_cols == 2)")
            samples = normalize_samples(body.get("samples", []), rows, cols)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        lms_service.set_dataset("custom", custom_samples=samples)
    else:
        try:
            lms_service.set_dataset(dataset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return lms_service.state()


@router.post("/step")
def lms_step() -> Dict[str, Any]:
    return lms_service.step()
