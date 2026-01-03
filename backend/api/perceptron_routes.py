from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Body, HTTPException

from backend.api.deps import perceptron_service
from backend.api.utils import normalize_samples, validate_grid_shape

router = APIRouter()


@router.get("/state")
def state() -> Dict[str, Any]:
    return perceptron_service.state()


@router.post("/step")
def step(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "lr" in body:
        try:
            perceptron_service.set_lr(float(body["lr"]))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    custom_payload: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None
    if body.get("dataset") == "custom" and "samples" in body:
        try:
            rows, cols = validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
            samples = normalize_samples(body.get("samples", []), rows, cols)
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


@router.post("/reset")
def reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if "lr" in body:
        try:
            perceptron_service.set_lr(float(body["lr"]))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    custom_payload: Tuple[List[Dict[str, Any]], Tuple[int, int]] | None = None
    if body.get("dataset") == "custom" and "samples" in body:
        try:
            rows, cols = validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
            samples = normalize_samples(body.get("samples", []), rows, cols)
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
