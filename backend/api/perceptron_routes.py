from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.deps import PerceptronServiceDep
from backend.api.utils import normalize_samples
from backend.schemas.request import PerceptronResetRequest, PerceptronStepRequest
from backend.schemas.response import PerceptronStateResponse, PerceptronStepResponse

router = APIRouter()


def _build_custom_payload(
    body: PerceptronStepRequest | PerceptronResetRequest,
) -> tuple[list[dict[str, Any]], tuple[int, int]] | None:
    """Build custom dataset payload from request body."""
    if body.dataset != "custom" or body.samples is None:
        return None
    if body.grid_rows is None or body.grid_cols is None:
        raise HTTPException(status_code=400, detail="grid_rows and grid_cols required for custom dataset")
    # Convert Pydantic models to dicts for normalize_samples
    raw_samples = [s.model_dump() for s in body.samples]
    samples = normalize_samples(raw_samples, body.grid_rows, body.grid_cols)
    return (samples, (body.grid_rows, body.grid_cols))


@router.get("/state")
def state(service: PerceptronServiceDep) -> PerceptronStateResponse:
    return service.state()


@router.post("/step")
def step(service: PerceptronServiceDep, body: PerceptronStepRequest) -> PerceptronStepResponse:
    if body.lr is not None:
        service.set_lr(body.lr)
    try:
        custom_payload = _build_custom_payload(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.dataset and body.dataset != service.dataset:
        try:
            service.set_dataset(body.dataset, custom=custom_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.step()


@router.post("/reset")
def reset(service: PerceptronServiceDep, body: PerceptronResetRequest) -> PerceptronStateResponse:
    if body.lr is not None:
        service.set_lr(body.lr)
    try:
        custom_payload = _build_custom_payload(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.dataset:
        try:
            service.set_dataset(body.dataset, custom=custom_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.reset()
