from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.deps import LmsServiceDep
from backend.api.utils import normalize_samples
from backend.schemas.request import LmsResetRequest
from backend.schemas.response import LmsStateResponse, LmsStepResponse

router = APIRouter(prefix="/lms")


@router.get("/state")
def lms_state(service: LmsServiceDep) -> LmsStateResponse:
    return LmsStateResponse(**service.state())


@router.post("/reset")
def lms_reset(service: LmsServiceDep, body: LmsResetRequest) -> LmsStateResponse:
    if body.lr is not None:
        service.set_lr(body.lr)
    dataset = body.dataset if body.dataset else service.dataset
    if dataset == "custom":
        if body.samples is None:
            raise HTTPException(status_code=400, detail="samples required for custom dataset")
        if body.grid_rows is None or body.grid_cols is None:
            raise HTTPException(status_code=400, detail="grid_rows and grid_cols required for custom dataset")
        try:
            raw_samples = [s.model_dump() for s in body.samples]
            samples = normalize_samples(raw_samples, body.grid_rows, body.grid_cols)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        service.set_dataset("custom", custom_samples=samples)
    else:
        try:
            service.set_dataset(dataset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LmsStateResponse(**service.state())


@router.post("/step")
def lms_step(service: LmsServiceDep) -> LmsStepResponse:
    return LmsStepResponse(**service.step())
