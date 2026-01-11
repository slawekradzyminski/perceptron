from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.deps import BackpropServiceDep
from backend.schemas.request import RegressionResetRequest, TinyGpsResetRequest
from backend.schemas.response import BackpropStateResponse

router = APIRouter(prefix="/backprop")


@router.get("/state")
def backprop_state(service: BackpropServiceDep) -> BackpropStateResponse:
    return BackpropStateResponse(**service.state())


@router.post("/tinygps/reset")
def tinygps_reset(service: BackpropServiceDep, body: TinyGpsResetRequest) -> dict[str, Any]:
    try:
        return service.reset_tinygps(
            dataset=body.dataset,
            lr=body.lr,
            params=body.params,
            order=body.order,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tinygps/step")
def tinygps_step(service: BackpropServiceDep) -> dict[str, Any]:
    return service.step_tinygps()


@router.post("/regression/reset")
def regression_reset(service: BackpropServiceDep, body: RegressionResetRequest) -> dict[str, Any]:
    # Convert samples to dict format if provided
    samples = None
    if body.samples is not None:
        samples = [{"x": s.x, "y": s.y} for s in body.samples]
    try:
        return service.reset_regression(
            loss=body.loss,
            lr=body.lr,
            samples=samples,
            order=body.order,
            params=body.params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/regression/step")
def regression_step(service: BackpropServiceDep) -> dict[str, Any]:
    return service.step_regression()
