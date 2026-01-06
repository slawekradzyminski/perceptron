from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from backend.api.deps import backprop_service

router = APIRouter(prefix="/backprop")


@router.get("/state")
def backprop_state() -> dict[str, Any]:
    return backprop_service.state()


@router.post("/tinygps/reset")
def tinygps_reset(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    dataset = body.get("dataset")
    lr = body.get("lr")
    params = body.get("params")
    order = body.get("order")
    if lr is not None:
        try:
            lr = float(lr)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be an object")
    if order is not None and not isinstance(order, list):
        raise HTTPException(status_code=400, detail="order must be a list")
    try:
        return backprop_service.reset_tinygps(dataset=dataset, lr=lr, params=params, order=order)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tinygps/step")
def tinygps_step() -> dict[str, Any]:
    return backprop_service.step_tinygps()


@router.post("/regression/reset")
def regression_reset(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    loss = body.get("loss")
    lr = body.get("lr")
    samples = body.get("samples")
    order = body.get("order")
    params = body.get("params")
    if lr is not None:
        try:
            lr = float(lr)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc
    if samples is not None and not isinstance(samples, list):
        raise HTTPException(status_code=400, detail="samples must be a list")
    if order is not None and not isinstance(order, list):
        raise HTTPException(status_code=400, detail="order must be a list")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be an object")
    try:
        return backprop_service.reset_regression(loss=loss, lr=lr, samples=samples, order=order, params=params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/regression/step")
def regression_step() -> dict[str, Any]:
    return backprop_service.step_regression()
