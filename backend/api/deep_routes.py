"""Chapter 4 Deep Learning routes (Geometry of Depth)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.api.deps import DeepServiceDep
from backend.schemas.request import DeepResetRequest, DeepStepRequest
from backend.schemas.response import (
    ClearedResponse,
    DeepBoundaryResponse,
    DeepComparisonEntry,
    DeepComparisonResponse,
    DeepHistoryResponse,
    DeepRegionsResponse,
    DeepStateResponse,
    DeepStepResponse,
)

router = APIRouter(prefix="/deep")


@router.get("/state")
def deep_state(service: DeepServiceDep) -> DeepStateResponse:
    """Get current deep learning experiment state."""
    return service.state()


@router.post("/reset")
def deep_reset(service: DeepServiceDep, body: DeepResetRequest) -> DeepStateResponse:
    """Reset the model with new configuration."""
    try:
        return service.reset(
            dataset=body.dataset,
            hidden_dims=body.hidden_dims,
            lr=body.lr,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/step")
def deep_step(service: DeepServiceDep, body: DeepStepRequest) -> DeepStepResponse:
    """Take training steps."""
    return service.step(batch_size=body.batch_size)


@router.post("/epoch")
def deep_epoch(service: DeepServiceDep) -> DeepStepResponse:
    """Train for one full epoch."""
    return service.train_epoch()


@router.get("/regions")
def deep_regions(
    service: DeepServiceDep,
    resolution: int = Query(default=50, ge=1, le=200, description="Grid resolution for counting"),
) -> DeepRegionsResponse:
    """Get region counting information."""
    return service.get_regions(resolution=resolution)


@router.get("/boundary")
def deep_boundary(
    service: DeepServiceDep,
    resolution: int = Query(default=50, ge=1, le=200, description="Grid resolution"),
) -> DeepBoundaryResponse:
    """Get boundary and tiling data for visualization."""
    return service.get_boundary(resolution=resolution)


@router.get("/history")
def deep_history(service: DeepServiceDep) -> DeepHistoryResponse:
    """Get training history."""
    return DeepHistoryResponse(history=service.get_history())


@router.post("/comparison/add")
def deep_comparison_add(service: DeepServiceDep) -> DeepComparisonEntry:
    """Add current architecture to comparison table."""
    return service.add_to_comparison()


@router.get("/comparison")
def deep_comparison(service: DeepServiceDep) -> DeepComparisonResponse:
    """Get architecture comparison table."""
    return DeepComparisonResponse(table=service.get_comparison_table())


@router.post("/comparison/clear")
def deep_comparison_clear(service: DeepServiceDep) -> ClearedResponse:
    """Clear the comparison table."""
    service.clear_comparison_table()
    return ClearedResponse()
