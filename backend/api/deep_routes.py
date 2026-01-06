"""Chapter 4 Deep Learning routes (Geometry of Depth)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from backend.api.deps import deep_service

router = APIRouter(prefix="/deep")


@router.get("/state")
def deep_state() -> dict[str, Any]:
    """Get current deep learning experiment state."""
    return deep_service.state()


@router.post("/reset")
def deep_reset(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Reset the model with new configuration.

    Body:
        dataset: str - Dataset name (baarle, xor, circles, spiral)
        hidden_dims: list[int] - Width of each hidden layer, e.g. [16, 16, 16]
        lr: float - Learning rate
        seed: int - Random seed
    """
    dataset = body.get("dataset")
    hidden_dims = body.get("hidden_dims")
    lr = body.get("lr")
    seed = body.get("seed")

    # Validate hidden_dims
    if hidden_dims is not None:
        if not isinstance(hidden_dims, list):
            raise HTTPException(status_code=400, detail="hidden_dims must be a list")
        if len(hidden_dims) == 0:
            raise HTTPException(status_code=400, detail="hidden_dims must not be empty")
        if len(hidden_dims) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 hidden layers allowed")
        for i, dim in enumerate(hidden_dims):
            if not isinstance(dim, int) or dim <= 0 or dim > 256:
                raise HTTPException(status_code=400, detail=f"hidden_dims[{i}] must be an integer in [1, 256]")

    # Validate lr
    if lr is not None:
        try:
            lr = float(lr)
            if lr <= 0 or lr > 10:
                raise HTTPException(status_code=400, detail="lr must be in (0, 10]")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="lr must be a number") from exc

    # Validate seed
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="seed must be an integer") from exc

    try:
        return deep_service.reset(dataset=dataset, hidden_dims=hidden_dims, lr=lr, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/step")
def deep_step(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Take training steps.

    Body:
        batch_size: int - Number of samples to train on (default: 1)
    """
    batch_size = body.get("batch_size", 1)

    if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 10000:
        raise HTTPException(status_code=400, detail="batch_size must be an integer in [1, 10000]")

    return deep_service.step(batch_size=batch_size)


@router.post("/epoch")
def deep_epoch() -> dict[str, Any]:
    """Train for one full epoch."""
    return deep_service.train_epoch()


@router.get("/regions")
def deep_regions(resolution: int = 50) -> dict[str, Any]:
    """Get region counting information.

    Query:
        resolution: int - Grid resolution for counting (default: 50)
    """
    if resolution <= 0 or resolution > 200:
        raise HTTPException(status_code=400, detail="resolution must be in [1, 200]")

    return deep_service.get_regions(resolution=resolution)


@router.get("/boundary")
def deep_boundary(resolution: int = 50) -> dict[str, Any]:
    """Get boundary and tiling data for visualization.

    Query:
        resolution: int - Grid resolution (default: 50)

    Returns:
        predictions: 2D grid of predicted classes
        region_ids: 2D grid of region IDs for tiling visualization
        region_count: Number of unique regions
        theoretical_max: Theoretical maximum regions
    """
    if resolution <= 0 or resolution > 200:
        raise HTTPException(status_code=400, detail="resolution must be in [1, 200]")

    return deep_service.get_boundary(resolution=resolution)


@router.get("/history")
def deep_history() -> dict[str, Any]:
    """Get training history."""
    return {"history": deep_service.get_history()}


@router.post("/comparison/add")
def deep_comparison_add() -> dict[str, Any]:
    """Add current architecture to comparison table."""
    return deep_service.add_to_comparison()


@router.get("/comparison")
def deep_comparison() -> dict[str, Any]:
    """Get architecture comparison table."""
    return {"table": deep_service.get_comparison_table()}


@router.post("/comparison/clear")
def deep_comparison_clear() -> dict[str, Any]:
    """Clear the comparison table."""
    deep_service.clear_comparison_table()
    return {"cleared": True}

