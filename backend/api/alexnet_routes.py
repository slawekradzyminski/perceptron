"""AlexNet visualization routes for Chapter 5."""

from __future__ import annotations

import base64
import logging
import traceback
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.api.deps import AlexNetServiceDep
from backend.api.utils import validate_image_upload

logger = logging.getLogger("perceptron_api")

router = APIRouter(prefix="/alexnet")


@router.get("/state")
def alexnet_state(service: AlexNetServiceDep) -> dict[str, Any]:
    """Get current AlexNet explorer state."""
    return service.state()


@router.get("/layers")
def alexnet_layers(service: AlexNetServiceDep) -> dict[str, Any]:
    """Get information about all convolutional layers."""
    state = service.state()
    return {
        "layers": state["layers"],
        "param_count": state["param_count"],
    }


@router.post("/layer/{layer}")
def alexnet_set_layer(service: AlexNetServiceDep, layer: int) -> dict[str, Any]:
    """Set the current layer for exploration.

    Args:
        layer: Layer number (1-5)
    """
    if layer < 1 or layer > 5:
        raise HTTPException(status_code=400, detail="Layer must be between 1 and 5")
    try:
        return service.set_layer(layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/filters")
def alexnet_filters(
    service: AlexNetServiceDep,
    layer: int = Query(default=1, ge=1, le=5),
    max_filters: int = Query(default=64, ge=1, le=256),
) -> dict[str, Any]:
    """Get filter visualizations for a layer.

    Args:
        layer: Layer number (1-5)
        max_filters: Maximum number of filters to return
    """
    try:
        return service.get_filters(layer=layer, max_filters=max_filters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/activations")
def alexnet_activations_sample(
    service: AlexNetServiceDep,
    sample: str = Query(default="gradient"),
    layer: int = Query(default=1, ge=1, le=5),
    max_activations: int = Query(default=64, ge=1, le=256),
) -> dict[str, Any]:
    """Get activation maps for a sample image.

    Args:
        sample: Sample image name ('gradient', 'checkerboard', 'noise')
        layer: Layer number (1-5)
        max_activations: Maximum number of activation maps
    """
    try:
        return service.get_activations(
            sample_name=sample,
            layer=layer,
            max_activations=max_activations,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/activations")
async def alexnet_activations_upload(
    service: AlexNetServiceDep,
    file: UploadFile = File(...),
    layer: int = Query(default=1, ge=1, le=5),
    max_activations: int = Query(default=64, ge=1, le=256),
) -> dict[str, Any]:
    """Get activation maps for an uploaded image.

    Args:
        file: Image file (JPEG/PNG)
        layer: Layer number (1-5)
        max_activations: Maximum number of activation maps
    """
    logger.info(f"Upload request: filename={file.filename}, content_type={file.content_type}, layer={layer}")

    image_bytes = await validate_image_upload(file)
    logger.info(f"Read {len(image_bytes)} bytes from uploaded file")

    try:
        result = service.get_activations(
            image_bytes=image_bytes,
            layer=layer,
            max_activations=max_activations,
        )
        logger.info(f"Successfully processed image, got {len(result.get('activations', []))} activations")
        return result

    except ValueError as e:
        logger.error(f"ValueError processing upload: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Exception processing upload: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}") from e


@router.get("/sample/{name}")
def alexnet_sample_image(service: AlexNetServiceDep, name: str) -> dict[str, Any]:
    """Get a sample test image as base64.

    Args:
        name: Sample name ('gradient', 'checkerboard', 'noise')
    """
    try:
        image_bytes = service.get_sample_image(name)
        return {
            "name": name,
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "content_type": "image/png",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
