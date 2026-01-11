"""Convolution visualization routes for Chapter 6 (CNN Lab)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.api.deps import ConvServiceDep
from backend.api.utils import validate_image_upload
from backend.schemas.request import (
    ConvCalculateRequest,
    ConvImageSizeRequest,
    ConvKernelRequest,
    ConvParamsRequest,
)

logger = logging.getLogger("perceptron_api")

router = APIRouter(prefix="/conv")


def _add_base64_images(service: ConvServiceDep, state: dict[str, Any]) -> dict[str, Any]:
    """Add base64 image representations to state."""
    state["image_base64"] = service.image_to_base64()
    state["activation_base64"] = service.activation_to_base64()
    return state


@router.get("/state")
def conv_state(service: ConvServiceDep) -> dict[str, Any]:
    """Get current convolution explorer state."""
    return _add_base64_images(service, service.state())


@router.get("/presets")
def conv_presets(service: ConvServiceDep) -> dict[str, Any]:
    """Get available image and kernel presets."""
    return service.get_presets()


@router.post("/image/size")
def conv_set_image_size(service: ConvServiceDep, body: ConvImageSizeRequest) -> dict[str, Any]:
    """Set the image size."""
    try:
        state = service.set_image_size(body.size)
        return _add_base64_images(service, state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/image/upload")
async def conv_upload_image(
    service: ConvServiceDep,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a custom image. Converted to grayscale and resized."""
    logger.info(f"Upload request: filename={file.filename}, content_type={file.content_type}")

    image_bytes = await validate_image_upload(file)
    logger.info(f"Read {len(image_bytes)} bytes from uploaded file")

    try:
        state = service.upload_image(image_bytes)
        return _add_base64_images(service, state)
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}") from e


@router.post("/image/{name}")
def conv_set_image(service: ConvServiceDep, name: str) -> dict[str, Any]:
    """Set the input image to a preset sample."""
    try:
        state = service.set_image(name)
        return _add_base64_images(service, state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/kernel")
def conv_set_kernel(service: ConvServiceDep, body: ConvKernelRequest) -> dict[str, Any]:
    """Set the convolution kernel."""
    try:
        state = service.set_kernel(name=body.name, weights=body.weights)
        return _add_base64_images(service, state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/params")
def conv_set_params(service: ConvServiceDep, body: ConvParamsRequest) -> dict[str, Any]:
    """Set convolution parameters (padding and stride)."""
    try:
        state = service.set_params(padding=body.padding, stride=body.stride)
        return _add_base64_images(service, state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/step")
def conv_get_step(
    service: ConvServiceDep,
    row: int = Query(..., ge=0),
    col: int = Query(..., ge=0),
) -> dict[str, Any]:
    """Get step-by-step calculation for a specific output position."""
    try:
        return service.get_step(row, col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/calculate")
def conv_calculate(service: ConvServiceDep, body: ConvCalculateRequest) -> dict[str, Any]:
    """Calculate output dimensions from parameters."""
    return service.calculate_output_size(
        input_size=body.input_size,
        kernel_size=body.kernel_size,
        padding=body.padding,
        stride=body.stride,
    )


@router.post("/digit/load")
def conv_load_digit_model(service: ConvServiceDep) -> dict[str, Any]:
    """Load the pretrained digit recognition model."""
    logger.info("Loading pretrained digit recognition model")
    try:
        info = service.load_digit_model()
        logger.info(f"Model loaded: {info}")
        return info
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e


@router.post("/digit/recognize")
async def conv_recognize_digit(
    service: ConvServiceDep,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Recognize a digit in an uploaded image."""
    logger.info(f"Digit recognition request: filename={file.filename}")

    image_bytes = await validate_image_upload(file)

    try:
        result = service.recognize_digit_with_activations(image_bytes)
        logger.info(f"Digit recognized: {result['prediction']} ({result['confidence']:.2%})")
        return result
    except Exception as e:
        logger.error(f"Digit recognition error: {e}")
        raise HTTPException(status_code=400, detail=f"Recognition failed: {e}") from e


@router.get("/digit/info")
def conv_digit_model_info(service: ConvServiceDep) -> dict[str, Any]:
    """Get information about the digit recognition model."""
    return service.get_digit_model_info()
