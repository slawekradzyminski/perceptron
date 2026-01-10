"""Convolution visualization routes for Chapter 6 (CNN Lab)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from backend.api.deps import conv_service

logger = logging.getLogger("perceptron_api")

router = APIRouter(prefix="/conv")


@router.get("/state")
def conv_state() -> dict[str, Any]:
    """Get current convolution explorer state.

    Returns:
        Current image, kernel, parameters, and computed activation map
    """
    state = conv_service.state()
    # Add base64 images for display
    state["image_base64"] = conv_service.image_to_base64()
    state["activation_base64"] = conv_service.activation_to_base64()
    return state


@router.get("/presets")
def conv_presets() -> dict[str, Any]:
    """Get available image and kernel presets.

    Returns:
        Lists of available preset names and kernel weights
    """
    return conv_service.get_presets()


# Static image routes must come before dynamic {name} route
@router.post("/image/size")
def conv_set_image_size(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Set the image size.

    Body:
        size: New image size (4-28)
    """
    size = body.get("size")
    if size is None:
        raise HTTPException(status_code=400, detail="size is required")

    try:
        size = int(size)
        state = conv_service.set_image_size(size)
        state["image_base64"] = conv_service.image_to_base64()
        state["activation_base64"] = conv_service.activation_to_base64()
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/image/upload")
async def conv_upload_image(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a custom image.

    The image is converted to grayscale and resized.

    Args:
        file: Image file (JPEG/PNG)
    """
    logger.info(f"Upload request: filename={file.filename}, content_type={file.content_type}")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/gif"]
    if file.content_type and file.content_type not in allowed_types:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be JPEG, PNG, WebP, or GIF.",
        )

    try:
        image_bytes = await file.read()
        logger.info(f"Read {len(image_bytes)} bytes from uploaded file")

        state = conv_service.upload_image(image_bytes)
        state["image_base64"] = conv_service.image_to_base64()
        state["activation_base64"] = conv_service.activation_to_base64()
        return state
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}") from e


# Dynamic route after static routes
@router.post("/image/{name}")
def conv_set_image(name: str) -> dict[str, Any]:
    """Set the input image to a preset sample.

    Args:
        name: Sample image name (gradient, checkerboard, cross, edges, etc.)
    """
    try:
        state = conv_service.set_image(name)
        state["image_base64"] = conv_service.image_to_base64()
        state["activation_base64"] = conv_service.activation_to_base64()
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/kernel")
def conv_set_kernel(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Set the convolution kernel.

    Body:
        name: Preset kernel name (optional, ignored if weights provided)
        weights: Custom 3x3 kernel weights (optional)
    """
    name = body.get("name")
    weights = body.get("weights")

    if name is None and weights is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'name' or 'weights'",
        )

    try:
        state = conv_service.set_kernel(name=name, weights=weights)
        state["image_base64"] = conv_service.image_to_base64()
        state["activation_base64"] = conv_service.activation_to_base64()
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/params")
def conv_set_params(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Set convolution parameters (padding and stride).

    Body:
        padding: Padding value (0-4, optional)
        stride: Stride value (1-4, optional)
    """
    padding = body.get("padding")
    stride = body.get("stride")

    try:
        if padding is not None:
            padding = int(padding)
        if stride is not None:
            stride = int(stride)

        state = conv_service.set_params(padding=padding, stride=stride)
        state["image_base64"] = conv_service.image_to_base64()
        state["activation_base64"] = conv_service.activation_to_base64()
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/step")
def conv_get_step(
    row: int = Query(..., ge=0),
    col: int = Query(..., ge=0),
) -> dict[str, Any]:
    """Get step-by-step calculation for a specific output position.

    Args:
        row: Output row position
        col: Output column position

    Returns:
        Detailed calculation with patch, kernel, products, and sum
    """
    try:
        return conv_service.get_step(row, col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/calculate")
def conv_calculate(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Calculate output dimensions from parameters.

    Body:
        input_size: Input dimension (I)
        kernel_size: Kernel dimension (K)
        padding: Padding (P)
        stride: Stride (S)

    Returns:
        Output size and calculation breakdown
    """
    input_size = body.get("input_size")
    kernel_size = body.get("kernel_size")
    padding = body.get("padding")
    stride = body.get("stride")

    if input_size is None or kernel_size is None:
        raise HTTPException(
            status_code=400,
            detail="input_size and kernel_size are required",
        )

    try:
        input_size = int(input_size)
        kernel_size = int(kernel_size)
        padding = int(padding) if padding is not None else 0
        stride = int(stride) if stride is not None else 1

        return conv_service.calculate_output_size(
            input_size=input_size,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/digit/load")
def conv_load_digit_model() -> dict[str, Any]:
    """Load the pretrained digit recognition model.

    Returns:
        Model status and accuracy info
    """
    logger.info("Loading pretrained digit recognition model")
    try:
        info = conv_service.load_digit_model()
        logger.info(f"Model loaded: {info}")
        return info
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e


@router.post("/digit/recognize")
async def conv_recognize_digit(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Recognize a digit in an uploaded image.

    Uses a simple CNN trained on MNIST to classify handwritten digits.
    Returns prediction, probabilities, and layer activations for visualization.

    Args:
        file: Image file (JPEG/PNG)

    Returns:
        Prediction, confidence, probabilities, and layer activations
    """
    logger.info(f"Digit recognition request: filename={file.filename}")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/gif"]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image.",
        )

    try:
        image_bytes = await file.read()
        result = conv_service.recognize_digit_with_activations(image_bytes)
        logger.info(f"Digit recognized: {result['prediction']} ({result['confidence']:.2%})")
        return result
    except Exception as e:
        logger.error(f"Digit recognition error: {e}")
        raise HTTPException(status_code=400, detail=f"Recognition failed: {e}") from e


@router.get("/digit/info")
def conv_digit_model_info() -> dict[str, Any]:
    """Get information about the digit recognition model.

    Returns:
        Model architecture and parameter count
    """
    return conv_service.get_digit_model_info()
