"""Convolution service for CNN Lab visualization (Chapter 6).

Manages state for interactive convolution exploration including
image selection, kernel editing, and step-by-step calculation.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from backend.nn.conv_utils import (
    PRESET_KERNELS,
    ConvStepResult,
    compute_output_size,
    conv_full,
    conv_step,
    create_sample_image,
    normalize_activation_map,
)
from backend.nn.digit_cnn import DigitRecognizer

# Available sample images
SAMPLE_IMAGES = [
    "gradient",
    "checkerboard",
    "cross",
    "edges",
    "diagonal",
    "corner",
    "digit_0",
    "digit_1",
]


@dataclass
class ConvState:
    """State for convolution exploration."""

    image: list[list[float]]
    image_name: str
    kernel: list[list[float]]
    kernel_name: str
    padding: int
    stride: int
    image_size: int


class ConvService:
    """Manages state for CNN Lab convolution exploration."""

    def __init__(self) -> None:
        self._image_size = 8  # Default to 8x8 for clear visualization
        self._image = create_sample_image("cross", self._image_size)
        self._image_name = "cross"
        self._kernel = PRESET_KERNELS["sobel_x"].copy()
        self._kernel_name = "sobel_x"
        self._padding = 1
        self._stride = 1
        self._digit_recognizer: DigitRecognizer | None = None

    @property
    def digit_recognizer(self) -> DigitRecognizer:
        """Lazy load digit recognizer with pretrained weights."""
        if self._digit_recognizer is None:
            self._digit_recognizer = DigitRecognizer(model_path="data/digit_cnn.pt")
        return self._digit_recognizer

    def load_digit_model(self) -> dict[str, Any]:
        """Load the pretrained digit recognition model.

        Returns:
            Model status and accuracy info
        """
        # Force model loading by accessing the property
        _ = self.digit_recognizer.model
        info = self.digit_recognizer.get_info()
        return {
            "model_ready": info.get("model_ready", True),
            "accuracy": info.get("accuracy", 0.98),
        }

    def recognize_digit(self, image_bytes: bytes) -> dict[str, Any]:
        """Recognize a digit in an uploaded image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            Recognition result with prediction and probabilities
        """
        result = self.digit_recognizer.recognize(image_bytes)
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "probabilities": [
                {"digit": d, "probability": p}
                for d, p in result.probabilities
            ],
        }

    def recognize_digit_with_activations(self, image_bytes: bytes) -> dict[str, Any]:
        """Recognize a digit and return layer activations for visualization.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            Recognition result with prediction, probabilities, and layer activations
        """
        return self.digit_recognizer.recognize_with_activations(image_bytes)

    def get_digit_model_info(self) -> dict[str, Any]:
        """Get information about the digit recognition model."""
        return self.digit_recognizer.get_info()

    def state(self) -> dict[str, Any]:
        """Get current state including computed activation map."""
        # Compute the full convolution
        result = conv_full(
            self._image,
            self._kernel,
            self._padding,
            self._stride,
        )

        # Normalize for visualization
        normalized = normalize_activation_map(result.activation_map)

        return {
            "image": self._image,
            "image_name": self._image_name,
            "image_size": self._image_size,
            "kernel": self._kernel,
            "kernel_name": self._kernel_name,
            "padding": self._padding,
            "stride": self._stride,
            "input_shape": result.input_shape,
            "kernel_shape": result.kernel_shape,
            "output_shape": result.output_shape,
            "activation_map": result.activation_map,
            "activation_map_normalized": normalized,
        }

    def set_image(self, name: str) -> dict[str, Any]:
        """Set to a preset sample image.

        Args:
            name: Sample image name

        Returns:
            Updated state
        """
        if name not in SAMPLE_IMAGES:
            raise ValueError(f"Unknown image: {name}. Options: {SAMPLE_IMAGES}")

        self._image = create_sample_image(name, self._image_size)
        self._image_name = name
        return self.state()

    def set_image_size(self, size: int) -> dict[str, Any]:
        """Set image size and regenerate current image.

        Args:
            size: New image size (4-28)

        Returns:
            Updated state
        """
        if size < 4 or size > 28:
            raise ValueError("Image size must be between 4 and 28")

        self._image_size = size
        self._image = create_sample_image(self._image_name, size)
        return self.state()

    def upload_image(self, image_bytes: bytes) -> dict[str, Any]:
        """Upload a custom image.

        The image is converted to grayscale and resized to current image_size.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            Updated state
        """
        # Load and convert to grayscale
        img = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Resize to current image size
        img = img.resize((self._image_size, self._image_size), Image.Resampling.LANCZOS)

        # Convert to our format
        self._image = []
        for y in range(self._image_size):
            row: list[float] = []
            for x in range(self._image_size):
                row.append(float(img.getpixel((x, y))))
            self._image.append(row)

        self._image_name = "custom"
        return self.state()

    def set_kernel(
        self,
        name: str | None = None,
        weights: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Set the convolution kernel.

        Args:
            name: Preset kernel name (ignored if weights provided)
            weights: Custom kernel weights (3x3)

        Returns:
            Updated state
        """
        if weights is not None:
            # Validate custom weights
            if len(weights) != 3 or any(len(row) != 3 for row in weights):
                raise ValueError("Kernel must be 3x3")
            self._kernel = [[float(v) for v in row] for row in weights]
            self._kernel_name = "custom"
        elif name is not None:
            if name not in PRESET_KERNELS:
                raise ValueError(f"Unknown kernel: {name}. Options: {list(PRESET_KERNELS.keys())}")
            self._kernel = [row.copy() for row in PRESET_KERNELS[name]]
            self._kernel_name = name
        else:
            raise ValueError("Must provide either name or weights")

        return self.state()

    def set_params(
        self,
        padding: int | None = None,
        stride: int | None = None,
    ) -> dict[str, Any]:
        """Set convolution parameters.

        Args:
            padding: New padding value (0-4)
            stride: New stride value (1-4)

        Returns:
            Updated state
        """
        if padding is not None:
            if padding < 0 or padding > 4:
                raise ValueError("Padding must be between 0 and 4")
            self._padding = padding

        if stride is not None:
            if stride < 1 or stride > 4:
                raise ValueError("Stride must be between 1 and 4")
            self._stride = stride

        return self.state()

    def get_step(self, row: int, col: int) -> dict[str, Any]:
        """Get detailed step calculation for a position.

        Args:
            row: Output row position
            col: Output column position

        Returns:
            Step details with full math trace
        """
        # Validate position
        kernel_size = len(self._kernel)
        output_height = compute_output_size(
            len(self._image), kernel_size, self._padding, self._stride
        )
        output_width = compute_output_size(
            len(self._image[0]), kernel_size, self._padding, self._stride
        )

        if row < 0 or row >= output_height:
            raise ValueError(f"Row {row} out of range [0, {output_height})")
        if col < 0 or col >= output_width:
            raise ValueError(f"Col {col} out of range [0, {output_width})")

        # Compute step with full trace
        result: ConvStepResult = conv_step(
            self._image,
            self._kernel,
            row,
            col,
            self._padding,
            self._stride,
        )

        return {
            "output_row": result.row,
            "output_col": result.col,
            "input_row": result.input_row,
            "input_col": result.input_col,
            "patch": result.patch,
            "kernel": result.kernel,
            "products": result.products,
            "sum": result.sum_value,
            "kernel_size": kernel_size,
            "padding": self._padding,
            "stride": self._stride,
        }

    def calculate_output_size(
        self,
        input_size: int,
        kernel_size: int,
        padding: int,
        stride: int,
    ) -> dict[str, Any]:
        """Calculate output dimensions with formula details.

        Args:
            input_size: Input dimension (I)
            kernel_size: Kernel dimension (K)
            padding: Padding (P)
            stride: Stride (S)

        Returns:
            Output size and calculation breakdown
        """
        try:
            output_size = compute_output_size(input_size, kernel_size, padding, stride)
            numerator = input_size - kernel_size + 2 * padding

            return {
                "input_size": input_size,
                "kernel_size": kernel_size,
                "padding": padding,
                "stride": stride,
                "output_size": output_size,
                "formula": f"({input_size} - {kernel_size} + 2×{padding}) / {stride} + 1",
                "numerator": numerator,
                "valid": True,
                "error": None,
            }
        except ValueError as e:
            return {
                "input_size": input_size,
                "kernel_size": kernel_size,
                "padding": padding,
                "stride": stride,
                "output_size": None,
                "formula": f"({input_size} - {kernel_size} + 2×{padding}) / {stride} + 1",
                "numerator": input_size - kernel_size + 2 * padding,
                "valid": False,
                "error": str(e),
            }

    def get_presets(self) -> dict[str, Any]:
        """Get available presets.

        Returns:
            Dictionary with image and kernel presets
        """
        return {
            "images": SAMPLE_IMAGES,
            "kernels": list(PRESET_KERNELS.keys()),
            "kernel_weights": dict(PRESET_KERNELS),
        }

    def image_to_base64(self) -> str:
        """Convert current image to base64 PNG for display.

        Returns:
            Base64-encoded PNG image
        """
        # Create PIL image
        size = len(self._image)
        img = Image.new("L", (size, size))

        for y in range(size):
            for x in range(size):
                val = int(max(0, min(255, self._image[y][x])))
                img.putpixel((x, y), val)

        # Scale up for visibility
        display_size = max(64, size * 8)
        img = img.resize((display_size, display_size), Image.Resampling.NEAREST)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def activation_to_base64(self) -> str:
        """Convert current activation map to base64 PNG for display.

        Returns:
            Base64-encoded PNG image
        """
        result = conv_full(
            self._image,
            self._kernel,
            self._padding,
            self._stride,
        )

        normalized = normalize_activation_map(result.activation_map)

        # Create PIL image
        height = len(normalized)
        width = len(normalized[0]) if height > 0 else 0

        if height == 0 or width == 0:
            # Return 1x1 gray image
            img = Image.new("L", (1, 1))
            img.putpixel((0, 0), 128)
        else:
            img = Image.new("L", (width, height))
            for y in range(height):
                for x in range(width):
                    val = int(max(0, min(255, normalized[y][x])))
                    img.putpixel((x, y), val)

            # Scale up for visibility
            display_size = max(64, max(width, height) * 8)
            img = img.resize((display_size, display_size), Image.Resampling.NEAREST)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
