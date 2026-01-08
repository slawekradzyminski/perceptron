"""AlexNet service for Chapter 5 visualization."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from backend.nn.vision_utils import (
    ALEXNET_LAYERS,
    AlexNetExtractor,
    activations_to_grid,
    create_sample_image,
    filters_to_grid,
)


@dataclass
class AlexNetState:
    """State for AlexNet explorer."""

    current_layer: int
    sample_images: list[str]  # Names of available sample images


SAMPLE_IMAGES = ["gradient", "checkerboard", "noise", "face", "cat", "car", "landscape"]


class AlexNetService:
    """Manages state for AlexNet exploration."""

    def __init__(self) -> None:
        self._extractor: AlexNetExtractor | None = None
        self._current_layer = 1
        self._cached_sample_images: dict[str, bytes] = {}

    @property
    def extractor(self) -> AlexNetExtractor:
        """Lazy load the extractor."""
        if self._extractor is None:
            self._extractor = AlexNetExtractor()
        return self._extractor

    def state(self) -> dict[str, Any]:
        """Get current state."""
        layers = self.extractor.get_layer_info()
        return {
            "current_layer": self._current_layer,
            "sample_images": SAMPLE_IMAGES,
            "layers": [
                {
                    "layer": layer.layer,
                    "name": layer.name,
                    "filters": layer.filters,
                    "kernel_size": layer.kernel_size,
                    "in_channels": layer.in_channels,
                }
                for layer in layers
            ],
            "param_count": self.extractor.get_param_count(),
        }

    def set_layer(self, layer: int) -> dict[str, Any]:
        """Set the current layer for exploration."""
        if layer not in ALEXNET_LAYERS:
            raise ValueError(f"Invalid layer: {layer}. Must be 1-5.")
        self._current_layer = layer
        return self.state()

    def get_filters(self, layer: int | None = None, max_filters: int = 64) -> dict[str, Any]:
        """Get filter visualizations for a layer.

        Args:
            layer: Layer number (1-5), uses current_layer if None
            max_filters: Maximum number of filters to return

        Returns:
            Grid structure with filter images
        """
        layer = layer if layer is not None else self._current_layer
        if layer not in ALEXNET_LAYERS:
            raise ValueError(f"Invalid layer: {layer}. Must be 1-5.")

        filters = self.extractor.extract_filters(layer=layer, max_filters=max_filters)
        grid = filters_to_grid(filters)
        grid["layer"] = layer
        grid["layer_info"] = ALEXNET_LAYERS[layer]
        return grid

    def get_activations(
        self,
        image_bytes: bytes | None = None,
        sample_name: str | None = None,
        layer: int | None = None,
        max_activations: int = 64,
    ) -> dict[str, Any]:
        """Get activation maps for an input image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG), or None to use sample
            sample_name: Name of sample image if image_bytes is None
            layer: Layer number (1-5), uses current_layer if None
            max_activations: Maximum number of activation maps

        Returns:
            Grid structure with activation heatmaps, including input_image
        """
        layer = layer if layer is not None else self._current_layer
        if layer not in ALEXNET_LAYERS:
            raise ValueError(f"Invalid layer: {layer}. Must be 1-5.")

        # Get image bytes
        used_sample_name: str | None = None
        if image_bytes is None:
            sample_name = sample_name or "gradient"
            if sample_name not in SAMPLE_IMAGES:
                raise ValueError(f"Unknown sample: {sample_name}. Options: {SAMPLE_IMAGES}")
            if sample_name not in self._cached_sample_images:
                self._cached_sample_images[sample_name] = create_sample_image(pattern=sample_name)
            image_bytes = self._cached_sample_images[sample_name]
            used_sample_name = sample_name

        activations = self.extractor.extract_activations(
            image_bytes=image_bytes,
            layer=layer,
            max_activations=max_activations,
        )
        grid = activations_to_grid(activations)
        grid["layer"] = layer
        grid["layer_info"] = ALEXNET_LAYERS[layer]
        grid["sample_name"] = used_sample_name
        # Include input image as base64 for display
        grid["input_image"] = base64.b64encode(image_bytes).decode("utf-8")
        return grid

    def get_sample_image(self, name: str) -> bytes:
        """Get a sample test image.

        Args:
            name: Sample name ('gradient', 'checkerboard', 'noise')

        Returns:
            PNG image bytes
        """
        if name not in SAMPLE_IMAGES:
            raise ValueError(f"Unknown sample: {name}. Options: {SAMPLE_IMAGES}")
        if name not in self._cached_sample_images:
            self._cached_sample_images[name] = create_sample_image(pattern=name)
        return self._cached_sample_images[name]

