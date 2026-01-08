"""Vision utilities for AlexNet visualization (Chapter 5)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# AlexNet layer info
ALEXNET_LAYERS = {
    1: {"name": "Conv1", "filters": 64, "kernel_size": 11, "in_channels": 3},
    2: {"name": "Conv2", "filters": 192, "kernel_size": 5, "in_channels": 64},
    3: {"name": "Conv3", "filters": 384, "kernel_size": 3, "in_channels": 192},
    4: {"name": "Conv4", "filters": 256, "kernel_size": 3, "in_channels": 384},
    5: {"name": "Conv5", "filters": 256, "kernel_size": 3, "in_channels": 256},
}


@dataclass
class FilterInfo:
    """Information about a convolutional filter."""

    layer: int
    index: int
    kernel_size: int
    image_base64: str  # RGB image of the filter


@dataclass
class ActivationInfo:
    """Information about an activation map."""

    layer: int
    filter_index: int
    width: int
    height: int
    max_activation: float
    image_base64: str  # Grayscale heatmap


@dataclass
class LayerInfo:
    """Information about a convolutional layer."""

    layer: int
    name: str
    filters: int
    kernel_size: int
    in_channels: int
    output_size: tuple[int, int] | None = None


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AlexNetExtractor:
    """Extracts filters and activations from AlexNet."""

    def __init__(self) -> None:
        self._model: models.AlexNet | None = None
        self._device = get_device()
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @property
    def model(self) -> models.AlexNet:
        """Lazy load the AlexNet model with pretrained weights."""
        if self._model is None:
            self._model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            self._model = self._model.to(self._device)
            self._model.eval()
        return self._model

    def get_layer_info(self) -> list[LayerInfo]:
        """Get information about all convolutional layers."""
        return [
            LayerInfo(
                layer=layer_num,
                name=info["name"],
                filters=info["filters"],
                kernel_size=info["kernel_size"],
                in_channels=info["in_channels"],
            )
            for layer_num, info in ALEXNET_LAYERS.items()
        ]

    def get_conv_layer(self, layer: int) -> torch.nn.Conv2d:
        """Get a specific convolutional layer from AlexNet.

        AlexNet features structure:
        - 0: Conv2d (layer 1)
        - 1: ReLU
        - 2: MaxPool2d
        - 3: Conv2d (layer 2)
        - 4: ReLU
        - 5: MaxPool2d
        - 6: Conv2d (layer 3)
        - 7: ReLU
        - 8: Conv2d (layer 4)
        - 9: ReLU
        - 10: Conv2d (layer 5)
        - 11: ReLU
        - 12: MaxPool2d
        """
        layer_indices = {1: 0, 2: 3, 3: 6, 4: 8, 5: 10}
        if layer not in layer_indices:
            raise ValueError(f"Invalid layer: {layer}. Must be 1-5.")
        return self.model.features[layer_indices[layer]]  # type: ignore[return-value]

    def extract_filters(self, layer: int = 1, max_filters: int | None = None) -> list[FilterInfo]:
        """Extract filter weights as images.

        Args:
            layer: Which convolutional layer (1-5)
            max_filters: Maximum number of filters to return (None = all)

        Returns:
            List of FilterInfo with base64-encoded RGB images
        """
        conv = self.get_conv_layer(layer)
        weights = conv.weight.detach().cpu().numpy()  # (out_channels, in_channels, H, W)

        n_filters = weights.shape[0]
        if max_filters is not None:
            n_filters = min(n_filters, max_filters)

        filters: list[FilterInfo] = []
        for i in range(n_filters):
            kernel = weights[i]  # (in_channels, H, W)

            if kernel.shape[0] == 3:
                # RGB filter (layer 1) - can visualize directly
                # Normalize to 0-255
                kernel_normalized = kernel.transpose(1, 2, 0)  # (H, W, 3)
                kernel_normalized = (kernel_normalized - kernel_normalized.min()) / (
                    kernel_normalized.max() - kernel_normalized.min() + 1e-8
                )
                kernel_normalized = (kernel_normalized * 255).astype(np.uint8)
            else:
                # Multi-channel filter - average across channels for grayscale
                kernel_avg = kernel.mean(axis=0)  # (H, W)
                kernel_normalized = (kernel_avg - kernel_avg.min()) / (
                    kernel_avg.max() - kernel_avg.min() + 1e-8
                )
                kernel_normalized = (kernel_normalized * 255).astype(np.uint8)
                # Convert to RGB
                kernel_normalized = np.stack([kernel_normalized] * 3, axis=-1)

            # Create PIL image and encode as base64
            img = Image.fromarray(kernel_normalized, mode="RGB")
            # Scale up for visibility
            img = img.resize((64, 64), Image.Resampling.NEAREST)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            filters.append(FilterInfo(
                layer=layer,
                index=i,
                kernel_size=kernel.shape[1],
                image_base64=img_base64,
            ))

        return filters

    def extract_activations(
        self,
        image_bytes: bytes,
        layer: int = 1,
        max_activations: int | None = None,
    ) -> list[ActivationInfo]:
        """Extract activation maps for an input image.

        Args:
            image_bytes: Input image as bytes (JPEG/PNG)
            layer: Which convolutional layer (1-5)
            max_activations: Maximum number of activation maps (None = all)

        Returns:
            List of ActivationInfo with base64-encoded heatmaps
        """
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self._transform(img).unsqueeze(0).to(self._device)

        # Get activations up to the specified layer
        layer_indices = {1: 1, 2: 4, 3: 7, 4: 9, 5: 11}  # After ReLU
        if layer not in layer_indices:
            raise ValueError(f"Invalid layer: {layer}. Must be 1-5.")

        with torch.no_grad():
            x = input_tensor
            for i, module in enumerate(self.model.features):
                x = module(x)
                if i == layer_indices[layer]:
                    break

        activations = x.squeeze(0).cpu().numpy()  # (C, H, W)

        n_maps = activations.shape[0]
        if max_activations is not None:
            n_maps = min(n_maps, max_activations)

        results: list[ActivationInfo] = []
        for i in range(n_maps):
            act_map = activations[i]  # (H, W)
            max_val = float(act_map.max())

            # Normalize to 0-255
            if max_val > 0:
                act_normalized = (act_map / max_val * 255).astype(np.uint8)
            else:
                act_normalized = np.zeros_like(act_map, dtype=np.uint8)

            # Create heatmap image
            img = Image.fromarray(act_normalized, mode="L")
            # Scale up for visibility
            img = img.resize((64, 64), Image.Resampling.NEAREST)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            results.append(ActivationInfo(
                layer=layer,
                filter_index=i,
                width=act_map.shape[1],
                height=act_map.shape[0],
                max_activation=max_val,
                image_base64=img_base64,
            ))

        return results

    def get_param_count(self) -> int:
        """Get total parameter count for AlexNet."""
        return sum(p.numel() for p in self.model.parameters())


def create_sample_image(width: int = 224, height: int = 224, pattern: str = "gradient") -> bytes:
    """Create a sample test image.

    Args:
        width: Image width
        height: Image height
        pattern: One of 'gradient', 'checkerboard', 'noise', 'face', 'cat', 'car', 'landscape'

    Returns:
        PNG image as bytes
    """
    if pattern == "gradient":
        # Horizontal gradient
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            arr[:, x, 0] = int(x / width * 255)  # Red gradient
            arr[:, x, 2] = int((width - x) / width * 255)  # Blue gradient
        arr[:, :, 1] = 128  # Green constant

    elif pattern == "checkerboard":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        block_size = 32
        for y in range(height):
            for x in range(width):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    arr[y, x] = [255, 255, 255]
                else:
                    arr[y, x] = [0, 0, 0]

    elif pattern == "noise":
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    elif pattern == "face":
        # Simple stylized face - oval with features
        arr = np.ones((height, width, 3), dtype=np.uint8) * 220  # Light gray background
        cy, cx = height // 2, width // 2

        # Face oval (skin tone)
        for y in range(height):
            for x in range(width):
                # Ellipse for face
                if ((x - cx) / 70) ** 2 + ((y - cy) / 90) ** 2 < 1:
                    arr[y, x] = [210, 180, 140]  # Skin tone

        # Eyes
        eye_y = cy - 20
        for eye_x in [cx - 25, cx + 25]:
            for y in range(eye_y - 10, eye_y + 10):
                for x in range(eye_x - 10, eye_x + 10):
                    if 0 <= y < height and 0 <= x < width:
                        if (x - eye_x) ** 2 + (y - eye_y) ** 2 < 100:
                            arr[y, x] = [255, 255, 255]  # White
                        if (x - eye_x) ** 2 + (y - eye_y) ** 2 < 25:
                            arr[y, x] = [50, 30, 20]  # Pupil

        # Simple mouth
        mouth_y = cy + 40
        for x in range(cx - 30, cx + 30):
            if 0 <= mouth_y < height and 0 <= x < width:
                arr[mouth_y, x] = [180, 80, 80]
                if mouth_y + 1 < height:
                    arr[mouth_y + 1, x] = [180, 80, 80]

    elif pattern == "cat":
        # Simple cat silhouette with ears
        arr = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light background
        cy, cx = height // 2 + 20, width // 2

        # Body (dark gray)
        for y in range(height):
            for x in range(width):
                # Main body circle
                if ((x - cx) / 60) ** 2 + ((y - cy) / 50) ** 2 < 1:
                    arr[y, x] = [60, 60, 70]  # Dark gray

        # Head
        head_y = cy - 60
        for y in range(height):
            for x in range(width):
                if (x - cx) ** 2 + (y - head_y) ** 2 < 1600:
                    arr[y, x] = [60, 60, 70]

        # Ears (triangles)
        for ear_x in [cx - 30, cx + 30]:
            ear_top = head_y - 50
            for y in range(ear_top, head_y - 10):
                half_width = (y - ear_top) // 2
                for x in range(ear_x - half_width, ear_x + half_width):
                    if 0 <= y < height and 0 <= x < width:
                        arr[y, x] = [60, 60, 70]

        # Eyes (yellow)
        eye_y = head_y
        for eye_x in [cx - 15, cx + 15]:
            for y in range(eye_y - 6, eye_y + 6):
                for x in range(eye_x - 6, eye_x + 6):
                    if (
                        0 <= y < height
                        and 0 <= x < width
                        and (x - eye_x) ** 2 + (y - eye_y) ** 2 < 36
                    ):
                        arr[y, x] = [50, 200, 255]  # Yellow

    elif pattern == "car":
        # Simple car shape
        arr = np.ones((height, width, 3), dtype=np.uint8) * 180  # Gray road
        cx = width // 2

        # Road
        for y in range(height * 2 // 3, height):
            arr[y, :] = [100, 100, 100]

        # Car body (red)
        car_top = height // 3
        car_bottom = height * 2 // 3
        car_left = cx - 70
        car_right = cx + 70
        for y in range(car_top + 30, car_bottom):
            for x in range(car_left, car_right):
                if 0 <= y < height and 0 <= x < width:
                    arr[y, x] = [50, 50, 200]  # Red (BGR-ish)

        # Car roof
        for y in range(car_top, car_top + 40):
            roof_width = 40 + (y - car_top)
            for x in range(cx - roof_width, cx + roof_width):
                if 0 <= y < height and 0 <= x < width:
                    arr[y, x] = [50, 50, 200]

        # Windows
        for y in range(car_top + 5, car_top + 35):
            for x in range(cx - 30, cx + 30):
                if 0 <= y < height and 0 <= x < width:
                    arr[y, x] = [200, 200, 150]  # Light blue

        # Wheels
        for wheel_x in [car_left + 20, car_right - 20]:
            wheel_y = car_bottom - 5
            for y in range(wheel_y - 15, wheel_y + 15):
                for x in range(wheel_x - 15, wheel_x + 15):
                    if (
                        0 <= y < height
                        and 0 <= x < width
                        and (x - wheel_x) ** 2 + (y - wheel_y) ** 2 < 225
                    ):
                        arr[y, x] = [30, 30, 30]  # Black

    elif pattern == "landscape":
        # Simple landscape with sky, mountain, grass
        arr = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky (gradient blue)
        for y in range(height * 2 // 3):
            blue = 220 - int(y / height * 100)
            arr[y, :] = [blue, 180, 135]  # Sky blue

        # Sun
        sun_x, sun_y = width * 3 // 4, height // 5
        for y in range(height):
            for x in range(width):
                if (x - sun_x) ** 2 + (y - sun_y) ** 2 < 400:
                    arr[y, x] = [100, 220, 255]  # Yellow

        # Mountains
        for x in range(width):
            mountain_height = int(
                50 * np.sin(x / 30) + 40 * np.sin(x / 50) + height // 3
            )
            for y in range(mountain_height, height * 2 // 3):
                if 0 <= y < height:
                    arr[y, x] = [100, 100, 120]  # Gray mountain

        # Grass
        for y in range(height * 2 // 3, height):
            for x in range(width):
                green = 100 + int((y - height * 2 // 3) / (height // 3) * 50)
                arr[y, x] = [50, green, 50]  # Green grass

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    img = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def filters_to_grid(filters: list[FilterInfo], cols: int = 8) -> dict[str, Any]:
    """Convert filter list to a grid structure for visualization.

    Args:
        filters: List of FilterInfo objects
        cols: Number of columns in the grid

    Returns:
        Dictionary with grid metadata and images
    """
    rows = (len(filters) + cols - 1) // cols
    return {
        "rows": rows,
        "cols": cols,
        "count": len(filters),
        "filters": [
            {
                "index": f.index,
                "kernel_size": f.kernel_size,
                "image": f.image_base64,
            }
            for f in filters
        ],
    }


def activations_to_grid(activations: list[ActivationInfo], cols: int = 8) -> dict[str, Any]:
    """Convert activation list to a grid structure for visualization.

    Args:
        activations: List of ActivationInfo objects
        cols: Number of columns in the grid

    Returns:
        Dictionary with grid metadata and images
    """
    rows = (len(activations) + cols - 1) // cols
    return {
        "rows": rows,
        "cols": cols,
        "count": len(activations),
        "activations": [
            {
                "index": a.filter_index,
                "width": a.width,
                "height": a.height,
                "max_activation": a.max_activation,
                "image": a.image_base64,
            }
            for a in activations
        ],
    }

