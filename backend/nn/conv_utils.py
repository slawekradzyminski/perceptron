"""Convolution utilities for CNN Lab visualization (Chapter 6).

Provides pure functions for 2D convolution operations with step-by-step
math tracing to help users understand how CNNs process images.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConvStepResult:
    """Result of a single convolution step at position (row, col).

    Attributes:
        row: Output row position (in activation map)
        col: Output column position (in activation map)
        input_row: Starting row in (padded) input image
        input_col: Starting col in (padded) input image
        patch: Extracted input patch (receptive field)
        kernel: Kernel weights used
        products: Element-wise products (patch * kernel)
        sum_value: Final sum (dot product result)
    """

    row: int
    col: int
    input_row: int
    input_col: int
    patch: list[list[float]]
    kernel: list[list[float]]
    products: list[list[float]]
    sum_value: float


@dataclass
class ConvResult:
    """Full convolution result.

    Attributes:
        input_shape: (height, width) of original input
        kernel_shape: (height, width) of kernel
        output_shape: (height, width) of activation map
        padding: Padding used
        stride: Stride used
        activation_map: 2D activation map
    """

    input_shape: tuple[int, int]
    kernel_shape: tuple[int, int]
    output_shape: tuple[int, int]
    padding: int
    stride: int
    activation_map: list[list[float]]


# Preset kernels for common image processing operations
PRESET_KERNELS: dict[str, list[list[float]]] = {
    "identity": [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    "edge_horizontal": [
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    "edge_vertical": [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ],
    "sobel_x": [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ],
    "sobel_y": [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ],
    "blur": [
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ],
    "sharpen": [
        [0.0, -1.0, 0.0],
        [-1.0, 5.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    "emboss": [
        [-2.0, -1.0, 0.0],
        [-1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
    ],
    "laplacian": [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
}


def compute_output_size(
    input_size: int,
    kernel_size: int,
    padding: int = 0,
    stride: int = 1,
) -> int:
    """Compute the output dimension for a convolution.

    Formula: O = floor((I - K + 2P) / S) + 1

    Args:
        input_size: Input dimension (I)
        kernel_size: Kernel dimension (K)
        padding: Padding amount (P)
        stride: Stride value (S)

    Returns:
        Output dimension (O)

    Raises:
        ValueError: If parameters would result in invalid output
    """
    if stride <= 0:
        raise ValueError("Stride must be positive")
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    if input_size <= 0:
        raise ValueError("Input size must be positive")
    if padding < 0:
        raise ValueError("Padding must be non-negative")

    numerator = input_size - kernel_size + 2 * padding
    if numerator < 0:
        raise ValueError(
            f"Kernel size {kernel_size} is too large for input {input_size} "
            f"with padding {padding}"
        )

    output_size = numerator // stride + 1
    return output_size


def pad_image(
    image: list[list[float]],
    padding: int,
    pad_value: float = 0.0,
) -> list[list[float]]:
    """Add zero padding around an image.

    Args:
        image: 2D input image
        padding: Number of pixels to add on each side
        pad_value: Value to use for padding (default 0.0)

    Returns:
        Padded image
    """
    if padding == 0:
        return image

    height = len(image)
    width = len(image[0]) if height > 0 else 0

    new_height = height + 2 * padding
    new_width = width + 2 * padding

    # Create padded image
    padded: list[list[float]] = [
        [pad_value] * new_width for _ in range(new_height)
    ]

    # Copy original image into center
    for r in range(height):
        for c in range(width):
            padded[r + padding][c + padding] = image[r][c]

    return padded


def extract_patch(
    image: list[list[float]],
    row: int,
    col: int,
    kernel_size: int,
) -> list[list[float]]:
    """Extract a local patch (receptive field) from an image.

    Args:
        image: 2D input image (already padded if needed)
        row: Top-left row of patch
        col: Top-left column of patch
        kernel_size: Size of the patch to extract

    Returns:
        2D patch of size kernel_size × kernel_size
    """
    patch: list[list[float]] = []
    for r in range(kernel_size):
        row_data: list[float] = []
        for c in range(kernel_size):
            row_data.append(image[row + r][col + c])
        patch.append(row_data)
    return patch


def elementwise_product(
    patch: list[list[float]],
    kernel: list[list[float]],
) -> list[list[float]]:
    """Compute element-wise product of patch and kernel.

    Args:
        patch: Local input patch
        kernel: Kernel weights

    Returns:
        Element-wise product matrix
    """
    products: list[list[float]] = []
    for r in range(len(kernel)):
        row: list[float] = []
        for c in range(len(kernel[0])):
            row.append(patch[r][c] * kernel[r][c])
        products.append(row)
    return products


def sum_matrix(matrix: list[list[float]]) -> float:
    """Sum all elements in a 2D matrix."""
    total = 0.0
    for row in matrix:
        for val in row:
            total += val
    return total


def conv_step(
    image: list[list[float]],
    kernel: list[list[float]],
    output_row: int,
    output_col: int,
    padding: int = 0,
    stride: int = 1,
) -> ConvStepResult:
    """Compute a single convolution step with full math trace.

    This function performs the convolution for a single output position
    and returns all intermediate values for visualization.

    Args:
        image: Original (unpadded) input image
        kernel: Convolution kernel
        output_row: Row in the output activation map
        output_col: Column in the output activation map
        padding: Padding amount
        stride: Stride value

    Returns:
        ConvStepResult with full calculation details
    """
    kernel_size = len(kernel)

    # Pad the image
    padded = pad_image(image, padding)

    # Calculate position in padded image
    input_row = output_row * stride
    input_col = output_col * stride

    # Extract patch
    patch = extract_patch(padded, input_row, input_col, kernel_size)

    # Compute element-wise products
    products = elementwise_product(patch, kernel)

    # Sum to get activation
    sum_value = sum_matrix(products)

    return ConvStepResult(
        row=output_row,
        col=output_col,
        input_row=input_row,
        input_col=input_col,
        patch=patch,
        kernel=kernel,
        products=products,
        sum_value=sum_value,
    )


def conv_full(
    image: list[list[float]],
    kernel: list[list[float]],
    padding: int = 0,
    stride: int = 1,
) -> ConvResult:
    """Compute full 2D convolution.

    Args:
        image: 2D input image
        kernel: 2D convolution kernel
        padding: Padding amount
        stride: Stride value

    Returns:
        ConvResult with activation map and metadata
    """
    height = len(image)
    width = len(image[0]) if height > 0 else 0
    kernel_height = len(kernel)
    kernel_width = len(kernel[0]) if kernel_height > 0 else 0

    # Compute output dimensions
    out_height = compute_output_size(height, kernel_height, padding, stride)
    out_width = compute_output_size(width, kernel_width, padding, stride)

    # Pad image once
    padded = pad_image(image, padding)

    # Compute activation map
    activation_map: list[list[float]] = []
    for out_r in range(out_height):
        row: list[float] = []
        for out_c in range(out_width):
            # Get input position
            in_r = out_r * stride
            in_c = out_c * stride

            # Extract patch and compute dot product
            patch = extract_patch(padded, in_r, in_c, kernel_height)
            products = elementwise_product(patch, kernel)
            activation = sum_matrix(products)
            row.append(activation)
        activation_map.append(row)

    return ConvResult(
        input_shape=(height, width),
        kernel_shape=(kernel_height, kernel_width),
        output_shape=(out_height, out_width),
        padding=padding,
        stride=stride,
        activation_map=activation_map,
    )


def create_sample_image(pattern: str, size: int = 8) -> list[list[float]]:
    """Create a sample image for demonstration.

    Args:
        pattern: One of 'gradient', 'checkerboard', 'digit', 'cross', 'edges'
        size: Image size (size × size)

    Returns:
        2D grayscale image with values 0-255
    """
    if pattern == "gradient":
        # Horizontal gradient
        return [[c * 255 / (size - 1) for c in range(size)] for _ in range(size)]

    elif pattern == "checkerboard":
        # Alternating pattern
        image: list[list[float]] = []
        for r in range(size):
            row: list[float] = []
            for c in range(size):
                val = 255.0 if (r + c) % 2 == 0 else 0.0
                row.append(val)
            image.append(row)
        return image

    elif pattern == "cross":
        # Cross pattern in center
        image = [[0.0] * size for _ in range(size)]
        mid = size // 2
        for i in range(size):
            image[mid][i] = 255.0
            image[i][mid] = 255.0
        return image

    elif pattern == "edges":
        # Box with edges
        image = [[0.0] * size for _ in range(size)]
        for i in range(size):
            image[0][i] = 255.0
            image[size - 1][i] = 255.0
            image[i][0] = 255.0
            image[i][size - 1] = 255.0
        return image

    elif pattern == "diagonal":
        # Diagonal line
        image = [[0.0] * size for _ in range(size)]
        for i in range(size):
            image[i][i] = 255.0
            if i + 1 < size:
                image[i][i + 1] = 128.0
            if i > 0:
                image[i][i - 1] = 128.0
        return image

    elif pattern == "corner":
        # L-shape in corner
        image = [[0.0] * size for _ in range(size)]
        for i in range(size // 2 + 1):
            image[size // 4][i + size // 4] = 255.0
            image[i + size // 4][size // 4] = 255.0
        return image

    elif pattern == "digit_1":
        # Simple "1" digit
        image = [[0.0] * size for _ in range(size)]
        mid = size // 2
        for r in range(1, size - 1):
            image[r][mid] = 255.0
        image[1][mid - 1] = 128.0  # Top serif
        for c in range(mid - 1, mid + 2):
            image[size - 2][c] = 255.0  # Bottom base
        return image

    elif pattern == "digit_0":
        # Simple "0" digit (oval)
        image = [[0.0] * size for _ in range(size)]
        # Draw oval-ish shape
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                # Top and bottom curves
                if r == 1 or r == size - 2:
                    if 2 <= c <= size - 3:
                        image[r][c] = 255.0
                # Left and right sides
                elif (c == 1 or c == size - 2) and 2 <= r <= size - 3:
                    image[r][c] = 255.0
        return image

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def normalize_activation_map(
    activation_map: list[list[float]],
) -> list[list[float]]:
    """Normalize activation map to 0-255 range for visualization.

    Args:
        activation_map: Raw activation values

    Returns:
        Normalized values in [0, 255]
    """
    # Find min and max
    flat = [val for row in activation_map for val in row]
    min_val = min(flat)
    max_val = max(flat)

    # Normalize
    if max_val == min_val:
        return [[128.0] * len(row) for row in activation_map]

    normalized: list[list[float]] = []
    for row in activation_map:
        norm_row: list[float] = []
        for val in row:
            norm = (val - min_val) / (max_val - min_val) * 255
            norm_row.append(norm)
        normalized.append(norm_row)

    return normalized
