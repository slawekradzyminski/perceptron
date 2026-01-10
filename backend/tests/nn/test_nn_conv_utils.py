"""Tests for convolution utilities (Chapter 6)."""

import pytest

from backend.nn.conv_utils import (
    PRESET_KERNELS,
    compute_output_size,
    conv_full,
    conv_step,
    create_sample_image,
    elementwise_product,
    extract_patch,
    normalize_activation_map,
    pad_image,
    sum_matrix,
)


class TestComputeOutputSize:
    """Tests for compute_output_size function."""

    def test_basic_no_padding_stride_1(self) -> None:
        """Test basic convolution with no padding and stride 1."""
        # 8x8 input, 3x3 kernel, no padding, stride 1 -> 6x6 output
        result = compute_output_size(8, 3, 0, 1)
        assert result == 6

    def test_with_padding(self) -> None:
        """Test convolution with padding to preserve size."""
        # 8x8 input, 3x3 kernel, padding 1, stride 1 -> 8x8 output (same)
        result = compute_output_size(8, 3, 1, 1)
        assert result == 8

    def test_with_stride(self) -> None:
        """Test convolution with stride 2."""
        # 8x8 input, 3x3 kernel, no padding, stride 2 -> 3x3 output
        result = compute_output_size(8, 3, 0, 2)
        assert result == 3

    def test_with_padding_and_stride(self) -> None:
        """Test convolution with both padding and stride."""
        # 28x28 input, 5x5 kernel, padding 2, stride 2 -> 14x14 output
        result = compute_output_size(28, 5, 2, 2)
        assert result == 14

    def test_alexnet_conv1(self) -> None:
        """Test AlexNet Conv1 dimensions."""
        # 224x224 input, 11x11 kernel, no padding, stride 4 -> 54x54 output
        result = compute_output_size(224, 11, 0, 4)
        assert result == 54

    def test_invalid_stride(self) -> None:
        """Test that invalid stride raises error."""
        with pytest.raises(ValueError, match="Stride must be positive"):
            compute_output_size(8, 3, 0, 0)

    def test_invalid_kernel_size(self) -> None:
        """Test that invalid kernel size raises error."""
        with pytest.raises(ValueError, match="Kernel size must be positive"):
            compute_output_size(8, 0, 0, 1)

    def test_kernel_too_large(self) -> None:
        """Test that kernel too large for input raises error."""
        with pytest.raises(ValueError, match="too large"):
            compute_output_size(3, 5, 0, 1)


class TestPadImage:
    """Tests for pad_image function."""

    def test_no_padding(self) -> None:
        """Test that no padding returns original image."""
        image = [[1.0, 2.0], [3.0, 4.0]]
        result = pad_image(image, 0)
        assert result == image

    def test_padding_1(self) -> None:
        """Test padding of 1 around a 2x2 image."""
        image = [[1.0, 2.0], [3.0, 4.0]]
        result = pad_image(image, 1)

        # Result should be 4x4
        assert len(result) == 4
        assert len(result[0]) == 4

        # Corners should be 0
        assert result[0][0] == 0.0
        assert result[0][3] == 0.0
        assert result[3][0] == 0.0
        assert result[3][3] == 0.0

        # Center should be original
        assert result[1][1] == 1.0
        assert result[1][2] == 2.0
        assert result[2][1] == 3.0
        assert result[2][2] == 4.0


class TestExtractPatch:
    """Tests for extract_patch function."""

    def test_extract_top_left(self) -> None:
        """Test extracting patch from top-left."""
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        patch = extract_patch(image, 0, 0, 2)
        assert patch == [[1, 2], [4, 5]]

    def test_extract_bottom_right(self) -> None:
        """Test extracting patch from bottom-right."""
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        patch = extract_patch(image, 1, 1, 2)
        assert patch == [[5, 6], [8, 9]]


class TestElementwiseProduct:
    """Tests for elementwise_product function."""

    def test_basic_product(self) -> None:
        """Test basic element-wise multiplication."""
        patch = [[1.0, 2.0], [3.0, 4.0]]
        kernel = [[1.0, 0.0], [0.0, 1.0]]
        result = elementwise_product(patch, kernel)
        assert result == [[1.0, 0.0], [0.0, 4.0]]

    def test_all_ones_kernel(self) -> None:
        """Test with all-ones kernel (sum pooling)."""
        patch = [[1.0, 2.0], [3.0, 4.0]]
        kernel = [[1.0, 1.0], [1.0, 1.0]]
        result = elementwise_product(patch, kernel)
        assert result == patch


class TestSumMatrix:
    """Tests for sum_matrix function."""

    def test_basic_sum(self) -> None:
        """Test basic matrix sum."""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        assert sum_matrix(matrix) == 10.0

    def test_negative_values(self) -> None:
        """Test sum with negative values."""
        matrix = [[1.0, -1.0], [-1.0, 1.0]]
        assert sum_matrix(matrix) == 0.0


class TestConvStep:
    """Tests for conv_step function."""

    def test_identity_kernel(self) -> None:
        """Test with identity kernel."""
        image = [[0, 0, 0], [0, 100, 0], [0, 0, 0]]
        kernel = PRESET_KERNELS["identity"]
        # With padding=1, output[1,1] corresponds to input center
        result = conv_step(image, kernel, 1, 1, padding=1, stride=1)

        # Center value should be 100 * 1 = 100
        assert result.sum_value == 100.0

    def test_step_trace_structure(self) -> None:
        """Test that step returns proper structure."""
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        kernel = PRESET_KERNELS["identity"]
        result = conv_step(image, kernel, 0, 0, padding=1, stride=1)

        assert result.row == 0
        assert result.col == 0
        assert len(result.patch) == 3
        assert len(result.patch[0]) == 3
        assert len(result.products) == 3


class TestConvFull:
    """Tests for conv_full function."""

    def test_output_dimensions(self) -> None:
        """Test that output dimensions are correct."""
        image = [[float(i * 8 + j) for j in range(8)] for i in range(8)]
        kernel = PRESET_KERNELS["identity"]
        result = conv_full(image, kernel, padding=1, stride=1)

        assert result.input_shape == (8, 8)
        assert result.kernel_shape == (3, 3)
        assert result.output_shape == (8, 8)

    def test_no_padding_reduces_size(self) -> None:
        """Test that no padding reduces output size."""
        image = [[float(i * 8 + j) for j in range(8)] for i in range(8)]
        kernel = PRESET_KERNELS["identity"]
        result = conv_full(image, kernel, padding=0, stride=1)

        assert result.output_shape == (6, 6)

    def test_stride_2_halves_size(self) -> None:
        """Test that stride 2 approximately halves size."""
        image = [[float(i * 8 + j) for j in range(8)] for i in range(8)]
        kernel = PRESET_KERNELS["identity"]
        result = conv_full(image, kernel, padding=1, stride=2)

        assert result.output_shape == (4, 4)

    def test_identity_preserves_values(self) -> None:
        """Test that identity kernel preserves center values."""
        image = [[0, 0, 0], [0, 42, 0], [0, 0, 0]]
        kernel = PRESET_KERNELS["identity"]
        result = conv_full(image, kernel, padding=1, stride=1)

        # Center of output should be 42
        assert result.activation_map[1][1] == 42.0


class TestCreateSampleImage:
    """Tests for create_sample_image function."""

    def test_gradient(self) -> None:
        """Test gradient image creation."""
        image = create_sample_image("gradient", 8)
        assert len(image) == 8
        assert len(image[0]) == 8
        # First column should be 0, last should be 255
        assert image[0][0] == 0.0
        assert image[0][7] == 255.0

    def test_checkerboard(self) -> None:
        """Test checkerboard pattern."""
        image = create_sample_image("checkerboard", 4)
        assert image[0][0] == 255.0
        assert image[0][1] == 0.0
        assert image[1][0] == 0.0
        assert image[1][1] == 255.0

    def test_cross(self) -> None:
        """Test cross pattern."""
        image = create_sample_image("cross", 5)
        mid = 2
        # Center row and column should be white
        assert image[mid][0] == 255.0
        assert image[0][mid] == 255.0

    def test_invalid_pattern(self) -> None:
        """Test that invalid pattern raises error."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            create_sample_image("invalid")


class TestNormalizeActivationMap:
    """Tests for normalize_activation_map function."""

    def test_normalize_range(self) -> None:
        """Test that normalization produces 0-255 range."""
        activation = [[-100.0, 0.0], [50.0, 100.0]]
        result = normalize_activation_map(activation)

        # Min should be 0, max should be 255
        flat = [val for row in result for val in row]
        assert min(flat) == 0.0
        assert max(flat) == 255.0

    def test_constant_map(self) -> None:
        """Test normalization of constant activation map."""
        activation = [[5.0, 5.0], [5.0, 5.0]]
        result = normalize_activation_map(activation)

        # All values should be 128 (midpoint)
        for row in result:
            for val in row:
                assert val == 128.0


class TestPresetKernels:
    """Tests for preset kernels."""

    def test_all_kernels_are_3x3(self) -> None:
        """Test that all preset kernels are 3x3."""
        for name, kernel in PRESET_KERNELS.items():
            assert len(kernel) == 3, f"{name} kernel should have 3 rows"
            for row in kernel:
                assert len(row) == 3, f"{name} kernel rows should have 3 columns"

    def test_blur_sums_to_one(self) -> None:
        """Test that blur kernel sums to approximately 1."""
        blur = PRESET_KERNELS["blur"]
        total = sum(val for row in blur for val in row)
        assert abs(total - 1.0) < 0.0001

    def test_edge_kernels_sum_to_zero(self) -> None:
        """Test that edge detection kernels sum to approximately 0."""
        for name in ["edge_horizontal", "edge_vertical", "sobel_x", "sobel_y", "laplacian"]:
            kernel = PRESET_KERNELS[name]
            total = sum(val for row in kernel for val in row)
            assert abs(total) < 0.0001, f"{name} should sum to 0"
