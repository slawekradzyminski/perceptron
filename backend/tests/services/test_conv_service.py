"""Tests for ConvService (Chapter 6)."""

import pytest

from backend.services.conv_service import ConvService


class TestConvServiceState:
    """Tests for ConvService state management."""

    def test_initial_state(self) -> None:
        """Test initial state is valid."""
        service = ConvService()
        state = service.state()

        assert "image" in state
        assert "kernel" in state
        assert "padding" in state
        assert "stride" in state
        assert "activation_map" in state
        assert "output_shape" in state

    def test_initial_kernel_is_sobel_x(self) -> None:
        """Test initial kernel is sobel_x."""
        service = ConvService()
        state = service.state()
        assert state["kernel_name"] == "sobel_x"

    def test_initial_image_is_cross(self) -> None:
        """Test initial image is cross pattern."""
        service = ConvService()
        state = service.state()
        assert state["image_name"] == "cross"


class TestConvServiceSetImage:
    """Tests for set_image method."""

    def test_set_valid_image(self) -> None:
        """Test setting a valid preset image."""
        service = ConvService()
        state = service.set_image("gradient")

        assert state["image_name"] == "gradient"

    def test_set_invalid_image(self) -> None:
        """Test setting an invalid image raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="Unknown image"):
            service.set_image("nonexistent")


class TestConvServiceSetKernel:
    """Tests for set_kernel method."""

    def test_set_preset_kernel(self) -> None:
        """Test setting a preset kernel."""
        service = ConvService()
        state = service.set_kernel(name="blur")

        assert state["kernel_name"] == "blur"

    def test_set_custom_kernel(self) -> None:
        """Test setting a custom kernel."""
        service = ConvService()
        custom = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
        state = service.set_kernel(weights=custom)

        assert state["kernel_name"] == "custom"
        assert state["kernel"] == custom

    def test_set_invalid_kernel_size(self) -> None:
        """Test that non-3x3 kernel raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="3x3"):
            service.set_kernel(weights=[[1, 2], [3, 4]])

    def test_set_invalid_preset(self) -> None:
        """Test setting an invalid preset raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="Unknown kernel"):
            service.set_kernel(name="nonexistent")

    def test_set_kernel_requires_name_or_weights(self) -> None:
        """Test that either name or weights must be provided."""
        service = ConvService()

        with pytest.raises(ValueError, match="Must provide"):
            service.set_kernel()


class TestConvServiceSetParams:
    """Tests for set_params method."""

    def test_set_padding(self) -> None:
        """Test setting padding."""
        service = ConvService()
        state = service.set_params(padding=2)

        assert state["padding"] == 2

    def test_set_stride(self) -> None:
        """Test setting stride."""
        service = ConvService()
        state = service.set_params(stride=2)

        assert state["stride"] == 2

    def test_set_both_params(self) -> None:
        """Test setting both padding and stride."""
        service = ConvService()
        state = service.set_params(padding=2, stride=2)

        assert state["padding"] == 2
        assert state["stride"] == 2

    def test_invalid_padding(self) -> None:
        """Test that invalid padding raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="Padding"):
            service.set_params(padding=10)

    def test_invalid_stride(self) -> None:
        """Test that invalid stride raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="Stride"):
            service.set_params(stride=0)


class TestConvServiceGetStep:
    """Tests for get_step method."""

    def test_get_step_valid_position(self) -> None:
        """Test getting step at valid position."""
        service = ConvService()
        step = service.get_step(0, 0)

        assert "output_row" in step
        assert "output_col" in step
        assert "patch" in step
        assert "kernel" in step
        assert "products" in step
        assert "sum" in step
        assert step["output_row"] == 0
        assert step["output_col"] == 0

    def test_get_step_invalid_position(self) -> None:
        """Test getting step at invalid position raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="out of range"):
            service.get_step(100, 100)


class TestConvServiceCalculateOutputSize:
    """Tests for calculate_output_size method."""

    def test_valid_calculation(self) -> None:
        """Test valid output size calculation."""
        service = ConvService()
        result = service.calculate_output_size(28, 3, 1, 1)

        assert result["valid"] is True
        assert result["output_size"] == 28
        assert result["error"] is None

    def test_invalid_calculation(self) -> None:
        """Test invalid calculation returns error."""
        service = ConvService()
        result = service.calculate_output_size(3, 10, 0, 1)

        assert result["valid"] is False
        assert result["output_size"] is None
        assert result["error"] is not None


class TestConvServiceGetPresets:
    """Tests for get_presets method."""

    def test_presets_structure(self) -> None:
        """Test presets returns correct structure."""
        service = ConvService()
        presets = service.get_presets()

        assert "images" in presets
        assert "kernels" in presets
        assert "kernel_weights" in presets
        assert len(presets["images"]) > 0
        assert len(presets["kernels"]) > 0


class TestConvServiceSetImageSize:
    """Tests for set_image_size method."""

    def test_set_valid_size(self) -> None:
        """Test setting valid image size."""
        service = ConvService()
        state = service.set_image_size(12)

        assert state["image_size"] == 12
        assert len(state["image"]) == 12
        assert len(state["image"][0]) == 12

    def test_invalid_size_too_small(self) -> None:
        """Test that size < 4 raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="between 4 and 28"):
            service.set_image_size(2)

    def test_invalid_size_too_large(self) -> None:
        """Test that size > 28 raises error."""
        service = ConvService()

        with pytest.raises(ValueError, match="between 4 and 28"):
            service.set_image_size(50)


class TestConvServiceImageToBase64:
    """Tests for image_to_base64 method."""

    def test_returns_base64_string(self) -> None:
        """Test that image_to_base64 returns a valid base64 string."""
        service = ConvService()
        result = service.image_to_base64()

        # Should be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

        # Should be valid base64 (can be decoded)
        import base64
        decoded = base64.b64decode(result)
        # PNG files start with specific bytes
        assert decoded[:4] == b"\x89PNG"


class TestConvServiceActivationToBase64:
    """Tests for activation_to_base64 method."""

    def test_returns_base64_string(self) -> None:
        """Test that activation_to_base64 returns a valid base64 string."""
        service = ConvService()
        result = service.activation_to_base64()

        # Should be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

        # Should be valid base64 (can be decoded)
        import base64
        decoded = base64.b64decode(result)
        # PNG files start with specific bytes
        assert decoded[:4] == b"\x89PNG"
