"""Tests for AlexNetService (Chapter 5)."""

import pytest

from backend.nn.vision_utils import create_sample_image
from backend.services.alexnet_service import SAMPLE_IMAGES, AlexNetService


class TestAlexNetServiceInit:
    def test_init_creates_service(self) -> None:
        service = AlexNetService()
        assert service is not None

    def test_initial_layer(self) -> None:
        service = AlexNetService()
        state = service.state()
        assert state["current_layer"] == 1


class TestAlexNetServiceState:
    def test_state_has_layers(self) -> None:
        service = AlexNetService()
        state = service.state()
        assert "layers" in state
        assert len(state["layers"]) == 5

    def test_state_has_sample_images(self) -> None:
        service = AlexNetService()
        state = service.state()
        assert "sample_images" in state
        assert state["sample_images"] == SAMPLE_IMAGES

    def test_state_has_param_count(self) -> None:
        service = AlexNetService()
        state = service.state()
        assert "param_count" in state
        assert state["param_count"] > 60_000_000


class TestAlexNetServiceSetLayer:
    def test_set_valid_layer(self) -> None:
        service = AlexNetService()
        for layer in range(1, 6):
            state = service.set_layer(layer)
            assert state["current_layer"] == layer

    def test_set_invalid_layer(self) -> None:
        service = AlexNetService()
        with pytest.raises(ValueError, match="Invalid layer"):
            service.set_layer(0)
        with pytest.raises(ValueError, match="Invalid layer"):
            service.set_layer(6)


class TestAlexNetServiceGetFilters:
    def test_get_filters_default(self) -> None:
        service = AlexNetService()
        result = service.get_filters()
        assert "filters" in result
        assert result["layer"] == 1
        assert "layer_info" in result

    def test_get_filters_specific_layer(self) -> None:
        service = AlexNetService()
        result = service.get_filters(layer=2)
        assert result["layer"] == 2
        assert result["layer_info"]["name"] == "Conv2"

    def test_get_filters_max_limit(self) -> None:
        service = AlexNetService()
        result = service.get_filters(layer=1, max_filters=8)
        assert len(result["filters"]) == 8

    def test_get_filters_invalid_layer(self) -> None:
        service = AlexNetService()
        with pytest.raises(ValueError, match="Invalid layer"):
            service.get_filters(layer=0)


class TestAlexNetServiceGetActivations:
    def test_get_activations_sample(self) -> None:
        service = AlexNetService()
        result = service.get_activations(sample_name="gradient")
        assert "activations" in result
        assert result["layer"] == 1
        assert result["sample_name"] == "gradient"

    def test_get_activations_different_samples(self) -> None:
        service = AlexNetService()
        for sample in SAMPLE_IMAGES:
            result = service.get_activations(sample_name=sample)
            assert result["sample_name"] == sample

    def test_get_activations_specific_layer(self) -> None:
        service = AlexNetService()
        result = service.get_activations(sample_name="gradient", layer=2)
        assert result["layer"] == 2

    def test_get_activations_max_limit(self) -> None:
        service = AlexNetService()
        result = service.get_activations(sample_name="gradient", max_activations=4)
        assert len(result["activations"]) == 4

    def test_get_activations_custom_image(self) -> None:
        service = AlexNetService()
        img_bytes = create_sample_image(224, 224, "noise")
        result = service.get_activations(image_bytes=img_bytes)
        assert "activations" in result
        assert result["sample_name"] is None  # Not a named sample

    def test_get_activations_includes_input_image(self) -> None:
        service = AlexNetService()
        result = service.get_activations(sample_name="gradient")
        assert "input_image" in result
        assert isinstance(result["input_image"], str)
        assert len(result["input_image"]) > 0  # Base64 encoded

    def test_get_activations_invalid_sample(self) -> None:
        service = AlexNetService()
        with pytest.raises(ValueError, match="Unknown sample"):
            service.get_activations(sample_name="nonexistent")


class TestAlexNetServiceGetSampleImage:
    def test_get_sample_image(self) -> None:
        service = AlexNetService()
        for sample in SAMPLE_IMAGES:
            img_bytes = service.get_sample_image(sample)
            assert isinstance(img_bytes, bytes)
            assert len(img_bytes) > 0

    def test_get_sample_image_cached(self) -> None:
        service = AlexNetService()
        img1 = service.get_sample_image("gradient")
        img2 = service.get_sample_image("gradient")
        assert img1 is img2  # Same object (cached)

    def test_get_sample_image_invalid(self) -> None:
        service = AlexNetService()
        with pytest.raises(ValueError, match="Unknown sample"):
            service.get_sample_image("nonexistent")

