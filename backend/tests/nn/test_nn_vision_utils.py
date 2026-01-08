"""Tests for vision_utils module (Chapter 5)."""

import pytest

from backend.nn.vision_utils import (
    ALEXNET_LAYERS,
    ActivationInfo,
    AlexNetExtractor,
    FilterInfo,
    LayerInfo,
    activations_to_grid,
    create_sample_image,
    filters_to_grid,
    get_device,
)


class TestGetDevice:
    def test_returns_device(self) -> None:
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestAlexNetLayers:
    def test_has_five_layers(self) -> None:
        assert len(ALEXNET_LAYERS) == 5

    def test_layer_keys(self) -> None:
        assert set(ALEXNET_LAYERS.keys()) == {1, 2, 3, 4, 5}

    def test_layer_info_structure(self) -> None:
        for _layer_num, info in ALEXNET_LAYERS.items():
            assert "name" in info
            assert "filters" in info
            assert "kernel_size" in info
            assert "in_channels" in info

    def test_layer1_properties(self) -> None:
        layer1 = ALEXNET_LAYERS[1]
        assert layer1["name"] == "Conv1"
        assert layer1["filters"] == 64
        assert layer1["kernel_size"] == 11
        assert layer1["in_channels"] == 3


class TestCreateSampleImage:
    def test_gradient_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "gradient")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_checkerboard_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "checkerboard")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_noise_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "noise")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_face_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "face")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_cat_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "cat")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_car_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "car")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_landscape_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "landscape")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_invalid_pattern(self) -> None:
        with pytest.raises(ValueError, match="Unknown pattern"):
            create_sample_image(224, 224, "invalid")

    def test_custom_size(self) -> None:
        img_bytes = create_sample_image(100, 100, "gradient")
        assert isinstance(img_bytes, bytes)


class TestAlexNetExtractor:
    @pytest.fixture
    def extractor(self) -> AlexNetExtractor:
        return AlexNetExtractor()

    def test_model_loads(self, extractor: AlexNetExtractor) -> None:
        model = extractor.model
        assert model is not None

    def test_model_is_cached(self, extractor: AlexNetExtractor) -> None:
        model1 = extractor.model
        model2 = extractor.model
        assert model1 is model2

    def test_get_layer_info(self, extractor: AlexNetExtractor) -> None:
        layers = extractor.get_layer_info()
        assert len(layers) == 5
        for layer in layers:
            assert isinstance(layer, LayerInfo)

    def test_get_conv_layer_valid(self, extractor: AlexNetExtractor) -> None:
        for layer_num in range(1, 6):
            conv = extractor.get_conv_layer(layer_num)
            assert conv is not None

    def test_get_conv_layer_invalid(self, extractor: AlexNetExtractor) -> None:
        with pytest.raises(ValueError, match="Invalid layer"):
            extractor.get_conv_layer(0)
        with pytest.raises(ValueError, match="Invalid layer"):
            extractor.get_conv_layer(6)

    def test_extract_filters_layer1(self, extractor: AlexNetExtractor) -> None:
        filters = extractor.extract_filters(layer=1, max_filters=8)
        assert len(filters) == 8
        for f in filters:
            assert isinstance(f, FilterInfo)
            assert f.layer == 1
            assert f.kernel_size == 11
            assert len(f.image_base64) > 0

    def test_extract_filters_layer2(self, extractor: AlexNetExtractor) -> None:
        filters = extractor.extract_filters(layer=2, max_filters=8)
        assert len(filters) == 8
        for f in filters:
            assert f.layer == 2
            assert f.kernel_size == 5

    def test_extract_filters_max_limit(self, extractor: AlexNetExtractor) -> None:
        filters = extractor.extract_filters(layer=1, max_filters=4)
        assert len(filters) == 4

    def test_extract_activations(self, extractor: AlexNetExtractor) -> None:
        img_bytes = create_sample_image(224, 224, "gradient")
        activations = extractor.extract_activations(img_bytes, layer=1, max_activations=8)
        assert len(activations) == 8
        for a in activations:
            assert isinstance(a, ActivationInfo)
            assert a.layer == 1
            assert a.max_activation >= 0
            assert len(a.image_base64) > 0

    def test_extract_activations_layer2(self, extractor: AlexNetExtractor) -> None:
        img_bytes = create_sample_image(224, 224, "checkerboard")
        activations = extractor.extract_activations(img_bytes, layer=2, max_activations=4)
        assert len(activations) == 4
        for a in activations:
            assert a.layer == 2

    def test_extract_activations_invalid_layer(self, extractor: AlexNetExtractor) -> None:
        img_bytes = create_sample_image(224, 224, "gradient")
        with pytest.raises(ValueError, match="Invalid layer"):
            extractor.extract_activations(img_bytes, layer=0)

    def test_get_param_count(self, extractor: AlexNetExtractor) -> None:
        count = extractor.get_param_count()
        # AlexNet has ~61M parameters
        assert count > 60_000_000
        assert count < 70_000_000


class TestFiltersToGrid:
    def test_basic_grid(self) -> None:
        filters = [
            FilterInfo(layer=1, index=i, kernel_size=11, image_base64=f"img{i}")
            for i in range(16)
        ]
        grid = filters_to_grid(filters, cols=4)
        assert grid["rows"] == 4
        assert grid["cols"] == 4
        assert grid["count"] == 16
        assert len(grid["filters"]) == 16

    def test_partial_last_row(self) -> None:
        filters = [
            FilterInfo(layer=1, index=i, kernel_size=11, image_base64=f"img{i}")
            for i in range(10)
        ]
        grid = filters_to_grid(filters, cols=4)
        assert grid["rows"] == 3  # 10 / 4 = 2.5, rounds up to 3
        assert grid["count"] == 10


class TestActivationsToGrid:
    def test_basic_grid(self) -> None:
        activations = [
            ActivationInfo(layer=1, filter_index=i, width=55, height=55, max_activation=1.0, image_base64=f"img{i}")
            for i in range(16)
        ]
        grid = activations_to_grid(activations, cols=4)
        assert grid["rows"] == 4
        assert grid["cols"] == 4
        assert grid["count"] == 16
        assert len(grid["activations"]) == 16

