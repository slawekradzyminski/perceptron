"""Tests for /alexnet API routes (Chapter 5)."""

import io

from fastapi.testclient import TestClient

from backend.api_app import app
from backend.nn.vision_utils import create_sample_image

client = TestClient(app)


class TestAlexNetState:
    def test_get_state(self) -> None:
        response = client.get("/alexnet/state")
        assert response.status_code == 200
        data = response.json()
        assert "current_layer" in data
        assert "layers" in data
        assert "sample_images" in data
        assert "param_count" in data

    def test_state_has_five_layers(self) -> None:
        response = client.get("/alexnet/state")
        assert response.status_code == 200
        data = response.json()
        assert len(data["layers"]) == 5


class TestAlexNetLayers:
    def test_get_layers(self) -> None:
        response = client.get("/alexnet/layers")
        assert response.status_code == 200
        data = response.json()
        assert "layers" in data
        assert "param_count" in data
        assert len(data["layers"]) == 5

    def test_layer_info_structure(self) -> None:
        response = client.get("/alexnet/layers")
        assert response.status_code == 200
        data = response.json()
        for layer in data["layers"]:
            assert "layer" in layer
            assert "name" in layer
            assert "filters" in layer
            assert "kernel_size" in layer


class TestAlexNetSetLayer:
    def test_set_valid_layer(self) -> None:
        for layer in range(1, 6):
            response = client.post(f"/alexnet/layer/{layer}")
            assert response.status_code == 200
            data = response.json()
            assert data["current_layer"] == layer

    def test_set_invalid_layer_zero(self) -> None:
        response = client.post("/alexnet/layer/0")
        assert response.status_code == 400

    def test_set_invalid_layer_six(self) -> None:
        response = client.post("/alexnet/layer/6")
        assert response.status_code == 400


class TestAlexNetFilters:
    def test_get_filters_default(self) -> None:
        response = client.get("/alexnet/filters")
        assert response.status_code == 200
        data = response.json()
        assert "filters" in data
        assert "layer" in data
        assert "layer_info" in data
        assert data["layer"] == 1

    def test_get_filters_specific_layer(self) -> None:
        response = client.get("/alexnet/filters?layer=2")
        assert response.status_code == 200
        data = response.json()
        assert data["layer"] == 2

    def test_get_filters_max_limit(self) -> None:
        response = client.get("/alexnet/filters?max_filters=8")
        assert response.status_code == 200
        data = response.json()
        assert len(data["filters"]) == 8

    def test_filter_has_image(self) -> None:
        response = client.get("/alexnet/filters?max_filters=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["filters"]) == 1
        assert "image" in data["filters"][0]
        assert len(data["filters"][0]["image"]) > 0  # base64 string

    def test_get_filters_invalid_layer(self) -> None:
        response = client.get("/alexnet/filters?layer=0")
        assert response.status_code == 422  # Validation error


class TestAlexNetActivationsSample:
    def test_get_activations_default(self) -> None:
        response = client.get("/alexnet/activations")
        assert response.status_code == 200
        data = response.json()
        assert "activations" in data
        assert "layer" in data

    def test_get_activations_sample_gradient(self) -> None:
        response = client.get("/alexnet/activations?sample=gradient")
        assert response.status_code == 200
        data = response.json()
        assert data["sample_name"] == "gradient"

    def test_get_activations_sample_checkerboard(self) -> None:
        response = client.get("/alexnet/activations?sample=checkerboard")
        assert response.status_code == 200
        data = response.json()
        assert data["sample_name"] == "checkerboard"

    def test_get_activations_specific_layer(self) -> None:
        response = client.get("/alexnet/activations?layer=2")
        assert response.status_code == 200
        data = response.json()
        assert data["layer"] == 2

    def test_get_activations_max_limit(self) -> None:
        response = client.get("/alexnet/activations?max_activations=4")
        assert response.status_code == 200
        data = response.json()
        assert len(data["activations"]) == 4

    def test_activation_has_image(self) -> None:
        response = client.get("/alexnet/activations?max_activations=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["activations"]) == 1
        assert "image" in data["activations"][0]


class TestAlexNetActivationsUpload:
    def test_upload_image(self) -> None:
        img_bytes = create_sample_image(224, 224, "gradient")
        files = {"file": ("test.png", io.BytesIO(img_bytes), "image/png")}
        response = client.post("/alexnet/activations", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "activations" in data

    def test_upload_with_layer(self) -> None:
        img_bytes = create_sample_image(224, 224, "checkerboard")
        files = {"file": ("test.png", io.BytesIO(img_bytes), "image/png")}
        response = client.post("/alexnet/activations?layer=2", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["layer"] == 2

    def test_upload_invalid_type(self) -> None:
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        response = client.post("/alexnet/activations", files=files)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]


class TestAlexNetSampleImage:
    def test_get_sample_gradient(self) -> None:
        response = client.get("/alexnet/sample/gradient")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "gradient"
        assert "image" in data
        assert data["content_type"] == "image/png"

    def test_get_sample_checkerboard(self) -> None:
        response = client.get("/alexnet/sample/checkerboard")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "checkerboard"

    def test_get_sample_noise(self) -> None:
        response = client.get("/alexnet/sample/noise")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "noise"

    def test_get_sample_invalid(self) -> None:
        response = client.get("/alexnet/sample/nonexistent")
        assert response.status_code == 400

