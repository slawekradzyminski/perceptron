"""Tests for convolution API routes (Chapter 6)."""

import pytest
from fastapi.testclient import TestClient

from backend.api_app import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestConvState:
    """Tests for GET /conv/state."""

    def test_get_state(self, client: TestClient) -> None:
        """Test getting convolution state."""
        response = client.get("/conv/state")
        assert response.status_code == 200

        data = response.json()
        assert "image" in data
        assert "kernel" in data
        assert "padding" in data
        assert "stride" in data
        assert "activation_map" in data
        assert "image_base64" in data
        assert "activation_base64" in data


class TestConvPresets:
    """Tests for GET /conv/presets."""

    def test_get_presets(self, client: TestClient) -> None:
        """Test getting available presets."""
        response = client.get("/conv/presets")
        assert response.status_code == 200

        data = response.json()
        assert "images" in data
        assert "kernels" in data
        assert "kernel_weights" in data


class TestConvSetImage:
    """Tests for POST /conv/image/{name}."""

    def test_set_valid_image(self, client: TestClient) -> None:
        """Test setting a valid preset image."""
        response = client.post("/conv/image/gradient")
        assert response.status_code == 200

        data = response.json()
        assert data["image_name"] == "gradient"

    def test_set_invalid_image(self, client: TestClient) -> None:
        """Test setting an invalid image returns error."""
        response = client.post("/conv/image/nonexistent")
        assert response.status_code == 400


class TestConvSetImageSize:
    """Tests for POST /conv/image/size."""

    def test_set_valid_size(self, client: TestClient) -> None:
        """Test setting valid image size."""
        response = client.post("/conv/image/size", json={"size": 12})
        assert response.status_code == 200

        data = response.json()
        assert data["image_size"] == 12

    def test_set_invalid_size(self, client: TestClient) -> None:
        """Test setting invalid size returns Pydantic validation error."""
        response = client.post("/conv/image/size", json={"size": 100})
        assert response.status_code == 422  # Pydantic validation error

    def test_missing_size(self, client: TestClient) -> None:
        """Test missing size returns Pydantic validation error."""
        response = client.post("/conv/image/size", json={})
        assert response.status_code == 422  # Pydantic validation error


class TestConvSetKernel:
    """Tests for POST /conv/kernel."""

    def test_set_preset_kernel(self, client: TestClient) -> None:
        """Test setting a preset kernel."""
        response = client.post("/conv/kernel", json={"name": "blur"})
        assert response.status_code == 200

        data = response.json()
        assert data["kernel_name"] == "blur"

    def test_set_custom_kernel(self, client: TestClient) -> None:
        """Test setting a custom kernel."""
        custom = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        response = client.post("/conv/kernel", json={"weights": custom})
        assert response.status_code == 200

        data = response.json()
        assert data["kernel_name"] == "custom"

    def test_missing_name_and_weights(self, client: TestClient) -> None:
        """Test missing both name and weights returns Pydantic validation error."""
        response = client.post("/conv/kernel", json={})
        assert response.status_code == 422  # Pydantic validation error


class TestConvSetParams:
    """Tests for POST /conv/params."""

    def test_set_padding(self, client: TestClient) -> None:
        """Test setting padding."""
        response = client.post("/conv/params", json={"padding": 2})
        assert response.status_code == 200

        data = response.json()
        assert data["padding"] == 2

    def test_set_stride(self, client: TestClient) -> None:
        """Test setting stride."""
        response = client.post("/conv/params", json={"stride": 2})
        assert response.status_code == 200

        data = response.json()
        assert data["stride"] == 2

    def test_invalid_padding(self, client: TestClient) -> None:
        """Test invalid padding returns Pydantic validation error."""
        response = client.post("/conv/params", json={"padding": 10})
        assert response.status_code == 422  # Pydantic validation error


class TestConvGetStep:
    """Tests for GET /conv/step."""

    def test_get_step_valid(self, client: TestClient) -> None:
        """Test getting step at valid position."""
        response = client.get("/conv/step?row=0&col=0")
        assert response.status_code == 200

        data = response.json()
        assert "patch" in data
        assert "kernel" in data
        assert "products" in data
        assert "sum" in data

    def test_get_step_invalid_position(self, client: TestClient) -> None:
        """Test getting step at invalid position returns error."""
        response = client.get("/conv/step?row=100&col=100")
        assert response.status_code == 400


class TestConvCalculate:
    """Tests for POST /conv/calculate."""

    def test_valid_calculation(self, client: TestClient) -> None:
        """Test valid output size calculation."""
        response = client.post(
            "/conv/calculate",
            json={"input_size": 28, "kernel_size": 3, "padding": 1, "stride": 1},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert data["output_size"] == 28

    def test_calculation_with_stride(self, client: TestClient) -> None:
        """Test calculation with stride 2."""
        response = client.post(
            "/conv/calculate",
            json={"input_size": 28, "kernel_size": 3, "padding": 1, "stride": 2},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert data["output_size"] == 14

    def test_missing_required_fields(self, client: TestClient) -> None:
        """Test missing required fields returns Pydantic validation error."""
        response = client.post("/conv/calculate", json={"input_size": 28})
        assert response.status_code == 422  # Pydantic validation error

    def test_invalid_calculation(self, client: TestClient) -> None:
        """Test invalid calculation returns valid=False."""
        response = client.post(
            "/conv/calculate",
            json={"input_size": 3, "kernel_size": 10, "padding": 0, "stride": 1},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is False
