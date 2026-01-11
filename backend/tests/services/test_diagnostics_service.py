"""Tests for DiagnosticsService."""

import pytest

from backend.services.diagnostics_service import DiagnosticsService


@pytest.fixture
def service() -> DiagnosticsService:
    return DiagnosticsService()


@pytest.fixture
def or_samples() -> list[dict]:
    return [
        {"x": [-1.0, -1.0], "y": -1},
        {"x": [-1.0, 1.0], "y": 1},
        {"x": [1.0, -1.0], "y": 1},
        {"x": [1.0, 1.0], "y": 1},
    ]


class TestErrorSurface:
    def test_basic_surface(self, service: DiagnosticsService, or_samples: list[dict]) -> None:
        """Test error surface computation returns expected shape."""
        grid = service.error_surface(
            samples=or_samples,
            w_min=-1.0,
            w_max=1.0,
            steps=10,
            b=0.0,
        )
        assert len(grid) == 10
        assert len(grid[0]) == 10
        assert all(isinstance(row, list) for row in grid)
        assert all(isinstance(val, float) for row in grid for val in row)

    def test_surface_values_non_negative(self, service: DiagnosticsService, or_samples: list[dict]) -> None:
        """MSE should always be non-negative."""
        grid = service.error_surface(
            samples=or_samples,
            w_min=-2.0,
            w_max=2.0,
            steps=5,
            b=0.0,
        )
        for row in grid:
            for val in row:
                assert val >= 0


class TestMlpInternals:
    def test_basic_internals(self, service: DiagnosticsService, or_samples: list[dict]) -> None:
        """Test MLP internals computation returns expected structure."""
        result = service.mlp_internals(
            samples=or_samples,
            rows=1,
            cols=2,
            hidden_dim=2,
            lr=0.5,
            seed=42,
            sample_index=0,
        )

        # Check top-level keys
        assert "hidden_dim" in result
        assert "grid_rows" in result
        assert "grid_cols" in result
        assert "sample_index" in result
        assert "sample_count" in result
        assert "x" in result
        assert "y" in result
        assert "loss" in result
        assert "p_hat" in result

        # Check nested structures
        assert "hidden" in result
        assert "output" in result
        assert "gradients" in result

        # Check hidden layer structure
        hidden = result["hidden"]
        assert "weights_before" in hidden
        assert "weights_after" in hidden
        assert "templates_before" in hidden
        assert "templates_after" in hidden

    def test_sample_index_wrapping(self, service: DiagnosticsService, or_samples: list[dict]) -> None:
        """Test that sample index wraps correctly."""
        result = service.mlp_internals(
            samples=or_samples,
            rows=1,
            cols=2,
            hidden_dim=2,
            lr=0.5,
            seed=42,
            sample_index=10,  # Should wrap to 10 % 4 = 2
        )
        assert result["sample_index"] == 2

    def test_templates_shape(self, service: DiagnosticsService, or_samples: list[dict]) -> None:
        """Test that templates have correct shape."""
        hidden_dim = 3
        result = service.mlp_internals(
            samples=or_samples,
            rows=1,
            cols=2,
            hidden_dim=hidden_dim,
            lr=0.5,
            seed=42,
            sample_index=0,
        )

        templates = result["hidden"]["templates_before"]
        assert len(templates) == hidden_dim
        # Each template should be rows x cols
        for template in templates:
            assert len(template) == 1  # rows
            assert len(template[0]) == 2  # cols
