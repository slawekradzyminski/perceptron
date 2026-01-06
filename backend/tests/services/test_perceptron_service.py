"""Tests for PerceptronService."""

import pytest

from backend.services.perceptron_service import PerceptronService


class TestPerceptronServiceInit:
    def test_init_default(self) -> None:
        service = PerceptronService()
        state = service.state()
        assert state["dataset"] == "or"
        assert state["lr"] == 1.0
        assert len(state["w"]) == 2

    def test_init_with_dataset(self) -> None:
        service = PerceptronService(dataset="xor")
        state = service.state()
        assert state["dataset"] == "xor"

    def test_init_with_lr(self) -> None:
        service = PerceptronService(lr=0.5)
        state = service.state()
        assert state["lr"] == 0.5

    def test_init_with_seed(self) -> None:
        service = PerceptronService(seed=42)
        # Two services with same seed should have same initial weights
        service2 = PerceptronService(seed=42)
        assert service.state()["w"] == service2.state()["w"]


class TestPerceptronServiceSetLr:
    def test_set_lr(self) -> None:
        service = PerceptronService()
        service.set_lr(0.3)
        assert service.lr == 0.3
        assert service.perceptron.lr == 0.3


class TestPerceptronServiceSetDataset:
    def test_set_or_dataset(self) -> None:
        service = PerceptronService()
        service.set_dataset("or")
        state = service.state()
        assert state["dataset"] == "or"
        assert state["sample_count"] == 4

    def test_set_xor_dataset(self) -> None:
        service = PerceptronService()
        service.set_dataset("xor")
        state = service.state()
        assert state["dataset"] == "xor"
        assert state["sample_count"] == 4

    def test_set_invalid_dataset(self) -> None:
        service = PerceptronService()
        with pytest.raises(ValueError, match="dataset must be"):
            service.set_dataset("invalid")

    def test_set_custom_dataset(self) -> None:
        service = PerceptronService()
        custom_samples = [
            {"x": [1, 1], "y": 1},
            {"x": [-1, -1], "y": -1},
        ]
        service.set_dataset("custom", custom=(custom_samples, (1, 2)))
        state = service.state()
        assert state["dataset"] == "custom"
        assert state["sample_count"] == 2

    def test_set_custom_without_data(self) -> None:
        service = PerceptronService()
        with pytest.raises(ValueError, match="custom dataset requires"):
            service.set_dataset("custom")


class TestPerceptronServiceStep:
    def test_step_returns_result(self) -> None:
        service = PerceptronService()
        result = service.step()
        assert "w" in result
        assert "b" in result
        assert "x" in result
        assert "y" in result
        assert "score" in result
        assert "pred" in result
        assert "mistake" in result
        assert "delta_w" in result
        assert "delta_b" in result
        assert "idx" in result

    def test_step_increments_idx(self) -> None:
        service = PerceptronService()
        service.step()
        assert service.idx == 1

    def test_step_wraps_idx(self) -> None:
        service = PerceptronService()
        for _ in range(4):
            service.step()
        assert service.idx == 0

    def test_step_provides_next_sample(self) -> None:
        service = PerceptronService()
        result = service.step()
        assert "next_x" in result
        assert "next_y" in result


class TestPerceptronServiceReset:
    def test_reset_restores_initial_state(self) -> None:
        service = PerceptronService()
        service.step()
        service.step()
        service.reset()
        state = service.state()
        assert state["idx"] == 0
        assert all(w == 0 for w in state["w"])
        assert state["b"] == 0


class TestPerceptronServiceState:
    def test_state_structure(self) -> None:
        service = PerceptronService()
        state = service.state()
        assert "w" in state
        assert "b" in state
        assert "idx" in state
        assert "dataset" in state
        assert "lr" in state
        assert "next_x" in state
        assert "next_y" in state
        assert "grid_rows" in state
        assert "grid_cols" in state
        assert "sample_count" in state

    def test_state_grid_shape(self) -> None:
        service = PerceptronService()
        state = service.state()
        assert state["grid_rows"] == 1
        assert state["grid_cols"] == 2

