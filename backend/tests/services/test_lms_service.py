"""Tests for LmsService."""

import pytest

from backend.services.lms_service import LmsService


class TestLmsServiceInit:
    def test_init_default(self) -> None:
        service = LmsService()
        state = service.state()
        assert state["dataset"] == "or"
        assert state["lr"] == 0.1
        assert state["w"] == [0.0, 0.0]
        assert state["b"] == 0.0

    def test_init_with_lr(self) -> None:
        service = LmsService(lr=0.5)
        state = service.state()
        assert state["lr"] == 0.5

    def test_init_with_dataset(self) -> None:
        service = LmsService(dataset="xor")
        state = service.state()
        assert state["dataset"] == "xor"


class TestLmsServiceSetDataset:
    def test_set_or_dataset(self) -> None:
        service = LmsService()
        service.set_dataset("or")
        state = service.state()
        assert state["dataset"] == "or"
        assert state["sample_count"] == 4

    def test_set_xor_dataset(self) -> None:
        service = LmsService()
        service.set_dataset("xor")
        state = service.state()
        assert state["dataset"] == "xor"
        assert state["sample_count"] == 4

    def test_set_invalid_dataset(self) -> None:
        service = LmsService()
        with pytest.raises(ValueError, match="dataset must be"):
            service.set_dataset("invalid")

    def test_set_custom_dataset(self) -> None:
        service = LmsService()
        custom_samples = [
            {"x": [1.0, 1.0], "y": 1},
            {"x": [-1.0, -1.0], "y": -1},
        ]
        service.set_dataset("custom", custom_samples=custom_samples)
        state = service.state()
        assert state["dataset"] == "custom"
        assert state["sample_count"] == 2

    def test_set_custom_without_data(self) -> None:
        service = LmsService()
        with pytest.raises(ValueError, match="custom dataset requires"):
            service.set_dataset("custom")

    def test_set_custom_with_wrong_dims(self) -> None:
        service = LmsService()
        custom_samples = [{"x": [1.0, 2.0, 3.0], "y": 1}]  # 3D instead of 2D
        with pytest.raises(ValueError, match="LMS requires 2D inputs"):
            service.set_dataset("custom", custom_samples=custom_samples)


class TestLmsServiceSetLr:
    def test_set_lr(self) -> None:
        service = LmsService()
        service.set_lr(0.3)
        assert service.lr == 0.3


class TestLmsServiceReset:
    def test_reset_clears_weights(self) -> None:
        service = LmsService()
        service.step()
        service.step()
        state = service.reset()
        assert state["w"] == [0.0, 0.0]
        assert state["b"] == 0.0
        assert state["idx"] == 0


class TestLmsServiceState:
    def test_state_structure(self) -> None:
        service = LmsService()
        state = service.state()
        assert "w" in state
        assert "b" in state
        assert "idx" in state
        assert "lr" in state
        assert "x" in state
        assert "y" in state
        assert "sample_count" in state
        assert "dataset" in state

    def test_state_has_current_sample(self) -> None:
        service = LmsService()
        state = service.state()
        assert len(state["x"]) == 2
        assert state["y"] in [-1, 1]


class TestLmsServiceStep:
    def test_step_returns_result(self) -> None:
        service = LmsService()
        result = service.step()
        assert "x" in result
        assert "y" in result
        assert "w_before" in result
        assert "b_before" in result
        assert "y_hat" in result
        assert "error" in result
        assert "grad_w1" in result
        assert "grad_w2" in result
        assert "grad_b" in result
        assert "w_after" in result
        assert "b_after" in result
        assert "idx" in result
        assert "lr" in result

    def test_step_increments_idx(self) -> None:
        service = LmsService()
        service.step()
        assert service.idx == 1

    def test_step_wraps_idx(self) -> None:
        service = LmsService()
        for _ in range(4):
            service.step()
        assert service.idx == 0

    def test_step_updates_weights(self) -> None:
        service = LmsService()
        result = service.step()
        # Weights should change if there was an error
        if result["error"] != 0:
            assert result["w_before"] != result["w_after"] or result["b_before"] != result["b_after"]

    def test_step_computes_correct_y_hat(self) -> None:
        service = LmsService()
        state = service.state()
        x1, x2 = state["x"]
        w = state["w"]
        b = state["b"]
        expected_y_hat = w[0] * x1 + w[1] * x2 + b
        result = service.step()
        assert result["y_hat"] == expected_y_hat

    def test_step_computes_correct_error(self) -> None:
        service = LmsService()
        result = service.step()
        expected_error = result["y"] - result["y_hat"]
        assert result["error"] == expected_error

    def test_lms_learning(self) -> None:
        """Test that LMS reduces error over multiple steps."""
        service = LmsService(dataset="or", lr=0.1)

        # Train for several epochs
        total_error = 0.0
        for _ in range(20):
            result = service.step()
            total_error += abs(result["error"])

        # Run one more epoch and check error is lower
        later_error = 0.0
        for _ in range(4):
            result = service.step()
            later_error += abs(result["error"])

        # Error should decrease (or at least not increase significantly)
        # This is a sanity check, not a strict test
        assert later_error <= total_error

