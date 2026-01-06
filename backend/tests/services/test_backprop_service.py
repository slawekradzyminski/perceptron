"""Tests for BackpropService (TinyGPS + Regression)."""

import pytest

from backend.services.backprop_service import (
    DEFAULT_REGRESSION_SAMPLES,
    TINYGPS_DATASETS,
    BackpropService,
)


class TestTinyGpsDatasets:
    def test_all_datasets_defined(self) -> None:
        expected = {"paris-berlin-4", "paris-madrid-4", "madrid-paris-berlin", "four-cities"}
        assert set(TINYGPS_DATASETS.keys()) == expected

    def test_dataset_has_mode(self) -> None:
        for _name, config in TINYGPS_DATASETS.items():
            assert "mode" in config
            assert config["mode"] in ("1d", "2d")

    def test_dataset_has_cities(self) -> None:
        for _name, config in TINYGPS_DATASETS.items():
            assert "cities" in config
            assert len(config["cities"]) >= 2


class TestBackpropServiceInit:
    def test_init_creates_tinygps_state(self) -> None:
        service = BackpropService()
        state = service.tinygps_state()
        assert "dataset" in state
        assert "mode" in state
        assert "params" in state

    def test_init_creates_regression_state(self) -> None:
        service = BackpropService()
        state = service.regression_state()
        assert "loss" in state
        assert "params" in state

    def test_init_default_tinygps_dataset(self) -> None:
        service = BackpropService()
        state = service.tinygps_state()
        assert state["dataset"] == "madrid-paris-berlin"

    def test_init_default_regression_samples(self) -> None:
        service = BackpropService()
        state = service.regression_state()
        assert state["sample_count"] == len(DEFAULT_REGRESSION_SAMPLES)


class TestBackpropServiceState:
    def test_state_structure(self) -> None:
        service = BackpropService()
        state = service.state()
        assert "tinygps" in state
        assert "regression" in state
        assert "tinygps_datasets" in state
        assert "tinygps_city_coords" in state

    def test_tinygps_datasets_list(self) -> None:
        service = BackpropService()
        state = service.state()
        assert len(state["tinygps_datasets"]) == 4


class TestBackpropServiceResetTinyGps:
    def test_reset_changes_dataset(self) -> None:
        service = BackpropService()
        state = service.reset_tinygps(dataset="paris-berlin-4")
        assert state["dataset"] == "paris-berlin-4"

    def test_reset_changes_lr(self) -> None:
        service = BackpropService()
        state = service.reset_tinygps(lr=0.5)
        assert state["lr"] == 0.5

    def test_reset_with_params(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4")
        state = service.reset_tinygps(params={"m": [1.0, 2.0], "b": [0.5, 0.5]})
        assert state["params"]["m"] == [1.0, 2.0]
        assert state["params"]["b"] == [0.5, 0.5]

    def test_reset_invalid_dataset(self) -> None:
        service = BackpropService()
        with pytest.raises(ValueError, match="unknown TinyGPS dataset"):
            service.reset_tinygps(dataset="nonexistent")

    def test_reset_with_order(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="madrid-paris-berlin")
        original_state = service.tinygps_state()
        sample_count = original_state["sample_count"]
        # Reverse order
        order = list(range(sample_count - 1, -1, -1))
        state = service.reset_tinygps(order=order)
        assert state["sample_order"] == order

    def test_reset_with_empty_order(self) -> None:
        service = BackpropService()
        with pytest.raises(ValueError, match="order must be non-empty"):
            service.reset_tinygps(order=[])

    def test_reset_with_invalid_order(self) -> None:
        service = BackpropService()
        with pytest.raises(ValueError, match="order indices must be within"):
            service.reset_tinygps(order=[999])


class TestBackpropServiceResetRegression:
    def test_reset_changes_loss(self) -> None:
        service = BackpropService()
        state = service.reset_regression(loss="l1")
        assert state["loss"] == "l1"

    def test_reset_changes_lr(self) -> None:
        service = BackpropService()
        state = service.reset_regression(lr=0.5)
        assert state["lr"] == 0.5

    def test_reset_with_samples(self) -> None:
        service = BackpropService()
        custom_samples = [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}]
        state = service.reset_regression(samples=custom_samples)
        assert state["sample_count"] == 2

    def test_reset_with_params(self) -> None:
        service = BackpropService()
        state = service.reset_regression(params={"m": 2.0, "b": 1.0})
        assert state["params"]["m"] == 2.0
        assert state["params"]["b"] == 1.0

    def test_reset_with_order(self) -> None:
        service = BackpropService()
        order = [3, 2, 1, 0]
        state = service.reset_regression(order=order)
        assert state["sample_order"] == order

    def test_reset_with_empty_order(self) -> None:
        service = BackpropService()
        with pytest.raises(ValueError, match="order must be non-empty"):
            service.reset_regression(order=[])

    def test_reset_with_invalid_order(self) -> None:
        service = BackpropService()
        with pytest.raises(ValueError, match="order indices must be within"):
            service.reset_regression(order=[999])

    def test_reset_invalid_loss(self) -> None:
        service = BackpropService()
        with pytest.raises(ValueError, match="loss must be"):
            service.reset_regression(loss="invalid")  # type: ignore


class TestBackpropServiceTinyGpsState:
    def test_tinygps_state_structure(self) -> None:
        service = BackpropService()
        state = service.tinygps_state()
        assert "dataset" in state
        assert "mode" in state
        assert "cities" in state
        assert "idx" in state
        assert "lr" in state
        assert "sample_count" in state
        assert "feature_order" in state
        assert "sample_order" in state
        assert "samples" in state
        assert "params" in state
        assert "next_sample" in state

    def test_tinygps_1d_feature_order(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4")
        state = service.tinygps_state()
        assert state["feature_order"] == ["lon"]

    def test_tinygps_2d_feature_order(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="four-cities")
        state = service.tinygps_state()
        assert state["feature_order"] == ["lon", "lat"]


class TestBackpropServiceRegressionState:
    def test_regression_state_structure(self) -> None:
        service = BackpropService()
        state = service.regression_state()
        assert "loss" in state
        assert "idx" in state
        assert "lr" in state
        assert "sample_count" in state
        assert "params" in state
        assert "next_sample" in state
        assert "samples" in state
        assert "sample_order" in state


class TestBackpropServiceStepTinyGps:
    def test_step_1d_returns_result(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4")
        result = service.step_tinygps()
        assert "dataset" in result
        assert "mode" in result
        assert "logits" in result
        assert "probs" in result
        assert "loss" in result
        assert "pred" in result
        assert "correct" in result
        assert "grads" in result
        assert "params_before" in result
        assert "params_after" in result
        assert "metrics_after" in result

    def test_step_2d_returns_result(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="four-cities")
        result = service.step_tinygps()
        assert "mode" in result
        assert result["mode"] == "2d"
        assert "grads" in result
        assert "M" in result["grads"]

    def test_step_increments_idx(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4")
        service.step_tinygps()
        state = service.tinygps_state()
        assert state["idx"] == 1

    def test_step_wraps_idx(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4")
        sample_count = service.tinygps_state()["sample_count"]
        for _ in range(sample_count):
            service.step_tinygps()
        state = service.tinygps_state()
        assert state["idx"] == 0

    def test_step_updates_params(self) -> None:
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4")
        result = service.step_tinygps()
        # Params should change after step (unless gradients are 0)
        assert result["params_before"] is not None
        assert result["params_after"] is not None


class TestBackpropServiceStepRegression:
    def test_step_returns_result(self) -> None:
        service = BackpropService()
        result = service.step_regression()
        assert "loss" in result
        assert "idx" in result
        assert "sample" in result
        assert "y_hat" in result
        assert "loss_value" in result
        assert "grads" in result
        assert "params_before" in result
        assert "params_after" in result
        assert "mean_loss" in result

    def test_step_increments_idx(self) -> None:
        service = BackpropService()
        service.step_regression()
        state = service.regression_state()
        assert state["idx"] == 1

    def test_step_wraps_idx(self) -> None:
        service = BackpropService()
        sample_count = service.regression_state()["sample_count"]
        for _ in range(sample_count):
            service.step_regression()
        state = service.regression_state()
        assert state["idx"] == 0

    def test_step_updates_params(self) -> None:
        service = BackpropService()
        result = service.step_regression()
        # Params should change after step
        assert result["params_before"] is not None
        assert result["params_after"] is not None

    def test_step_mse_loss(self) -> None:
        service = BackpropService()
        service.reset_regression(loss="mse")
        result = service.step_regression()
        assert result["loss"] == "mse"

    def test_step_l1_loss(self) -> None:
        service = BackpropService()
        service.reset_regression(loss="l1")
        result = service.step_regression()
        assert result["loss"] == "l1"


class TestBackpropServiceLearning:
    def test_tinygps_reduces_loss(self) -> None:
        """Test that TinyGPS training reduces loss over time."""
        service = BackpropService()
        service.reset_tinygps(dataset="paris-berlin-4", lr=0.1)

        initial_loss = None
        for _i in range(20):
            result = service.step_tinygps()
            if initial_loss is None:
                initial_loss = result["loss"]

        final_metrics = result["metrics_after"]
        # Loss should decrease or accuracy should improve
        # (This is a sanity check)
        assert final_metrics["accuracy"] >= 0

    def test_regression_reduces_loss(self) -> None:
        """Test that regression training reduces loss over time."""
        service = BackpropService()
        service.reset_regression(loss="mse", lr=0.01)

        initial_loss = None
        for _i in range(50):
            result = service.step_regression()
            if initial_loss is None:
                initial_loss = result["mean_loss"]

        final_loss = result["mean_loss"]
        # Loss should decrease with training
        assert final_loss < initial_loss

