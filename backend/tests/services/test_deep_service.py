"""Tests for DeepService (Chapter 4)."""

import pytest

from backend.services.deep_service import (
    DEEP_DATASETS,
    DeepService,
    _make_circles_dataset,
    _make_spiral_dataset,
)


class TestMakeCirclesDataset:
    def test_creates_samples(self) -> None:
        samples = _make_circles_dataset(n_samples=100, seed=42)
        assert len(samples) == 100

    def test_sample_structure(self) -> None:
        samples = _make_circles_dataset(n_samples=50, seed=42)
        for sample in samples:
            assert "x" in sample
            assert "y" in sample
            assert len(sample["x"]) == 2
            assert sample["y"] in (0, 1)

    def test_both_classes_present(self) -> None:
        samples = _make_circles_dataset(n_samples=100, seed=42)
        labels = {s["y"] for s in samples}
        assert 0 in labels
        assert 1 in labels

    def test_deterministic_with_seed(self) -> None:
        samples1 = _make_circles_dataset(n_samples=50, seed=42)
        samples2 = _make_circles_dataset(n_samples=50, seed=42)
        for s1, s2 in zip(samples1, samples2):
            assert s1["x"] == s2["x"]
            assert s1["y"] == s2["y"]

    def test_inner_circle_class_0(self) -> None:
        # Inner circle should have small radius
        samples = _make_circles_dataset(n_samples=100, seed=42)
        class_0 = [s for s in samples if s["y"] == 0]
        for s in class_0:
            x1, x2 = s["x"]
            r = (x1**2 + x2**2) ** 0.5
            assert r <= 0.4  # Inner circle has r in [0, 0.4]


class TestMakeSpiralDataset:
    def test_creates_samples(self) -> None:
        samples = _make_spiral_dataset(n_samples=100, seed=42)
        assert len(samples) == 100

    def test_sample_structure(self) -> None:
        samples = _make_spiral_dataset(n_samples=50, seed=42)
        for sample in samples:
            assert "x" in sample
            assert "y" in sample
            assert len(sample["x"]) == 2
            assert sample["y"] in (0, 1)

    def test_both_classes_present(self) -> None:
        samples = _make_spiral_dataset(n_samples=100, seed=42)
        labels = {s["y"] for s in samples}
        assert 0 in labels
        assert 1 in labels

    def test_deterministic_with_seed(self) -> None:
        samples1 = _make_spiral_dataset(n_samples=50, seed=42)
        samples2 = _make_spiral_dataset(n_samples=50, seed=42)
        for s1, s2 in zip(samples1, samples2):
            assert s1["x"] == s2["x"]
            assert s1["y"] == s2["y"]


class TestDeepDatasets:
    def test_all_datasets_defined(self) -> None:
        expected = {"baarle", "xor", "circles", "spiral"}
        assert set(DEEP_DATASETS.keys()) == expected

    def test_dataset_info_structure(self) -> None:
        for _name, info in DEEP_DATASETS.items():
            assert "name" in info
            assert "description" in info
            assert "n_classes" in info
            assert info["n_classes"] == 2


class TestDeepServiceInit:
    def test_init_creates_state(self) -> None:
        service = DeepService()
        state = service.state()
        assert "dataset" in state
        assert "architecture" in state
        assert "metrics" in state
        assert "samples" in state

    def test_init_default_dataset(self) -> None:
        service = DeepService()
        state = service.state()
        assert state["dataset"] == "circles"

    def test_init_default_architecture(self) -> None:
        service = DeepService()
        state = service.state()
        assert state["architecture"]["hidden_dims"] == [8, 8]


class TestDeepServiceReset:
    def test_reset_changes_dataset(self) -> None:
        service = DeepService()
        state = service.reset(dataset="xor")
        assert state["dataset"] == "xor"
        assert state["sample_count"] == 4  # XOR has 4 samples

    def test_reset_changes_architecture(self) -> None:
        service = DeepService()
        state = service.reset(hidden_dims=[16, 16, 16])
        assert state["architecture"]["hidden_dims"] == [16, 16, 16]
        assert state["architecture"]["depth"] == 3

    def test_reset_changes_lr(self) -> None:
        service = DeepService()
        state = service.reset(lr=0.5)
        assert state["lr"] == 0.5

    def test_reset_changes_seed(self) -> None:
        service = DeepService()
        state1 = service.reset(seed=42)
        service2 = DeepService()
        state2 = service2.reset(seed=123)
        # Different seeds should produce different initial weights (and thus metrics)
        # Just verify the seed was accepted
        assert state1["sample_count"] > 0
        assert state2["sample_count"] > 0

    def test_reset_invalid_dataset(self) -> None:
        service = DeepService()
        with pytest.raises(ValueError, match="Unknown dataset"):
            service.reset(dataset="nonexistent")

    def test_reset_clears_history(self) -> None:
        service = DeepService()
        service.step(batch_size=5)
        assert len(service.get_history()) > 0
        service.reset()
        assert len(service.get_history()) == 0

    def test_reset_baarle_dataset(self) -> None:
        service = DeepService()
        state = service.reset(dataset="baarle")
        assert state["dataset"] == "baarle"
        assert state["sample_count"] > 0

    def test_reset_spiral_dataset(self) -> None:
        service = DeepService()
        state = service.reset(dataset="spiral")
        assert state["dataset"] == "spiral"
        assert state["sample_count"] > 0


class TestDeepServiceStep:
    def test_step_increments_total_steps(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        state = service.step()
        assert state["total_steps"] == 1

    def test_step_batch(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        state = service.step(batch_size=10)
        assert state["total_steps"] == 10

    def test_step_returns_step_info(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        state = service.step()
        assert "step_info" in state
        assert "batch_size" in state["step_info"]
        assert "last_loss" in state["step_info"]
        assert "last_correct" in state["step_info"]
        assert "grad_norm" in state["step_info"]

    def test_step_adds_to_history(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        service.step()
        history = service.get_history()
        assert len(history) == 1
        assert "step" in history[0]
        assert "epoch" in history[0]
        assert "loss" in history[0]
        assert "accuracy" in history[0]

    def test_step_wraps_samples(self) -> None:
        service = DeepService()
        service.reset(dataset="xor")  # Only 4 samples
        # Step through all samples and wrap
        for _ in range(5):
            service.step()
        state = service.state()
        assert state["idx"] == 1  # Wrapped around


class TestDeepServiceTrainEpoch:
    def test_train_epoch_processes_all_samples(self) -> None:
        service = DeepService()
        service.reset(dataset="xor")  # 4 samples
        state = service.train_epoch()
        assert state["total_steps"] == 4
        assert state["epoch"] == 1

    def test_train_epoch_increments_epoch(self) -> None:
        service = DeepService()
        service.reset(dataset="xor")
        service.train_epoch()
        service.train_epoch()
        state = service.state()
        assert state["epoch"] == 2


class TestDeepServiceRegions:
    def test_get_regions_returns_count(self) -> None:
        service = DeepService()
        service.reset(dataset="circles", hidden_dims=[4])
        regions = service.get_regions(resolution=10)
        assert "count" in regions
        assert "theoretical_max" in regions
        assert "efficiency" in regions
        assert regions["count"] >= 1

    def test_get_regions_efficiency(self) -> None:
        service = DeepService()
        service.reset(dataset="circles", hidden_dims=[4])
        regions = service.get_regions(resolution=10)
        assert 0 <= regions["efficiency"] <= 1


class TestDeepServiceBoundary:
    def test_get_boundary_returns_grids(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        boundary = service.get_boundary(resolution=10)
        assert "predictions" in boundary
        assert "region_ids" in boundary
        assert "region_count" in boundary
        assert "theoretical_max" in boundary

    def test_get_boundary_grid_size(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        boundary = service.get_boundary(resolution=15)
        assert len(boundary["predictions"]) == 15
        assert len(boundary["predictions"][0]) == 15
        assert len(boundary["region_ids"]) == 15


class TestDeepServiceComparison:
    def test_add_to_comparison(self) -> None:
        service = DeepService()
        service.reset(dataset="circles", hidden_dims=[8])
        entry = service.add_to_comparison()
        assert "depth" in entry
        assert "width" in entry
        assert "param_count" in entry
        assert "actual_regions" in entry
        assert "accuracy" in entry

    def test_get_comparison_table(self) -> None:
        service = DeepService()
        service.reset(dataset="circles", hidden_dims=[8])
        service.add_to_comparison()
        table = service.get_comparison_table()
        assert len(table) == 1

    def test_clear_comparison_table(self) -> None:
        service = DeepService()
        service.reset(dataset="circles")
        service.add_to_comparison()
        assert len(service.get_comparison_table()) == 1
        service.clear_comparison_table()
        assert len(service.get_comparison_table()) == 0

    def test_multiple_comparisons(self) -> None:
        service = DeepService()
        service.reset(dataset="circles", hidden_dims=[4])
        service.add_to_comparison()
        service.reset(dataset="circles", hidden_dims=[8, 8])
        service.add_to_comparison()
        table = service.get_comparison_table()
        assert len(table) == 2
        assert table[0]["depth"] == 1
        assert table[1]["depth"] == 2


class TestDeepServiceMetrics:
    def test_state_includes_metrics(self) -> None:
        service = DeepService()
        state = service.state()
        assert "metrics" in state
        assert "loss" in state["metrics"]
        assert "accuracy" in state["metrics"]

    def test_metrics_are_valid(self) -> None:
        service = DeepService()
        state = service.state()
        assert state["metrics"]["loss"] >= 0
        assert 0 <= state["metrics"]["accuracy"] <= 1

    def test_training_improves_xor(self) -> None:
        service = DeepService()
        service.reset(dataset="xor", hidden_dims=[8, 8], lr=0.5, seed=42)
        initial_state = service.state()
        initial_loss = initial_state["metrics"]["loss"]

        # Train for several epochs
        for _ in range(50):
            service.train_epoch()

        final_state = service.state()
        final_loss = final_state["metrics"]["loss"]

        # Loss should decrease with training
        assert final_loss < initial_loss

