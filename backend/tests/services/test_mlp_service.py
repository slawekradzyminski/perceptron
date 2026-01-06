"""Tests for MlpService."""

import pytest

from backend.services.mlp_service import MlpService, _pred_from_prob


class TestPredFromProb:
    def test_above_threshold(self) -> None:
        assert _pred_from_prob(0.8) == 1
        assert _pred_from_prob(0.51) == 1

    def test_below_threshold(self) -> None:
        assert _pred_from_prob(0.2) == -1
        assert _pred_from_prob(0.49) == -1

    def test_at_threshold(self) -> None:
        assert _pred_from_prob(0.5) == 1


class TestMlpServiceInit:
    def test_init_default(self) -> None:
        service = MlpService()
        snapshot = service.snapshot()
        assert snapshot["dataset"] == "xor"
        assert snapshot["hidden_dim"] == 2
        assert snapshot["lr"] == 0.5

    def test_init_with_dataset(self) -> None:
        service = MlpService(dataset="or")
        snapshot = service.snapshot()
        assert snapshot["dataset"] == "or"

    def test_init_with_hidden_dim(self) -> None:
        service = MlpService(hidden_dim=8)
        snapshot = service.snapshot()
        assert snapshot["hidden_dim"] == 8

    def test_init_with_lr(self) -> None:
        service = MlpService(lr=0.1)
        snapshot = service.snapshot()
        assert snapshot["lr"] == 0.1

    def test_init_with_seed(self) -> None:
        service1 = MlpService(seed=42)
        service2 = MlpService(seed=42)
        # Same seed should give same initial weights
        snapshot1 = service1.snapshot()
        snapshot2 = service2.snapshot()
        assert snapshot1["hidden"]["weights"] == snapshot2["hidden"]["weights"]


class TestMlpServiceSetHyperparams:
    def test_set_hidden_dim(self) -> None:
        service = MlpService()
        service.set_hyperparams(hidden_dim=16)
        assert service.hidden_dim == 16

    def test_set_invalid_hidden_dim(self) -> None:
        service = MlpService()
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            service.set_hyperparams(hidden_dim=0)

    def test_set_lr(self) -> None:
        service = MlpService()
        service.set_hyperparams(lr=0.01)
        assert service.lr == 0.01

    def test_set_seed(self) -> None:
        service = MlpService()
        service.set_hyperparams(seed=123)
        assert service.seed == 123


class TestMlpServiceSetDataset:
    def test_set_or_dataset(self) -> None:
        service = MlpService()
        service.set_dataset("or")
        snapshot = service.snapshot()
        assert snapshot["dataset"] == "or"
        assert snapshot["sample_count"] == 4

    def test_set_xor_dataset(self) -> None:
        service = MlpService()
        service.set_dataset("xor")
        snapshot = service.snapshot()
        assert snapshot["dataset"] == "xor"
        assert snapshot["sample_count"] == 4

    def test_set_invalid_dataset(self) -> None:
        service = MlpService()
        with pytest.raises(ValueError, match="dataset must be"):
            service.set_dataset("invalid")

    def test_set_custom_dataset(self) -> None:
        service = MlpService()
        custom_samples = [
            {"x": [1, 1], "y": 1},
            {"x": [-1, -1], "y": -1},
        ]
        service.set_dataset("custom", custom=(custom_samples, (1, 2)))
        snapshot = service.snapshot()
        assert snapshot["dataset"] == "custom"
        assert snapshot["sample_count"] == 2

    def test_set_custom_without_data(self) -> None:
        service = MlpService()
        with pytest.raises(ValueError, match="custom dataset requires"):
            service.set_dataset("custom")


class TestMlpServiceResetModel:
    def test_reset_model_clears_idx(self) -> None:
        service = MlpService()
        service.step()
        service.step()
        service.reset_model()
        assert service.idx == 0

    def test_reset_model_reinitializes_idx(self) -> None:
        service = MlpService(seed=42)

        # Train a bit
        for _ in range(10):
            service.step()

        # idx should have advanced
        assert service.idx != 0 or service.idx == 10 % 4

        # Reset
        service.reset_model()

        # idx should be back to 0
        assert service.idx == 0


class TestMlpServiceSnapshot:
    def test_snapshot_structure(self) -> None:
        service = MlpService()
        snapshot = service.snapshot()
        assert "dataset" in snapshot
        assert "grid_rows" in snapshot
        assert "grid_cols" in snapshot
        assert "hidden_dim" in snapshot
        assert "lr" in snapshot
        assert "seed" in snapshot
        assert "idx" in snapshot
        assert "sample_count" in snapshot
        assert "next_x" in snapshot
        assert "next_y" in snapshot
        assert "hidden" in snapshot
        assert "output" in snapshot
        assert "evals" in snapshot

    def test_snapshot_hidden_structure(self) -> None:
        service = MlpService()
        snapshot = service.snapshot()
        hidden = snapshot["hidden"]
        assert "weights" in hidden
        assert "bias" in hidden
        assert "templates" in hidden

    def test_snapshot_output_structure(self) -> None:
        service = MlpService()
        snapshot = service.snapshot()
        output = snapshot["output"]
        assert "weights" in output
        assert "bias" in output

    def test_snapshot_evals_structure(self) -> None:
        service = MlpService()
        snapshot = service.snapshot()
        evals = snapshot["evals"]
        assert len(evals) == 4  # 4 samples in XOR
        for eval_item in evals:
            assert "x" in eval_item
            assert "y" in eval_item
            assert "p_hat" in eval_item
            assert "pred" in eval_item


class TestMlpServiceStep:
    def test_step_returns_snapshot_and_internals(self) -> None:
        service = MlpService()
        snapshot, internals = service.step()
        assert "dataset" in snapshot
        assert hasattr(internals, "x")

    def test_step_increments_idx(self) -> None:
        service = MlpService()
        service.step()
        assert service.idx == 1

    def test_step_wraps_idx(self) -> None:
        service = MlpService()
        for _ in range(4):
            service.step()
        assert service.idx == 0

    def test_step_internals_has_data(self) -> None:
        service = MlpService()
        _, internals = service.step()
        assert hasattr(internals, "x")
        assert hasattr(internals, "y")
        assert hasattr(internals, "loss")

    def test_mlp_learning_xor(self) -> None:
        """Test that MLP can learn XOR."""
        service = MlpService(dataset="xor", hidden_dim=4, lr=0.5, seed=42)

        # Train for several epochs
        for _ in range(200):
            service.step()

        snapshot = service.snapshot()
        evals = snapshot["evals"]

        # Count correct predictions
        correct = sum(1 for e in evals if e["pred"] == e["y"])

        # Should get at least 3/4 correct after training
        assert correct >= 3

