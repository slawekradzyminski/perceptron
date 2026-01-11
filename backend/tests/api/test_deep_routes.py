"""Tests for /deep API routes (Chapter 4)."""

from fastapi.testclient import TestClient

from backend.api_app import app

client = TestClient(app)


class TestDeepState:
    def test_get_state(self) -> None:
        response = client.get("/deep/state")
        assert response.status_code == 200
        data = response.json()
        assert "dataset" in data
        assert "architecture" in data
        assert "metrics" in data

    def test_state_has_samples(self) -> None:
        response = client.get("/deep/state")
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert "sample_count" in data
        assert data["sample_count"] == len(data["samples"])

    def test_state_has_dataset_info(self) -> None:
        client.post("/deep/reset", json={"dataset": "circles"})
        response = client.get("/deep/state")
        assert response.status_code == 200
        data = response.json()
        assert "dataset_info" in data
        assert data["dataset_info"]["name"] == "Concentric Circles"


class TestDeepReset:
    def test_reset_default(self) -> None:
        response = client.post("/deep/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert "dataset" in data
        assert "architecture" in data

    def test_reset_with_hidden_dims(self) -> None:
        response = client.post("/deep/reset", json={"hidden_dims": [8, 8, 8]})
        assert response.status_code == 200
        data = response.json()
        assert data["architecture"]["hidden_dims"] == [8, 8, 8]
        assert data["architecture"]["depth"] == 3

    def test_reset_with_dataset(self) -> None:
        response = client.post("/deep/reset", json={"dataset": "xor"})
        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "xor"

    def test_reset_with_lr(self) -> None:
        response = client.post("/deep/reset", json={"lr": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["lr"] == 0.5

    def test_reset_with_seed(self) -> None:
        response = client.post("/deep/reset", json={"seed": 123})
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data

    def test_reset_invalid_dataset(self) -> None:
        response = client.post("/deep/reset", json={"dataset": "invalid"})
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_reset_invalid_hidden_dims_empty(self) -> None:
        response = client.post("/deep/reset", json={"hidden_dims": []})
        assert response.status_code == 422  # Pydantic validation error

    def test_reset_baarle_dataset(self) -> None:
        response = client.post("/deep/reset", json={"dataset": "baarle"})
        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "baarle"
        assert data["sample_count"] > 0

    def test_reset_spiral_dataset(self) -> None:
        response = client.post("/deep/reset", json={"dataset": "spiral"})
        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "spiral"

    def test_reset_preserves_previous_values(self) -> None:
        # Set up with specific values
        client.post("/deep/reset", json={"dataset": "xor", "hidden_dims": [16], "lr": 0.3})
        # Reset with only dataset change
        response = client.post("/deep/reset", json={"dataset": "circles"})
        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "circles"
        # Previous architecture should be preserved
        assert data["architecture"]["hidden_dims"] == [16]


class TestDeepStep:
    def test_step_single(self) -> None:
        # Reset first
        client.post("/deep/reset", json={"dataset": "circles"})
        response = client.post("/deep/step", json={})
        assert response.status_code == 200
        data = response.json()
        assert "step_info" in data
        assert data["total_steps"] == 1

    def test_step_batch(self) -> None:
        client.post("/deep/reset", json={"dataset": "circles"})
        response = client.post("/deep/step", json={"batch_size": 10})
        assert response.status_code == 200
        data = response.json()
        assert data["total_steps"] == 10

    def test_step_info_contents(self) -> None:
        client.post("/deep/reset", json={"dataset": "circles"})
        response = client.post("/deep/step", json={})
        assert response.status_code == 200
        data = response.json()
        step_info = data["step_info"]
        assert "batch_size" in step_info
        assert "last_loss" in step_info
        assert "last_correct" in step_info
        assert "last_prediction" in step_info
        assert "grad_norm" in step_info

    def test_step_increments_epoch(self) -> None:
        client.post("/deep/reset", json={"dataset": "xor"})  # 4 samples
        # Step through all 4 samples
        response = client.post("/deep/step", json={"batch_size": 4})
        assert response.status_code == 200
        data = response.json()
        assert data["epoch"] == 1

    def test_step_multiple_batches(self) -> None:
        client.post("/deep/reset", json={"dataset": "circles"})
        client.post("/deep/step", json={"batch_size": 5})
        response = client.post("/deep/step", json={"batch_size": 5})
        assert response.status_code == 200
        data = response.json()
        assert data["total_steps"] == 10


class TestDeepEpoch:
    def test_epoch(self) -> None:
        client.post("/deep/reset", json={"dataset": "xor"})
        response = client.post("/deep/epoch")
        assert response.status_code == 200
        data = response.json()
        assert data["epoch"] == 1
        assert data["total_steps"] == 4  # XOR has 4 samples

    def test_multiple_epochs(self) -> None:
        client.post("/deep/reset", json={"dataset": "xor"})
        client.post("/deep/epoch")
        response = client.post("/deep/epoch")
        assert response.status_code == 200
        data = response.json()
        assert data["epoch"] == 2
        assert data["total_steps"] == 8


class TestDeepBoundary:
    def test_get_boundary(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/boundary?resolution=10")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "region_ids" in data
        assert "region_count" in data
        assert len(data["predictions"]) == 10
        assert len(data["predictions"][0]) == 10

    def test_get_boundary_default_resolution(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/boundary")
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 50  # Default resolution

    def test_boundary_predictions_are_binary(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/boundary?resolution=5")
        assert response.status_code == 200
        data = response.json()
        for row in data["predictions"]:
            for pred in row:
                assert pred in (0, 1)

    def test_boundary_has_theoretical_max(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/boundary?resolution=10")
        assert response.status_code == 200
        data = response.json()
        assert "theoretical_max" in data
        assert data["theoretical_max"] >= data["region_count"]


class TestDeepRegions:
    def test_get_regions(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/regions?resolution=10")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "theoretical_max" in data
        assert "efficiency" in data

    def test_get_regions_default_resolution(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/regions")
        assert response.status_code == 200

    def test_regions_efficiency_in_range(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/regions?resolution=10")
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["efficiency"] <= 1


class TestDeepHistory:
    def test_get_history_empty(self) -> None:
        client.post("/deep/reset", json={})
        response = client.get("/deep/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert data["history"] == []

    def test_get_history_after_steps(self) -> None:
        client.post("/deep/reset", json={})
        client.post("/deep/step", json={"batch_size": 3})
        response = client.get("/deep/history")
        assert response.status_code == 200
        data = response.json()
        # History is added once per step() call, not per sample
        assert len(data["history"]) >= 1


class TestDeepComparison:
    def test_add_and_get_comparison(self) -> None:
        # Reset and train
        client.post("/deep/reset", json={"hidden_dims": [8]})
        client.post("/deep/step", json={"batch_size": 10})

        # Add to comparison
        response = client.post("/deep/comparison/add")
        assert response.status_code == 200
        entry = response.json()
        assert "depth" in entry
        assert "actual_regions" in entry

        # Get comparison table
        response = client.get("/deep/comparison")
        assert response.status_code == 200
        data = response.json()
        assert "table" in data
        assert len(data["table"]) >= 1

    def test_clear_comparison(self) -> None:
        client.post("/deep/comparison/clear")
        response = client.get("/deep/comparison")
        assert response.status_code == 200
        data = response.json()
        assert data["table"] == []

    def test_comparison_entry_contents(self) -> None:
        client.post("/deep/reset", json={"hidden_dims": [16, 16]})
        client.post("/deep/comparison/clear")
        response = client.post("/deep/comparison/add")
        assert response.status_code == 200
        entry = response.json()
        assert entry["depth"] == 2
        assert entry["width"] == 16
        assert "hidden_dims" in entry
        assert "param_count" in entry
        assert "actual_regions" in entry
        assert "theoretical_max" in entry
        assert "accuracy" in entry
        assert "loss" in entry

    def test_multiple_comparison_entries(self) -> None:
        client.post("/deep/comparison/clear")
        client.post("/deep/reset", json={"hidden_dims": [4]})
        client.post("/deep/comparison/add")
        client.post("/deep/reset", json={"hidden_dims": [8, 8]})
        client.post("/deep/comparison/add")
        response = client.get("/deep/comparison")
        assert response.status_code == 200
        data = response.json()
        assert len(data["table"]) == 2
        assert data["table"][0]["depth"] == 1
        assert data["table"][1]["depth"] == 2


class TestDeepEndpointIntegration:
    def test_full_workflow(self) -> None:
        # 1. Reset with specific config
        response = client.post("/deep/reset", json={
            "dataset": "xor",
            "hidden_dims": [8, 8],
            "lr": 0.5,
            "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "xor"
        initial_loss = data["metrics"]["loss"]

        # 2. Train for some steps
        for _ in range(10):
            client.post("/deep/step", json={"batch_size": 4})

        # 3. Check state improved
        response = client.get("/deep/state")
        data = response.json()
        assert data["total_steps"] == 40
        # Loss should have decreased
        assert data["metrics"]["loss"] <= initial_loss

        # 4. Get boundary visualization
        response = client.get("/deep/boundary?resolution=10")
        assert response.status_code == 200
        boundary = response.json()
        assert boundary["region_count"] >= 1

        # 5. Add to comparison
        response = client.post("/deep/comparison/add")
        assert response.status_code == 200

        # 6. Get history
        response = client.get("/deep/history")
        assert response.status_code == 200
        history = response.json()
        # History has one entry per step() call, not per sample
        assert len(history["history"]) >= 10

