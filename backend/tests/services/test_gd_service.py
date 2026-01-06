"""Tests for GdService (Gradient Descent exercises)."""

import math

import pytest

from backend.services.gd_service import GdService, _read_config


class TestGdConfig:
    def test_read_config_defaults(self) -> None:
        config = _read_config()
        assert config.token_source == "static"
        assert config.ollama_base_url == "http://127.0.0.1:11434"
        assert config.ollama_model == "llama3.2:1b"
        assert config.ollama_timeout_s == 30.0
        assert config.ollama_temperature == 0.0
        assert config.ollama_top_logprobs == 20


class TestGdServiceInit:
    def test_init_creates_examples(self) -> None:
        service = GdService()
        assert hasattr(service, "_examples")
        assert len(service._examples) >= 3

    def test_init_no_client(self) -> None:
        service = GdService()
        assert service._client is None


class TestGdServiceExamples:
    def test_examples_have_required_fields(self) -> None:
        service = GdService()
        for example in service._examples:
            assert "id" in example
            assert "title" in example
            assert "description" in example
            assert "rows" in example

    def test_examples_rows_have_required_fields(self) -> None:
        service = GdService()
        for example in service._examples:
            for row in example["rows"]:
                assert "context" in row
                assert "correct_token" in row
                assert "p_correct" in row
                assert "top_token" in row
                assert "top_prob" in row


class TestGdServiceLossCurves:
    def test_loss_curves_default(self) -> None:
        service = GdService()
        result = service.loss_curves()
        assert "points" in result
        assert len(result["points"]) == 50

    def test_loss_curves_custom_points(self) -> None:
        service = GdService()
        result = service.loss_curves(points=10)
        assert len(result["points"]) == 10

    def test_loss_curves_min_points(self) -> None:
        service = GdService()
        result = service.loss_curves(points=2)
        assert len(result["points"]) == 2

    def test_loss_curves_invalid_points(self) -> None:
        service = GdService()
        with pytest.raises(ValueError, match="points must be >= 2"):
            service.loss_curves(points=1)

    def test_loss_curves_structure(self) -> None:
        service = GdService()
        result = service.loss_curves(points=5)
        for point in result["points"]:
            assert "p" in point
            assert "l1" in point
            assert "ce" in point

    def test_loss_curves_l1_formula(self) -> None:
        service = GdService()
        result = service.loss_curves(points=10, eps=0.01)
        for point in result["points"]:
            expected_l1 = 1.0 - point["p"]
            assert abs(point["l1"] - expected_l1) < 1e-10

    def test_loss_curves_ce_formula(self) -> None:
        service = GdService()
        result = service.loss_curves(points=10, eps=0.01)
        for point in result["points"]:
            expected_ce = -math.log(point["p"])
            assert abs(point["ce"] - expected_ce) < 1e-10

    def test_loss_curves_p_range(self) -> None:
        service = GdService()
        eps = 0.001
        result = service.loss_curves(points=10, eps=eps)
        # First point should be near eps
        assert abs(result["points"][0]["p"] - eps) < 1e-10
        # Last point should be near 1.0
        assert abs(result["points"][-1]["p"] - 1.0) < 1e-10


class TestGdServiceTokenLossExamples:
    def test_token_loss_examples_static(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        assert result["source"] == "static"
        assert "examples" in result
        assert len(result["examples"]) >= 1

    def test_token_loss_examples_structure(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        for example in result["examples"]:
            assert "id" in example
            assert "title" in example
            assert "description" in example
            assert "average" in example
            assert "rows" in example

    def test_token_loss_examples_average(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        for example in result["examples"]:
            average = example["average"]
            assert "l1" in average
            assert "ce" in average
            assert average["l1"] >= 0
            assert average["ce"] >= 0

    def test_token_loss_examples_rows_structure(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        for example in result["examples"]:
            for row in example["rows"]:
                assert "context" in row
                assert "correct_token" in row
                assert "p_correct" in row
                assert "top_token" in row
                assert "top_prob" in row
                assert "l1_loss" in row
                assert "ce_loss" in row

    def test_token_loss_examples_filter_by_id(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static", example_id="france-paris")
        assert len(result["examples"]) == 1
        assert result["examples"][0]["id"] == "france-paris"

    def test_token_loss_examples_filter_nonexistent(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static", example_id="nonexistent")
        assert len(result["examples"]) == 0

    def test_token_loss_examples_l1_formula(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        for example in result["examples"]:
            for row in example["rows"]:
                expected_l1 = 1.0 - row["p_correct"]
                assert abs(row["l1_loss"] - expected_l1) < 1e-10

    def test_token_loss_examples_ce_formula(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        for example in result["examples"]:
            for row in example["rows"]:
                expected_ce = -math.log(max(row["p_correct"], 1e-12))
                assert abs(row["ce_loss"] - expected_ce) < 1e-10


class TestGdServiceExampleIds:
    def test_france_paris_exists(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static", example_id="france-paris")
        assert len(result["examples"]) == 1
        assert result["examples"][0]["title"] == "The capital of France is Paris"

    def test_apple_day_exists(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static", example_id="apple-day")
        assert len(result["examples"]) == 1
        assert result["examples"][0]["title"] == "An apple a day keeps the doctor away"

    def test_all_examples_have_rows(self) -> None:
        service = GdService()
        result = service.token_loss_examples(source_override="static")
        for example in result["examples"]:
            assert len(example["rows"]) >= 1

