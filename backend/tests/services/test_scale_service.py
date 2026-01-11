"""Tests for ScaleService (Chapter 5)."""

import pytest

from backend.services.scale_service import MODELS, ScaleService, format_params


class TestFormatParams:
    def test_format_trillions(self) -> None:
        assert format_params(1_800_000_000_000) == "1.8T"

    def test_format_billions(self) -> None:
        assert format_params(175_000_000_000) == "175.0B"

    def test_format_millions(self) -> None:
        assert format_params(61_000_000) == "61.0M"

    def test_format_thousands(self) -> None:
        assert format_params(60_000) == "60.0K"

    def test_format_small(self) -> None:
        assert format_params(100) == "100"


class TestModelsData:
    def test_has_models(self) -> None:
        assert len(MODELS) >= 4

    def test_model_structure(self) -> None:
        for model in MODELS:
            assert "name" in model
            assert "year" in model
            assert "params" in model
            assert "type" in model
            assert "description" in model

    def test_has_gpt2(self) -> None:
        names = [m["name"] for m in MODELS]
        assert any("GPT-2" in name for name in names)

    def test_has_gpt3(self) -> None:
        names = [m["name"] for m in MODELS]
        assert any("GPT-3" in name for name in names)

    def test_has_gpt4(self) -> None:
        names = [m["name"] for m in MODELS]
        assert any("GPT-4" in name for name in names)

    def test_has_gpt5(self) -> None:
        names = [m["name"] for m in MODELS]
        assert any("GPT-5" in name for name in names)


class TestScaleServiceGetAllModels:
    def test_get_all_models(self) -> None:
        service = ScaleService()
        result = service.get_all_models()
        assert "models" in result
        assert "total_count" in result
        assert result["total_count"] == len(MODELS)

    def test_models_have_formatted_params(self) -> None:
        service = ScaleService()
        result = service.get_all_models()
        for model in result["models"]:
            assert "params_formatted" in model


class TestScaleServiceGetCNNs:
    def test_get_cnns(self) -> None:
        service = ScaleService()
        result = service.get_cnns()
        assert "models" in result
        for model in result["models"]:
            assert model["type"] == "CNN"

    def test_returns_empty_when_no_cnns(self) -> None:
        # Current dataset is GPT-only (transformers)
        service = ScaleService()
        result = service.get_cnns()
        assert result["total_count"] == 0


class TestScaleServiceGetTransformers:
    def test_get_transformers(self) -> None:
        service = ScaleService()
        result = service.get_transformers()
        assert "models" in result
        for model in result["models"]:
            assert model["type"] == "Transformer"

    def test_includes_gpt(self) -> None:
        service = ScaleService()
        result = service.get_transformers()
        names = [m["name"] for m in result["models"]]
        assert any("GPT" in name for name in names)


class TestScaleServiceComparison:
    def test_compare_gpt2_gpt4(self) -> None:
        service = ScaleService()
        result = service.get_comparison("GPT-2 (XL)", "GPT-4 (Base)")
        assert "model1" in result
        assert "model2" in result
        assert "ratio" in result
        assert "explanation" in result
        assert result["ratio"] > 1

    def test_compare_gpt3_gpt5(self) -> None:
        service = ScaleService()
        result = service.get_comparison("GPT-3", "GPT-5 / Orion")
        assert result["ratio"] > 10  # GPT-5 is ~20x larger than GPT-3

    def test_compare_invalid_model(self) -> None:
        service = ScaleService()
        with pytest.raises(ValueError, match="Unknown model"):
            service.get_comparison("NonExistent", "GPT-3")
        with pytest.raises(ValueError, match="Unknown model"):
            service.get_comparison("GPT-3", "NonExistent")

    def test_compare_case_insensitive(self) -> None:
        service = ScaleService()
        result = service.get_comparison("gpt-3", "GPT-4 (Base)")
        assert result["model1"]["name"] == "GPT-3"


class TestScaleServiceGrowth:
    def test_get_growth_data(self) -> None:
        service = ScaleService()
        result = service.get_growth_data()
        assert "data" in result
        assert "insight" in result
        assert len(result["data"]) == len(MODELS)

    def test_growth_data_structure(self) -> None:
        service = ScaleService()
        result = service.get_growth_data()
        for item in result["data"]:
            assert "name" in item
            assert "year" in item
            assert "params" in item
            assert "log_params" in item
            assert "type" in item

