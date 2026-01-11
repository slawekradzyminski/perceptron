"""Tests for KVCacheService (Chapter 8)."""

from backend.services.kv_cache_service import KVCacheService


class TestKVCacheServiceDefaults:
    def test_get_comparison_returns_required_fields(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        assert "config" in result
        assert "architectures" in result
        assert "mha_to_mla_savings" in result
        assert "explanation" in result

    def test_architectures_present(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        arch_names = [a["name"] for a in result["architectures"]]
        assert "MHA" in arch_names
        assert "MQA" in arch_names
        assert "GQA" in arch_names
        assert "MLA" in arch_names


class TestKVCacheServiceArchitectures:
    def test_mha_is_baseline(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        mha = next(a for a in result["architectures"] if a["name"] == "MHA")
        assert mha["ratio_to_mha"] == 1.0

    def test_mla_smaller_than_mha(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        mha = next(a for a in result["architectures"] if a["name"] == "MHA")
        mla = next(a for a in result["architectures"] if a["name"] == "MLA")

        assert mla["memory_bytes"] < mha["memory_bytes"]
        assert mla["ratio_to_mha"] < 1.0

    def test_mqa_smaller_than_mha(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        mha = next(a for a in result["architectures"] if a["name"] == "MHA")
        mqa = next(a for a in result["architectures"] if a["name"] == "MQA")

        assert mqa["memory_bytes"] < mha["memory_bytes"]

    def test_gqa_between_mha_and_mqa(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        mha = next(a for a in result["architectures"] if a["name"] == "MHA")
        mqa = next(a for a in result["architectures"] if a["name"] == "MQA")
        gqa = next(a for a in result["architectures"] if a["name"] == "GQA")

        assert mqa["memory_bytes"] < gqa["memory_bytes"] < mha["memory_bytes"]


class TestKVCacheServiceCustomParams:
    def test_custom_context_length(self) -> None:
        service = KVCacheService()
        result = service.get_comparison(context_length=8192)

        assert result["config"]["context_length"] == 8192

    def test_custom_layers(self) -> None:
        service = KVCacheService()
        result = service.get_comparison(n_layers=48)

        assert result["config"]["n_layers"] == 48

    def test_custom_d_model(self) -> None:
        service = KVCacheService()
        result = service.get_comparison(d_model=8192)

        assert result["config"]["d_model"] == 8192

    def test_custom_heads(self) -> None:
        service = KVCacheService()
        result = service.get_comparison(n_heads=64)

        assert result["config"]["n_heads"] == 64
        assert result["config"]["d_head"] == 64  # 4096 / 64

    def test_larger_context_means_more_memory(self) -> None:
        service = KVCacheService()
        small = service.get_comparison(context_length=1024)
        large = service.get_comparison(context_length=4096)

        small_mha = next(a for a in small["architectures"] if a["name"] == "MHA")
        large_mha = next(a for a in large["architectures"] if a["name"] == "MHA")

        assert large_mha["memory_bytes"] > small_mha["memory_bytes"]


class TestKVCacheServiceFormatting:
    def test_memory_formatted_human_readable(self) -> None:
        service = KVCacheService()
        result = service.get_comparison(context_length=1024)

        for arch in result["architectures"]:
            assert any(
                unit in arch["memory_formatted"]
                for unit in ["B", "KB", "MB", "GB"]
            )

    def test_savings_ratio_positive(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        assert result["mha_to_mla_savings"] > 1.0

    def test_explanation_present(self) -> None:
        service = KVCacheService()
        result = service.get_comparison()

        assert len(result["explanation"]) > 50
        assert "MHA" in result["explanation"]
        assert "MLA" in result["explanation"]
