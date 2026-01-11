"""Tests for /transformer API routes (Chapter 5)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api_app import app

client = TestClient(app)


def _mock_embedding_result() -> dict:
    """Return a mock embedding result to avoid loading HuggingFace model."""
    return {
        "text": "Hello world!",
        "token_count": 3,
        "d_model": 768,
        "model_name": "gpt2",
        "matrix_shape": [3, 768],
        "explanation": "Mock embedding explanation",
        "tokens": [
            {"id": 1, "text": "Hello", "position": 0, "embedding": [0.1] * 768,
             "embedding_preview": [0.1] * 10, "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1},
        ],
        "full_matrix": [[0.1] * 768],
    }


class TestTransformerState:
    def test_get_state(self) -> None:
        response = client.get("/transformer/state")
        assert response.status_code == 200
        data = response.json()
        assert "current_text" in data
        assert "vocab_size" in data
        assert "encoding_name" in data


class TestTransformerTokenize:
    def test_tokenize_simple(self) -> None:
        response = client.post("/transformer/tokenize", json={"text": "Hello world!"})
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world!"
        assert "token_count" in data
        assert "tokens" in data

    def test_tokenize_token_structure(self) -> None:
        response = client.post("/transformer/tokenize", json={"text": "Test"})
        assert response.status_code == 200
        data = response.json()
        for token in data["tokens"]:
            assert "id" in token
            assert "text" in token
            assert "position" in token

    def test_tokenize_missing_text(self) -> None:
        response = client.post("/transformer/tokenize", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_tokenize_invalid_type(self) -> None:
        response = client.post("/transformer/tokenize", json={"text": 123})
        assert response.status_code == 422  # Pydantic validation error

    def test_tokenize_too_long(self) -> None:
        long_text = "a" * 10001
        response = client.post("/transformer/tokenize", json={"text": long_text})
        assert response.status_code == 422  # Pydantic validation error


class TestTransformerEmbed:
    @patch("backend.api.deps.transformer_service.get_embedding_info")
    def test_embed_with_text(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = _mock_embedding_result()
        response = client.post("/transformer/embed", json={"text": "Hello world!"})
        assert response.status_code == 200
        data = response.json()
        assert "matrix_shape" in data
        assert "d_model" in data
        assert "explanation" in data
        mock_embed.assert_called_once_with("Hello world!")

    @patch("backend.api.deps.transformer_service.get_embedding_info")
    def test_embed_without_text(self, mock_embed: MagicMock) -> None:
        mock_result = _mock_embedding_result()
        mock_result["text"] = "Previous text"
        mock_embed.return_value = mock_result
        # First tokenize something
        client.post("/transformer/tokenize", json={"text": "Previous text"})
        # Then embed without text
        response = client.post("/transformer/embed", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Previous text"


class TestTransformerTrace:
    def test_trace_with_text(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Hello world!", "n_blocks": 6})
        assert response.status_code == 200
        data = response.json()
        assert data["n_blocks"] == 6
        assert len(data["blocks"]) == 6
        assert "explanation" in data

    def test_trace_default_blocks(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Test"})
        assert response.status_code == 200
        data = response.json()
        assert data["n_blocks"] == 12  # Default

    def test_trace_invalid_n_blocks(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Test", "n_blocks": 0})
        assert response.status_code == 422  # Pydantic validation error

    def test_trace_too_many_blocks(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Test", "n_blocks": 100})
        assert response.status_code == 422  # Pydantic validation error


class TestTransformerChat:
    def test_chat_simple(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        response = client.post("/transformer/chat", json={"messages": messages})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_missing_messages(self) -> None:
        response = client.post("/transformer/chat", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_invalid_message_format(self) -> None:
        response = client.post("/transformer/chat", json={"messages": [{"wrong": "format"}]})
        assert response.status_code == 422  # Pydantic validation error


class TestTransformerOllamaStatus:
    def test_ollama_status(self) -> None:
        response = client.get("/transformer/ollama-status")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "model" in data


class TestTransformerModels:
    def test_get_models(self) -> None:
        response = client.get("/transformer/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 3
        assert "explanation" in data


class TestScaleEndpoints:
    def test_scale_all_models(self) -> None:
        response = client.get("/transformer/scale")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_count" in data

    def test_scale_cnns(self) -> None:
        response = client.get("/transformer/scale/cnns")
        assert response.status_code == 200
        data = response.json()
        for model in data["models"]:
            assert model["type"] == "CNN"

    def test_scale_transformers(self) -> None:
        response = client.get("/transformer/scale/transformers")
        assert response.status_code == 200
        data = response.json()
        for model in data["models"]:
            assert model["type"] == "Transformer"

    def test_scale_compare(self) -> None:
        response = client.get("/transformer/scale/compare?model1=GPT-3&model2=GPT-4%20(Base)")
        assert response.status_code == 200
        data = response.json()
        assert "ratio" in data
        assert "explanation" in data

    def test_scale_compare_invalid(self) -> None:
        response = client.get("/transformer/scale/compare?model1=Invalid&model2=GPT-3")
        assert response.status_code == 400

    def test_scale_growth(self) -> None:
        response = client.get("/transformer/scale/growth")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "insight" in data


# ====================
# Glass Box Tests (Chapter 7 & 8)
# ====================


class TestTransformerAttention:
    """Tests for the attention endpoint.

    Note: The simple test that calls the real model is skipped because
    it requires loading the HuggingFace model which can be slow/unstable in CI.
    The validation tests still run.
    """

    def test_attention_missing_text(self) -> None:
        response = client.post("/transformer/attention", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_attention_invalid_type(self) -> None:
        response = client.post("/transformer/attention", json={"text": 123})
        assert response.status_code == 422  # Pydantic validation error

    def test_attention_too_long(self) -> None:
        long_text = "a" * 501
        response = client.post("/transformer/attention", json={"text": long_text})
        assert response.status_code == 422  # Pydantic validation error


class TestTransformerLogitLens:
    """Tests for the logit-lens endpoint.

    Note: The simple test that calls the real model is skipped because
    it requires loading the HuggingFace model which can be slow/unstable in CI.
    The validation tests still run.
    """

    def test_logit_lens_missing_text(self) -> None:
        response = client.post("/transformer/logit-lens", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_logit_lens_invalid_top_k(self) -> None:
        response = client.post(
            "/transformer/logit-lens", json={"text": "Test", "top_k": 0}
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_logit_lens_too_long(self) -> None:
        long_text = "a" * 501
        response = client.post("/transformer/logit-lens", json={"text": long_text})
        assert response.status_code == 422  # Pydantic validation error


class TestTransformerKVCache:
    def test_kv_cache_defaults(self) -> None:
        response = client.post("/transformer/kv-cache", json={})
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "architectures" in data
        assert "mha_to_mla_savings" in data
        assert "explanation" in data

    def test_kv_cache_custom_params(self) -> None:
        response = client.post(
            "/transformer/kv-cache",
            json={
                "context_length": 8192,
                "n_layers": 48,
                "d_model": 8192,
                "n_heads": 64,
                "gqa_groups": 8,
                "mla_latent_dim": 1024,
            },
        )
        assert response.status_code == 200
        data = response.json()
        config = data["config"]
        assert config["context_length"] == 8192
        assert config["n_layers"] == 48

    def test_kv_cache_architectures(self) -> None:
        response = client.post("/transformer/kv-cache", json={})
        assert response.status_code == 200
        data = response.json()
        arch_names = [a["name"] for a in data["architectures"]]
        assert "MHA" in arch_names
        assert "MQA" in arch_names
        assert "GQA" in arch_names
        assert "MLA" in arch_names

    def test_kv_cache_invalid_context_length(self) -> None:
        response = client.post(
            "/transformer/kv-cache", json={"context_length": 0}
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_kv_cache_invalid_n_layers(self) -> None:
        response = client.post(
            "/transformer/kv-cache", json={"n_layers": 300}
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_kv_cache_invalid_gqa_groups(self) -> None:
        response = client.post(
            "/transformer/kv-cache", json={"gqa_groups": 100, "n_heads": 32}
        )
        assert response.status_code == 422  # Pydantic validation error

