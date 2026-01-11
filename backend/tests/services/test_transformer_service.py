"""Tests for TransformerService (Chapter 5)."""

from unittest.mock import MagicMock, patch

import torch

from backend.services.transformer_service import GPT_MODELS, TransformerService


class TestGPTModels:
    def test_has_models(self) -> None:
        assert len(GPT_MODELS) >= 3

    def test_model_structure(self) -> None:
        for _model_id, info in GPT_MODELS.items():
            assert "name" in info
            assert "year" in info
            assert "params" in info
            assert "layers" in info
            assert "d_model" in info


class TestTransformerServiceInit:
    def test_init_creates_service(self) -> None:
        service = TransformerService()
        assert service is not None

    def test_initial_state(self) -> None:
        service = TransformerService()
        state = service.state()
        assert state["current_text"] == ""
        assert state["current_token_count"] == 0
        assert "vocab_size" in state


class TestTransformerServiceTokenize:
    def test_tokenize_simple(self) -> None:
        service = TransformerService()
        result = service.tokenize("Hello world!")
        assert result["text"] == "Hello world!"
        assert result["token_count"] >= 1
        assert len(result["tokens"]) == result["token_count"]

    def test_tokenize_token_structure(self) -> None:
        service = TransformerService()
        result = service.tokenize("Hello")
        for token in result["tokens"]:
            assert "id" in token
            assert "text" in token
            assert "position" in token

    def test_tokenize_updates_state(self) -> None:
        service = TransformerService()
        service.tokenize("Test text")
        state = service.state()
        assert state["current_text"] == "Test text"
        assert state["current_token_count"] >= 1

    def test_tokenize_empty(self) -> None:
        service = TransformerService()
        result = service.tokenize("")
        assert result["token_count"] == 0
        assert result["tokens"] == []

    def test_tokenize_long_text(self) -> None:
        service = TransformerService()
        long_text = "This is a longer sentence with multiple words and punctuation."
        result = service.tokenize(long_text)
        assert result["token_count"] > 5


def _create_mock_tokenizer() -> MagicMock:
    """Create a mock HuggingFace tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    def mock_decode(ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, list):
            return "token"
        # For tensor inputs
        return "token token token"

    mock_tokenizer.decode = MagicMock(side_effect=mock_decode)
    mock_tokenizer.eos_token_id = 0
    return mock_tokenizer


def _create_mock_model() -> MagicMock:
    """Create a mock HuggingFace model."""
    mock_model = MagicMock()

    # Mock embedding layer
    mock_wte = MagicMock()
    mock_wte.return_value = torch.randn(1, 3, 768)
    mock_model.transformer = MagicMock()
    mock_model.transformer.wte = mock_wte

    # Mock forward pass for generation
    mock_logits = torch.randn(1, 3, 50257)  # batch, seq, vocab
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    mock_model.return_value = mock_output

    mock_model.eval = MagicMock()
    return mock_model


class TestTransformerServiceEmbedding:
    @patch("backend.services.transformer_service.get_shared_hf_model")
    def test_get_embedding_info_with_text(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = TransformerService()

        result = service.get_embedding_info("Hello world!")
        assert "text" in result
        assert "token_count" in result
        assert "d_model" in result
        assert "matrix_shape" in result
        assert "explanation" in result

    @patch("backend.services.transformer_service.get_shared_hf_model")
    def test_get_embedding_info_uses_cached(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = TransformerService()

        service.tokenize("Cached text")
        result = service.get_embedding_info()  # No text provided
        # Will use default "Hello" since tokenize caches tiktoken tokens, not HF
        assert "text" in result

    @patch("backend.services.transformer_service.get_shared_hf_model")
    def test_matrix_shape_correct(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = TransformerService()

        result = service.get_embedding_info("Hello world!")
        shape = result["matrix_shape"]
        assert shape[0] == result["token_count"]
        assert shape[1] == result["d_model"]

    @patch("backend.services.transformer_service.get_shared_hf_model")
    def test_tokens_have_embeddings(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = TransformerService()

        result = service.get_embedding_info("Hello")
        for token in result["tokens"]:
            assert "embedding" in token
            assert "embedding_preview" in token
            assert "min" in token
            assert "max" in token
            assert "mean" in token
            assert "std" in token


class TestTransformerServiceBlockTrace:
    def test_get_block_trace(self) -> None:
        service = TransformerService()
        result = service.get_block_trace("Hello world!", n_blocks=12)
        assert result["n_blocks"] == 12
        assert len(result["blocks"]) == 12
        assert "explanation" in result

    def test_block_structure(self) -> None:
        service = TransformerService()
        result = service.get_block_trace("Hello", n_blocks=3)
        for block in result["blocks"]:
            assert "block" in block
            assert "input_shape" in block
            assert "output_shape" in block
            assert "operations" in block

    def test_shapes_preserved(self) -> None:
        service = TransformerService()
        result = service.get_block_trace("Test", n_blocks=5)
        # All blocks should have same shape
        first_shape = result["blocks"][0]["input_shape"]
        for block in result["blocks"]:
            assert block["input_shape"] == first_shape
            assert block["output_shape"] == first_shape

    def test_operations_have_details(self) -> None:
        service = TransformerService()
        result = service.get_block_trace("Hello", n_blocks=1)
        for op in result["blocks"][0]["operations"]:
            assert "name" in op
            assert "formula" in op
            assert "details" in op


class TestTransformerServiceChat:
    @patch("backend.services.transformer_service.TransformerService._ollama_available")
    @patch("backend.services.transformer_service.TransformerService._ollama_chat")
    def test_chat(self, mock_chat: MagicMock, mock_available: MagicMock) -> None:
        mock_available.return_value = True
        mock_chat.return_value = {
            "response": "Hello! How can I help you today?",
            "done": True,
            "eval_count": 8,
            "total_duration": 150000000,  # 150ms in nanoseconds
        }

        service = TransformerService()
        messages = [{"role": "user", "content": "Hello"}]
        result = service.chat(messages)
        assert result["response"] == "Hello! How can I help you today?"
        assert "generation_time_ms" in result

    @patch("backend.services.transformer_service.TransformerService._ollama_available")
    @patch("backend.services.transformer_service.TransformerService._ollama_chat")
    def test_chat_result_structure(self, mock_chat: MagicMock, mock_available: MagicMock) -> None:
        mock_available.return_value = True
        mock_chat.return_value = {
            "response": "Test response",
            "done": True,
            "eval_count": 3,
            "total_duration": 100000000,
        }

        service = TransformerService()
        messages = [{"role": "user", "content": "Test"}]
        result = service.chat(messages)
        assert "model_name" in result
        assert "response" in result
        assert "generation_time_ms" in result

    @patch("backend.services.transformer_service.TransformerService._ollama_available")
    def test_chat_ollama_unavailable(self, mock_available: MagicMock) -> None:
        mock_available.return_value = False

        service = TransformerService()
        messages = [{"role": "user", "content": "Hello"}]
        result = service.chat(messages)
        assert result["response"] == ""
        assert "error" in result
        assert result["error"] == "Ollama not available"


class TestTransformerServiceOllamaStatus:
    @patch("backend.services.transformer_service.urllib.request.urlopen")
    def test_ollama_status_online(self, mock_urlopen: MagicMock) -> None:
        import json
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "models": [{"name": "llama3.2:1b"}]
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        service = TransformerService()
        result = service.ollama_status()
        assert result["ok"] is True
        assert "model" in result
        assert "models" in result

    @patch("backend.services.transformer_service.urllib.request.urlopen")
    def test_ollama_status_offline(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("Connection refused")

        service = TransformerService()
        result = service.ollama_status()
        assert result["ok"] is False
        assert "error" in result


class TestTransformerServiceModelComparison:
    def test_get_model_comparison(self) -> None:
        service = TransformerService()
        result = service.get_model_comparison()
        assert "models" in result
        assert len(result["models"]) >= 3
        assert "explanation" in result

    def test_model_data_structure(self) -> None:
        service = TransformerService()
        result = service.get_model_comparison()
        for model in result["models"]:
            assert "id" in model
            assert "name" in model
            assert "params" in model
            assert "year" in model


# ====================
# Glass Box Tests (Chapter 7 & 8)
# These tests verify delegation to GlassBoxService
# ====================


def _mock_attention_result() -> dict:
    """Return a mock attention result for testing delegation."""
    return {
        "text": "Hello world!",
        "tokens": ["Hello", " world", "!"],
        "n_layers": 12,
        "n_heads": 12,
        "seq_len": 3,
        "model_name": "gpt2",
        "attentions": [[[
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.3, 0.4],
        ] for _ in range(12)] for _ in range(12)],
        "explanation": "Test explanation",
    }


def _mock_logit_lens_result() -> dict:
    """Return a mock logit lens result for testing delegation."""
    return {
        "text": "Hello world!",
        "tokens": ["Hello", " world", "!"],
        "n_layers": 12,
        "top_k": 5,
        "model_name": "gpt2",
        "layers": [
            {
                "layer": i,
                "layer_name": "Embedding" if i == 0 else f"Layer {i}",
                "predictions": [
                    {"token": f"tok{j}", "token_id": j, "probability": 0.2}
                    for j in range(5)
                ],
            }
            for i in range(13)  # embedding + 12 layers
        ],
        "explanation": "Test explanation",
    }


class TestTransformerServiceAttentionDelegation:
    def test_get_attention_patterns_delegates(self) -> None:
        service = TransformerService()
        service._glass_box_service.get_attention_patterns = MagicMock(
            return_value=_mock_attention_result()
        )

        result = service.get_attention_patterns("Hello world!")
        assert "text" in result
        assert "tokens" in result
        assert "n_layers" in result
        assert "n_heads" in result
        assert "attentions" in result
        assert "explanation" in result
        service._glass_box_service.get_attention_patterns.assert_called_once_with("Hello world!")

    def test_attention_shape_from_delegate(self) -> None:
        service = TransformerService()
        service._glass_box_service.get_attention_patterns = MagicMock(
            return_value=_mock_attention_result()
        )

        result = service.get_attention_patterns("Test")
        n_layers = result["n_layers"]
        n_heads = result["n_heads"]
        seq_len = result["seq_len"]

        # Check attention tensor shape
        attentions = result["attentions"]
        assert len(attentions) == n_layers
        assert len(attentions[0]) == n_heads
        assert len(attentions[0][0]) == seq_len
        assert len(attentions[0][0][0]) == seq_len


class TestTransformerServiceLogitLensDelegation:
    def test_get_logit_lens_delegates(self) -> None:
        service = TransformerService()
        service._glass_box_service.get_logit_lens = MagicMock(
            return_value=_mock_logit_lens_result()
        )

        result = service.get_logit_lens("Hello world!", top_k=5)
        assert "text" in result
        assert "tokens" in result
        assert "n_layers" in result
        assert "top_k" in result
        assert "layers" in result
        assert "explanation" in result
        service._glass_box_service.get_logit_lens.assert_called_once_with("Hello world!", 5)

    def test_logit_lens_layers_from_delegate(self) -> None:
        service = TransformerService()
        service._glass_box_service.get_logit_lens = MagicMock(
            return_value=_mock_logit_lens_result()
        )

        result = service.get_logit_lens("Test", top_k=3)
        layers = result["layers"]

        # Should have embedding layer + n_layers
        assert len(layers) == result["n_layers"] + 1

        # Each layer should have predictions
        for layer in layers:
            assert "layer" in layer
            assert "layer_name" in layer
            assert "predictions" in layer

    def test_logit_lens_prediction_structure_from_delegate(self) -> None:
        service = TransformerService()
        service._glass_box_service.get_logit_lens = MagicMock(
            return_value=_mock_logit_lens_result()
        )

        result = service.get_logit_lens("Test", top_k=5)
        pred = result["layers"][0]["predictions"][0]
        assert "token" in pred
        assert "token_id" in pred
        assert "probability" in pred


class TestTransformerServiceKVCache:
    def test_get_kv_cache_comparison_defaults(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison()

        assert "config" in result
        assert "architectures" in result
        assert "mha_to_mla_savings" in result
        assert "explanation" in result

    def test_kv_cache_architectures(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison()

        arch_names = [a["name"] for a in result["architectures"]]
        assert "MHA" in arch_names
        assert "MQA" in arch_names
        assert "GQA" in arch_names
        assert "MLA" in arch_names

    def test_kv_cache_architecture_structure(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison()

        for arch in result["architectures"]:
            assert "name" in arch
            assert "full_name" in arch
            assert "description" in arch
            assert "memory_bytes" in arch
            assert "memory_formatted" in arch
            assert "ratio_to_mha" in arch

    def test_kv_cache_mha_baseline(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison()

        mha = next(a for a in result["architectures"] if a["name"] == "MHA")
        assert mha["ratio_to_mha"] == 1.0

    def test_kv_cache_mla_smaller_than_mha(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison()

        mha = next(a for a in result["architectures"] if a["name"] == "MHA")
        mla = next(a for a in result["architectures"] if a["name"] == "MLA")

        assert mla["memory_bytes"] < mha["memory_bytes"]
        assert result["mha_to_mla_savings"] > 1

    def test_kv_cache_custom_params(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison(
            context_length=8192,
            n_layers=48,
            d_model=8192,
            n_heads=64,
            gqa_groups=8,
            mla_latent_dim=1024,
        )

        config = result["config"]
        assert config["context_length"] == 8192
        assert config["n_layers"] == 48
        assert config["d_model"] == 8192
        assert config["n_heads"] == 64

    def test_kv_cache_memory_format(self) -> None:
        service = TransformerService()
        result = service.get_kv_cache_comparison(context_length=1024)

        for arch in result["architectures"]:
            # Memory should be formatted as human-readable
            assert any(
                unit in arch["memory_formatted"]
                for unit in ["B", "KB", "MB", "GB"]
            )
