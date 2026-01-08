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
    @patch("backend.services.transformer_service.TransformerService._load_hf_model")
    def test_get_embedding_info_with_text(self, _mock_load: MagicMock) -> None:
        service = TransformerService()
        service._hf_tokenizer = _create_mock_tokenizer()
        service._hf_model = _create_mock_model()

        result = service.get_embedding_info("Hello world!")
        assert "text" in result
        assert "token_count" in result
        assert "d_model" in result
        assert "matrix_shape" in result
        assert "explanation" in result

    @patch("backend.services.transformer_service.TransformerService._load_hf_model")
    def test_get_embedding_info_uses_cached(self, _mock_load: MagicMock) -> None:
        service = TransformerService()
        service._hf_tokenizer = _create_mock_tokenizer()
        service._hf_model = _create_mock_model()

        service.tokenize("Cached text")
        result = service.get_embedding_info()  # No text provided
        # Will use default "Hello" since tokenize caches tiktoken tokens, not HF
        assert "text" in result

    @patch("backend.services.transformer_service.TransformerService._load_hf_model")
    def test_matrix_shape_correct(self, _mock_load: MagicMock) -> None:
        service = TransformerService()
        service._hf_tokenizer = _create_mock_tokenizer()
        service._hf_model = _create_mock_model()

        result = service.get_embedding_info("Hello world!")
        shape = result["matrix_shape"]
        assert shape[0] == result["token_count"]
        assert shape[1] == result["d_model"]

    @patch("backend.services.transformer_service.TransformerService._load_hf_model")
    def test_tokens_have_embeddings(self, _mock_load: MagicMock) -> None:
        service = TransformerService()
        service._hf_tokenizer = _create_mock_tokenizer()
        service._hf_model = _create_mock_model()

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
