"""Tests for GlassBoxService (Chapters 7 & 8)."""

from unittest.mock import MagicMock, patch

import torch

from backend.services.glass_box_service import GlassBoxService


def _create_mock_tokenizer() -> MagicMock:
    """Create a mock HuggingFace tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    def mock_decode(ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, list):
            return "token"
        return "token token token"

    mock_tokenizer.decode = MagicMock(side_effect=mock_decode)
    return mock_tokenizer


def _create_mock_model() -> MagicMock:
    """Create a mock HuggingFace model with attentions and hidden states."""
    mock_model = MagicMock()

    # Mock forward pass with attentions
    mock_logits = torch.randn(1, 3, 50257)  # batch, seq, vocab
    mock_attentions = tuple(
        torch.softmax(torch.randn(1, 12, 3, 3), dim=-1) for _ in range(12)
    )  # 12 layers, 12 heads, 3x3
    mock_hidden_states = tuple(
        torch.randn(1, 3, 768) for _ in range(13)
    )  # 13 = embedding + 12 layers

    mock_output = MagicMock()
    mock_output.logits = mock_logits
    mock_output.attentions = mock_attentions
    mock_output.hidden_states = mock_hidden_states
    mock_model.return_value = mock_output

    # Mock lm_head
    mock_lm_head = MagicMock()
    mock_lm_head.return_value = torch.randn(50257)
    mock_model.lm_head = mock_lm_head

    mock_model.eval = MagicMock()
    return mock_model


class TestGlassBoxServiceInit:
    def test_init_creates_service(self) -> None:
        service = GlassBoxService()
        assert service is not None

    @patch("backend.services.glass_box_service._shared_hf_model", None)
    def test_model_not_loaded_initially(self) -> None:
        service = GlassBoxService()
        assert service.model_loaded is False

    def test_model_name_default(self) -> None:
        service = GlassBoxService()
        assert service.model_name == "gpt2"


class TestGlassBoxServiceAttention:
    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_get_attention_patterns(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_attention_patterns("Hello world!")
        assert "text" in result
        assert "tokens" in result
        assert "n_layers" in result
        assert "n_heads" in result
        assert "attentions" in result
        assert "explanation" in result

    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_attention_shape(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

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

    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_attention_values_are_probabilities(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_attention_patterns("Test")

        # Check attention values sum to ~1 per row (softmax)
        for layer in result["attentions"]:
            for head in layer:
                for row in head:
                    row_sum = sum(row)
                    assert 0.99 < row_sum < 1.01, f"Row sum should be ~1, got {row_sum}"


class TestGlassBoxServiceLogitLens:
    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_get_logit_lens(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_logit_lens("Hello world!", top_k=5)
        assert "text" in result
        assert "tokens" in result
        assert "n_layers" in result
        assert "top_k" in result
        assert "layers" in result
        assert "explanation" in result

    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_logit_lens_layers(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_logit_lens("Test", top_k=3)
        layers = result["layers"]

        # Should have embedding layer + n_layers
        assert len(layers) == result["n_layers"] + 1

        # Each layer should have predictions
        for layer in layers:
            assert "layer" in layer
            assert "layer_name" in layer
            assert "predictions" in layer
            assert len(layer["predictions"]) == 3  # top_k

    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_logit_lens_prediction_structure(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_logit_lens("Test", top_k=5)
        pred = result["layers"][0]["predictions"][0]
        assert "token" in pred
        assert "token_id" in pred
        assert "probability" in pred

    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_logit_lens_first_layer_is_embedding(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_logit_lens("Test", top_k=3)
        assert result["layers"][0]["layer_name"] == "Embedding"
        assert result["layers"][0]["layer"] == 0

    @patch("backend.services.glass_box_service.get_shared_hf_model")
    def test_logit_lens_probabilities_valid(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = (_create_mock_tokenizer(), _create_mock_model())
        service = GlassBoxService()

        result = service.get_logit_lens("Test", top_k=5)

        for layer in result["layers"]:
            for pred in layer["predictions"]:
                assert 0 <= pred["probability"] <= 1
