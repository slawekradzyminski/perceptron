"""Glass Box service for Chapters 7 & 8 (attention patterns, logit lens)."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Shared HuggingFace model cache to avoid loading multiple copies
_shared_hf_tokenizer = None
_shared_hf_model = None


def reset_shared_hf_model():
    """Reset the shared model cache to force a reload."""
    global _shared_hf_tokenizer, _shared_hf_model
    _shared_hf_tokenizer = None
    _shared_hf_model = None
    logger.info("Shared HuggingFace model cache cleared")


def get_shared_hf_model():
    """Get or load the shared HuggingFace model and tokenizer."""
    global _shared_hf_tokenizer, _shared_hf_model
    if _shared_hf_tokenizer is None or _shared_hf_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "gpt2"
            logger.info(f"Loading shared HuggingFace model: {model_name}")
            _shared_hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Use 'eager' attention implementation to support output_attentions
            # (SDPA doesn't support returning attention weights)
            _shared_hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="eager"
            )

            # Fix for PyTorch 2.7+ on Apple Silicon: untie the lm_head weights
            # to avoid SIGBUS crash during matrix multiplication with tied weights
            has_lm_head = hasattr(_shared_hf_model, "lm_head")
            has_transformer = hasattr(_shared_hf_model, "transformer")
            if has_lm_head and has_transformer:
                weights_are_tied = (
                    _shared_hf_model.lm_head.weight
                    is _shared_hf_model.transformer.wte.weight
                )
                if weights_are_tied:
                    logger.info("Untying lm_head weights for stability...")
                    vocab_size = _shared_hf_model.lm_head.weight.shape[0]
                    hidden_size = _shared_hf_model.lm_head.weight.shape[1]
                    new_lm_head = torch.nn.Linear(
                        hidden_size, vocab_size, bias=False
                    )
                    with torch.no_grad():
                        new_lm_head.weight.copy_(
                            _shared_hf_model.transformer.wte.weight
                        )
                    _shared_hf_model.lm_head = new_lm_head

            _shared_hf_model.eval()
            logger.info(f"Successfully loaded shared {model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    return _shared_hf_tokenizer, _shared_hf_model


class GlassBoxService:
    """Service for mechanistic interpretability: attention and logit lens."""

    def __init__(self) -> None:
        self._model_name = "gpt2"

    def _get_model(self):
        """Get the shared model and tokenizer."""
        return get_shared_hf_model()

    def get_attention_patterns(self, text: str) -> dict[str, Any]:
        """Get attention patterns from all layers and heads.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with attention matrices and metadata
        """
        tokenizer, model = self._get_model()

        # Tokenize with HF tokenizer
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=True,
                output_hidden_states=False,
            )

        # attentions is a tuple of (n_layers) tensors, each (batch, heads, seq, seq)
        attentions = outputs.attentions

        if attentions is None:
            raise ValueError(
                "Model did not return attention weights. "
                "Please restart the backend to reload the model."
            )

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]
        seq_len = attentions[0].shape[2]

        # Decode tokens for labels
        token_labels = [
            tokenizer.decode([tid.item()]) for tid in input_ids[0]
        ]

        # Convert attention tensors to lists (for JSON serialization)
        attention_data: list[list[list[list[float]]]] = []
        for layer_idx in range(n_layers):
            layer_attention = attentions[layer_idx][0]  # Remove batch dim
            layer_data: list[list[list[float]]] = []
            for head_idx in range(n_heads):
                head_attention = layer_attention[head_idx].numpy()
                # Round to 4 decimal places for reasonable JSON size
                head_data = [
                    [round(float(v), 4) for v in row]
                    for row in head_attention
                ]
                layer_data.append(head_data)
            attention_data.append(layer_data)

        return {
            "text": text,
            "tokens": token_labels,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "model_name": self._model_name,
            "attentions": attention_data,
            "explanation": (
                f"Attention patterns from {self._model_name}. "
                f"Each of the {n_layers} layers has {n_heads} attention heads. "
                "Each head computes a {seq_len}×{seq_len} attention matrix showing "
                "how much each token attends to every other token."
            ),
        }

    def get_logit_lens(self, text: str, top_k: int = 5) -> dict[str, Any]:
        """Apply the Logit Lens to see predictions at each layer.

        The Logit Lens applies the final unembedding matrix to hidden states
        at intermediate layers, showing what the model "believes" at each depth.

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to return per layer

        Returns:
            Dictionary with layer-wise predictions
        """
        tokenizer, model = self._get_model()

        # Tokenize with HF tokenizer
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=False,
                output_hidden_states=True,
            )

        # hidden_states is tuple of (n_layers + 1) tensors, each (batch, seq, d_model)
        # First element is the embedding layer output
        hidden_states = outputs.hidden_states

        n_layers = len(hidden_states) - 1  # Exclude embedding layer

        # Get the lm_head (unembedding matrix)
        lm_head = model.lm_head

        # Decode input tokens for reference
        token_labels = [
            tokenizer.decode([tid.item()]) for tid in input_ids[0]
        ]

        # For each layer, apply lm_head to the final token position
        layer_predictions = []
        for layer_idx in range(n_layers + 1):
            hidden = hidden_states[layer_idx]  # (batch, seq, d_model)
            # Get the last token's hidden state
            last_hidden = hidden[0, -1, :]  # (d_model,)

            # Apply lm_head to get logits
            logits = lm_head(last_hidden)  # (vocab_size,)
            probs = torch.softmax(logits, dim=-1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                token_text = tokenizer.decode([idx.item()])
                predictions.append({
                    "token": token_text,
                    "token_id": int(idx.item()),
                    "probability": round(float(prob.item()), 4),
                })

            layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx}"
            layer_predictions.append({
                "layer": layer_idx,
                "layer_name": layer_name,
                "predictions": predictions,
            })

        return {
            "text": text,
            "tokens": token_labels,
            "n_layers": n_layers,
            "top_k": top_k,
            "model_name": self._model_name,
            "layers": layer_predictions,
            "explanation": (
                f"Logit Lens for {self._model_name}. "
                "By applying the output layer to intermediate hidden states, "
                "we can see how the model's prediction evolves through layers. "
                "Early layers often show generic tokens; later layers converge "
                "to the final prediction."
            ),
        }

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return _shared_hf_model is not None

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
