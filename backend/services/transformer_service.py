"""Transformer visualization service for Chapter 5."""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

import tiktoken
import torch

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


@dataclass
class TokenInfo:
    """Information about a token."""

    token_id: int
    text: str
    position: int


@dataclass
class TokenizationResult:
    """Result of tokenizing text."""

    text: str
    tokens: list[TokenInfo]
    token_count: int


# GPT model configurations for scale comparison
GPT_MODELS = {
    "gpt2": {
        "name": "GPT-2",
        "year": 2019,
        "params": 1_500_000_000,
        "layers": 48,
        "heads": 25,
        "d_model": 1600,
        "context_length": 1024,
    },
    "gpt3": {
        "name": "GPT-3",
        "year": 2020,
        "params": 175_000_000_000,
        "layers": 96,
        "heads": 96,
        "d_model": 12288,
        "context_length": 2048,
    },
    "gpt4": {
        "name": "GPT-4",
        "year": 2023,
        "params": 1_800_000_000_000,  # Estimated
        "layers": 120,  # Estimated
        "heads": 96,  # Estimated
        "d_model": 16384,  # Estimated
        "context_length": 8192,
    },
}


class TransformerService:
    """Service for transformer visualization and education using real HuggingFace models."""

    def __init__(self) -> None:
        self._encoding: tiktoken.Encoding | None = None
        self._current_text = ""
        self._current_tokens: list[TokenInfo] = []
        # Lazy-loaded HuggingFace models
        self._hf_tokenizer = None
        self._hf_model = None
        self._model_name = "gpt2"  # GPT-2 small (124M params) - better than distilgpt2

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Lazy load the tiktoken encoding."""
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def _load_hf_model(self) -> None:
        """Lazy load the HuggingFace model and tokenizer."""
        if self._hf_tokenizer is None or self._hf_model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"Loading HuggingFace model: {self._model_name}")
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    output_hidden_states=True,
                    torch_dtype=torch.float32,
                )
                self._hf_model.eval()
                logger.info(f"Successfully loaded {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {e}")
                raise

    def tokenize(self, text: str) -> dict[str, Any]:
        """Tokenize text and return detailed token information.

        Args:
            text: Input text to tokenize

        Returns:
            Dictionary with tokenization details
        """
        token_ids = self.encoding.encode(text)
        tokens: list[TokenInfo] = []

        for i, token_id in enumerate(token_ids):
            token_text = self.encoding.decode([token_id])
            tokens.append(TokenInfo(
                token_id=token_id,
                text=token_text,
                position=i,
            ))

        self._current_text = text
        self._current_tokens = tokens

        return {
            "text": text,
            "token_count": len(tokens),
            "tokens": [
                {
                    "id": t.token_id,
                    "text": t.text,
                    "position": t.position,
                }
                for t in tokens
            ],
        }

    def get_embedding_info(self, text: str | None = None) -> dict[str, Any]:
        """Get real embedding values from a HuggingFace model.

        Args:
            text: Text to get embeddings for (uses cached if None)

        Returns:
            Dictionary with embedding visualization info
        """
        if text is not None:
            self.tokenize(text)

        # Load HuggingFace model
        self._load_hf_model()

        # Get embeddings from the model
        text_to_embed = text or self._current_text
        if not text_to_embed:
            text_to_embed = "Hello"

        # Tokenize with HF tokenizer
        inputs = self._hf_tokenizer(text_to_embed, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            # Get the embedding layer output (before transformer blocks)
            embedding_layer = self._hf_model.transformer.wte
            embeddings_tensor = embedding_layer(input_ids)  # [1, seq_len, d_model]

        # Extract embeddings
        embeddings_np = embeddings_tensor.squeeze(0).numpy()  # [seq_len, d_model]
        token_count, d_model = embeddings_np.shape

        # Decode tokens from HF tokenizer
        hf_tokens = [self._hf_tokenizer.decode([tid.item()]) for tid in input_ids[0]]

        embeddings = []
        for i, token_text in enumerate(hf_tokens):
            embedding = embeddings_np[i].tolist()
            embedding_preview = [round(float(v), 4) for v in embedding[:10]]
            embedding_rounded = [round(float(v), 4) for v in embedding]

            mean_val = float(embeddings_np[i].mean())
            std_val = float(embeddings_np[i].std())

            embeddings.append({
                "id": int(input_ids[0][i].item()),
                "text": token_text,
                "position": i,
                "embedding": embedding_rounded,
                "embedding_preview": embedding_preview,
                "min": round(float(embeddings_np[i].min()), 4),
                "max": round(float(embeddings_np[i].max()), 4),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
            })

        return {
            "text": text_to_embed,
            "token_count": token_count,
            "d_model": d_model,
            "model_name": self._model_name,
            "matrix_shape": [token_count, d_model],
            "explanation": (
                f"Real embeddings from {self._model_name}. Each of the {token_count} tokens is "
                f"converted to a {d_model}-dimensional vector. These are learned representations "
                "that capture semantic meaning."
            ),
            "tokens": embeddings,
            "full_matrix": [e["embedding"] for e in embeddings],
        }

    def get_block_trace(self, text: str | None = None, n_blocks: int = 12) -> dict[str, Any]:
        """Show how the matrix flows through transformer blocks.

        Educational visualization of the transformer architecture.

        Args:
            text: Text to tokenize (uses cached if None)
            n_blocks: Number of blocks to show (for smaller visualization)

        Returns:
            Dictionary with block trace information
        """
        if text is not None:
            self.tokenize(text)

        token_count = len(self._current_tokens)
        d_model = 768
        n_heads = 12
        d_head = d_model // n_heads  # 64
        d_ff = d_model * 4  # 3072

        blocks = []
        for i in range(n_blocks):
            blocks.append({
                "block": i + 1,
                "input_shape": [token_count, d_model],
                "output_shape": [token_count, d_model],
                "operations": [
                    {
                        "name": "Multi-Head Attention",
                        "formula": "Attention(Q, K, V) = softmax(QK^T / √d_k) × V",
                        "details": {
                            "n_heads": n_heads,
                            "d_head": d_head,
                            "q_shape": [token_count, d_model],
                            "k_shape": [token_count, d_model],
                            "v_shape": [token_count, d_model],
                            "attention_scores_shape": [n_heads, token_count, token_count],
                            "explanation": f"Each head computes attention over all {token_count} positions. "
                                          f"Scores are scaled by 1/√{d_head} = {1/math.sqrt(d_head):.4f}",
                        },
                    },
                    {
                        "name": "Add & LayerNorm",
                        "formula": "LayerNorm(x + Attention(x))",
                        "details": {
                            "explanation": "Residual connection followed by layer normalization",
                        },
                    },
                    {
                        "name": "Feed Forward",
                        "formula": "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂",
                        "details": {
                            "hidden_dim": d_ff,
                            "input_shape": [token_count, d_model],
                            "hidden_shape": [token_count, d_ff],
                            "output_shape": [token_count, d_model],
                            "explanation": f"Expands to {d_ff} dims (4x), applies ReLU/GELU, projects back to {d_model}",
                        },
                    },
                    {
                        "name": "Add & LayerNorm",
                        "formula": "LayerNorm(x + FFN(x))",
                        "details": {
                            "explanation": "Residual connection followed by layer normalization",
                        },
                    },
                ],
            })

        return {
            "text": self._current_text,
            "token_count": token_count,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_head": d_head,
            "d_ff": d_ff,
            "n_blocks": n_blocks,
            "blocks": blocks,
            "final_shape": [token_count, d_model],
            "explanation": (
                f"The input matrix ({token_count}, {d_model}) passes through {n_blocks} transformer blocks. "
                "Each block maintains the same shape. "
                "Only the LAST COLUMN of the final output determines the next token."
            ),
        }

    def _ollama_chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Chat with Ollama using the chat API for natural responses.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Dictionary with response and metadata
        """
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
            },
        }

        url = f"{OLLAMA_BASE_URL}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                body = resp.read()
            result = json.loads(body.decode("utf-8"))
            message = result.get("message", {})
            return {
                "response": message.get("content", ""),
                "done": result.get("done", True),
                "eval_count": result.get("eval_count", 0),
                "total_duration": result.get("total_duration", 0),
            }
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    def chat_stream(self, messages: list[dict[str, str]]):
        """Stream chat response from Ollama token by token.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Yields:
            Dict with token or done status
        """
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
            },
        }

        url = f"{OLLAMA_BASE_URL}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        message = chunk.get("message", {})
                        content = message.get("content", "")
                        done = chunk.get("done", False)

                        if content:
                            yield {"token": content, "done": False}

                        if done:
                            yield {
                                "token": "",
                                "done": True,
                                "total_duration": chunk.get("total_duration", 0),
                                "eval_count": chunk.get("eval_count", 0),
                            }
                            break
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            yield {"token": "", "done": True, "error": str(e)}

    def _ollama_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            url = f"{OLLAMA_BASE_URL}/api/tags"
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def ollama_status(self) -> dict[str, Any]:
        """Get detailed Ollama status (similar to /gd endpoint)."""
        try:
            url = f"{OLLAMA_BASE_URL}/api/tags"
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                body = resp.read()
            data = json.loads(body.decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return {
                "ok": True,
                "model": OLLAMA_MODEL,
                "base_url": OLLAMA_BASE_URL,
                "models": models,
                "model_available": OLLAMA_MODEL in models or any(OLLAMA_MODEL.split(":")[0] in m for m in models),
            }
        except Exception as e:
            return {
                "ok": False,
                "model": OLLAMA_MODEL,
                "base_url": OLLAMA_BASE_URL,
                "error": str(e),
                "models": [],
                "model_available": False,
            }

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Chat with the LLM and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Dictionary with chat result
        """
        # Check if Ollama is available
        if not self._ollama_available():
            return {
                "model_name": OLLAMA_MODEL,
                "response": "",
                "generation_time_ms": 0,
                "error": "Ollama not available",
            }

        try:
            result = self._ollama_chat(messages)
            response = result["response"]
            duration_ns = result.get("total_duration", 0)
            duration_ms = duration_ns / 1_000_000 if duration_ns else 0

            return {
                "model_name": OLLAMA_MODEL,
                "response": response,
                "generation_time_ms": round(duration_ms, 1),
            }
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "model_name": OLLAMA_MODEL,
                "response": "",
                "generation_time_ms": 0,
                "error": f"Chat failed: {e}",
            }

    def get_model_comparison(self) -> dict[str, Any]:
        """Get comparison data for different GPT models.

        Returns:
            Dictionary with model scale comparison
        """
        return {
            "models": [
                {
                    "id": model_id,
                    **info,
                }
                for model_id, info in GPT_MODELS.items()
            ],
            "explanation": (
                "GPT models have grown exponentially in size. "
                "GPT-4 has roughly 1000x more parameters than GPT-2."
            ),
        }

    def state(self) -> dict[str, Any]:
        """Get current service state."""
        hf_model_loaded = self._hf_model is not None
        ollama_available = self._ollama_available()
        return {
            "current_text": self._current_text,
            "current_token_count": len(self._current_tokens),
            "encoding_name": "cl100k_base",
            "vocab_size": self.encoding.n_vocab,
            "hf_model": self._model_name,
            "hf_model_loaded": hf_model_loaded,
            "ollama_model": OLLAMA_MODEL,
            "ollama_available": ollama_available,
        }
