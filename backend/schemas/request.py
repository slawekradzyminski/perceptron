"""Pydantic request schemas for API endpoints.

This module provides type-safe request schemas with built-in validation,
eliminating manual validation code in route handlers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationInfo, field_validator

# ==============================================================================
# Deep Learning Request Schemas (Chapter 4)
# ==============================================================================


class DeepResetRequest(BaseModel):
    """Request schema for deep learning reset."""

    dataset: str | None = Field(None, description="Dataset name (baarle, xor, circles, spiral)")
    hidden_dims: list[int] | None = Field(
        None,
        min_length=1,
        max_length=10,
        description="Width of each hidden layer, e.g. [16, 16, 16]",
    )
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    seed: int | None = Field(None, description="Random seed")

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            for i, dim in enumerate(v):
                if dim <= 0 or dim > 256:
                    raise ValueError(f"hidden_dims[{i}] must be an integer in [1, 256]")
        return v


class DeepStepRequest(BaseModel):
    """Request schema for deep learning step."""

    batch_size: int = Field(
        default=1,
        ge=1,
        le=10000,
        description="Number of samples to train on",
    )


# ==============================================================================
# Transformer Request Schemas (Chapter 5)
# ==============================================================================


class TokenizeRequest(BaseModel):
    """Request schema for tokenization."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to tokenize")


class EmbedRequest(BaseModel):
    """Request schema for embedding."""

    text: str | None = Field(
        None,
        max_length=10000,
        description="Text to embed (uses cached if not provided)",
    )


class TraceRequest(BaseModel):
    """Request schema for block trace."""

    text: str | None = Field(None, max_length=10000, description="Text to process")
    n_blocks: int = Field(
        default=12,
        ge=1,
        le=96,
        description="Number of blocks to show",
    )


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request schema for chat."""

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Array of message objects",
    )


# ==============================================================================
# Glass Box Request Schemas (Chapters 7 & 8)
# ==============================================================================


class AttentionRequest(BaseModel):
    """Request schema for attention patterns."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text to analyze (max 500 chars for performance)",
    )


class LogitLensRequest(BaseModel):
    """Request schema for logit lens."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text to analyze",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top predictions per layer",
    )


class KVCacheRequest(BaseModel):
    """Request schema for KV cache comparison."""

    context_length: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
        description="Sequence length",
    )
    n_layers: int = Field(
        default=32,
        ge=1,
        le=200,
        description="Number of layers",
    )
    d_model: int = Field(
        default=4096,
        ge=64,
        le=65536,
        description="Hidden dimension",
    )
    n_heads: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Number of attention heads",
    )
    gqa_groups: int = Field(
        default=8,
        ge=1,
        description="Heads per KV group for GQA",
    )
    mla_latent_dim: int = Field(
        default=512,
        ge=1,
        description="Latent dimension for MLA",
    )

    @field_validator("gqa_groups")
    @classmethod
    def validate_gqa_groups(cls, v: int, info: ValidationInfo) -> int:
        n_heads = info.data.get("n_heads", 32) if info.data else 32
        if v > n_heads:
            raise ValueError("gqa_groups must be <= n_heads")
        return v

    @field_validator("mla_latent_dim")
    @classmethod
    def validate_mla_latent_dim(cls, v: int, info: ValidationInfo) -> int:
        d_model = info.data.get("d_model", 4096) if info.data else 4096
        if v > d_model:
            raise ValueError("mla_latent_dim must be <= d_model")
        return v
