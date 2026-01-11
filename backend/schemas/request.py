"""Pydantic request schemas for API endpoints.

This module provides type-safe request schemas with built-in validation,
eliminating manual validation code in route handlers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# ==============================================================================
# Common Types
# ==============================================================================


class GridSample(BaseModel):
    """A sample with either grid or flat x representation."""

    y: int = Field(..., description="Label (-1 or +1)")
    grid: list[list[int]] | None = Field(None, description="2D grid representation")
    x: list[float] | None = Field(None, description="Flat input vector")

    @field_validator("y")
    @classmethod
    def validate_y(cls, v: int) -> int:
        if v not in (-1, 1):
            raise ValueError("y must be -1 or +1")
        return v


# ==============================================================================
# Perceptron Request Schemas (Chapter 1)
# ==============================================================================


class PerceptronStepRequest(BaseModel):
    """Request schema for perceptron step."""

    dataset: str | None = Field(None, description="Dataset name (or, xor, custom)")
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    grid_rows: int | None = Field(None, ge=1, le=5, description="Grid rows for custom dataset")
    grid_cols: int | None = Field(None, ge=1, le=5, description="Grid cols for custom dataset")
    samples: list[GridSample] | None = Field(None, min_length=1, description="Custom samples")


class PerceptronResetRequest(BaseModel):
    """Request schema for perceptron reset."""

    dataset: str | None = Field(None, description="Dataset name (or, xor, custom)")
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    grid_rows: int | None = Field(None, ge=1, le=5, description="Grid rows for custom dataset")
    grid_cols: int | None = Field(None, ge=1, le=5, description="Grid cols for custom dataset")
    samples: list[GridSample] | None = Field(None, min_length=1, description="Custom samples")


# ==============================================================================
# LMS Request Schemas (Chapter 1)
# ==============================================================================


class LmsResetRequest(BaseModel):
    """Request schema for LMS reset."""

    dataset: str | None = Field(None, description="Dataset name (or, xor, custom)")
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    grid_rows: int | None = Field(None, ge=1, le=5, description="Grid rows for custom dataset")
    grid_cols: int | None = Field(None, ge=1, le=5, description="Grid cols for custom dataset")
    samples: list[GridSample] | None = Field(None, min_length=1, description="Custom samples")

    @model_validator(mode="after")
    def validate_lms_dimensions(self) -> LmsResetRequest:
        """LMS requires 2D inputs."""
        if (
            self.grid_rows is not None
            and self.grid_cols is not None
            and self.grid_rows * self.grid_cols != 2
        ):
            raise ValueError("LMS requires 2D inputs (grid_rows * grid_cols == 2)")
        return self


# ==============================================================================
# MLP Request Schemas (Chapter 3)
# ==============================================================================


class MlpResetRequest(BaseModel):
    """Request schema for MLP reset."""

    dataset: str | None = Field(None, description="Dataset name (or, xor, custom)")
    hidden_dim: int | None = Field(None, ge=1, le=256, description="Hidden layer dimension")
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    seed: int | None = Field(None, description="Random seed")
    grid_rows: int | None = Field(None, ge=1, le=5, description="Grid rows for custom dataset")
    grid_cols: int | None = Field(None, ge=1, le=5, description="Grid cols for custom dataset")
    samples: list[GridSample] | None = Field(None, min_length=1, description="Custom samples")


# ==============================================================================
# Backprop Request Schemas (Chapter 3)
# ==============================================================================


class RegressionSample(BaseModel):
    """A regression sample."""

    x: float = Field(..., description="Input value")
    y: float = Field(..., description="Target value")


class TinyGpsResetRequest(BaseModel):
    """Request schema for TinyGPS reset."""

    dataset: str | None = Field(None, description="Dataset name")
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    params: dict[str, float] | None = Field(None, description="Initial parameters")
    order: list[int] | None = Field(None, description="Sample order")


class RegressionResetRequest(BaseModel):
    """Request schema for regression reset."""

    loss: Literal["mse", "l1"] | None = Field(None, description="Loss function (mse, l1)")
    lr: float | None = Field(None, gt=0, le=10, description="Learning rate")
    samples: list[RegressionSample] | None = Field(None, min_length=1, description="Training samples")
    order: list[int] | None = Field(None, description="Sample order")
    params: dict[str, float] | None = Field(None, description="Initial parameters")


# ==============================================================================
# Diagnostics Request Schemas
# ==============================================================================


class ErrorSurfaceRequest(BaseModel):
    """Request schema for error surface computation."""

    dataset: str = Field(default="or", description="Dataset name (or, xor, custom)")
    grid_rows: int | None = Field(None, ge=1, le=5, description="Grid rows for custom")
    grid_cols: int | None = Field(None, ge=1, le=5, description="Grid cols for custom")
    samples: list[GridSample] | None = Field(None, description="Custom samples")
    steps: int = Field(default=25, ge=5, le=100, description="Grid resolution")
    w_min: float = Field(default=-2.0, description="Weight range min")
    w_max: float = Field(default=2.0, description="Weight range max")
    b: float = Field(default=0.0, description="Bias value")

    @model_validator(mode="after")
    def validate_2d_inputs(self) -> ErrorSurfaceRequest:
        """Error surface requires 2D inputs."""
        if (
            self.grid_rows is not None
            and self.grid_cols is not None
            and self.grid_rows * self.grid_cols != 2
        ):
            raise ValueError("error surface requires 2D inputs (grid_rows * grid_cols == 2)")
        return self


class MlpInternalsRequest(BaseModel):
    """Request schema for MLP internals visualization."""

    dataset: str = Field(default="or", description="Dataset name (or, xor, custom)")
    grid_rows: int | None = Field(None, ge=1, le=5, description="Grid rows for custom")
    grid_cols: int | None = Field(None, ge=1, le=5, description="Grid cols for custom")
    samples: list[GridSample] | None = Field(None, description="Custom samples")
    hidden_dim: int = Field(default=2, ge=1, le=256, description="Hidden dimension")
    lr: float = Field(default=0.5, gt=0, le=10, description="Learning rate")
    seed: int | None = Field(None, description="Random seed")
    sample_index: int = Field(default=0, ge=0, description="Sample index to inspect")


# ==============================================================================
# Convolution Request Schemas (Chapter 6)
# ==============================================================================


class ConvKernelRequest(BaseModel):
    """Request schema for setting convolution kernel."""

    name: str | None = Field(None, description="Preset kernel name")
    weights: list[list[float]] | None = Field(None, description="Custom 3x3 kernel weights")

    @model_validator(mode="after")
    def validate_kernel(self) -> ConvKernelRequest:
        if self.name is None and self.weights is None:
            raise ValueError("Must provide either 'name' or 'weights'")
        return self


class ConvParamsRequest(BaseModel):
    """Request schema for convolution parameters."""

    padding: int | None = Field(None, ge=0, le=4, description="Padding value")
    stride: int | None = Field(None, ge=1, le=4, description="Stride value")


class ConvImageSizeRequest(BaseModel):
    """Request schema for image size."""

    size: int = Field(..., ge=4, le=28, description="Image size")


class ConvCalculateRequest(BaseModel):
    """Request schema for output size calculation."""

    input_size: int = Field(..., ge=1, description="Input dimension")
    kernel_size: int = Field(..., ge=1, description="Kernel dimension")
    padding: int = Field(default=0, ge=0, description="Padding")
    stride: int = Field(default=1, ge=1, description="Stride")


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
