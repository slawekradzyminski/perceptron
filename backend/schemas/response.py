"""Pydantic response models for API endpoints.

This module provides type-safe response models for all API endpoints,
ensuring consistent JSON serialization and automatic OpenAPI documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ==============================================================================
# Perceptron Models
# ==============================================================================


class PerceptronStateResponse(BaseModel):
    """Response model for perceptron state."""

    w: list[float] = Field(description="Current weights")
    b: float = Field(description="Current bias")
    idx: int = Field(description="Current sample index")
    dataset: str = Field(description="Dataset name")
    lr: float = Field(description="Learning rate")
    next_x: list[float] = Field(description="Next sample input")
    next_y: int = Field(description="Next sample label")
    grid_rows: int = Field(description="Grid rows")
    grid_cols: int = Field(description="Grid columns")
    sample_count: int = Field(description="Total samples in dataset")


class PerceptronStepResponse(PerceptronStateResponse):
    """Response model for perceptron step."""

    x: list[float] = Field(description="Input used in this step")
    y: int = Field(description="True label for this step")
    score: float = Field(description="Computed score")
    pred: int = Field(description="Prediction")
    mistake: bool = Field(description="Whether a mistake was made")
    delta_w: list[float] = Field(description="Weight update")
    delta_b: float = Field(description="Bias update")


# ==============================================================================
# Deep Learning Models (Chapter 4)
# ==============================================================================


class DeepArchitecture(BaseModel):
    """Architecture information for deep MLP."""

    input_dim: int = Field(description="Input dimension")
    hidden_dims: list[int] = Field(description="Hidden layer dimensions")
    output_dim: int = Field(description="Output dimension")
    depth: int = Field(description="Number of hidden layers")
    width: int = Field(description="Maximum width of hidden layers")
    param_count: int = Field(description="Total parameter count")


class DeepMetrics(BaseModel):
    """Training metrics for deep MLP."""

    loss: float = Field(description="Current loss")
    accuracy: float = Field(description="Current accuracy")


class DeepDatasetInfo(BaseModel):
    """Dataset information."""

    name: str = Field(description="Display name")
    description: str = Field(description="Dataset description")
    n_classes: int = Field(description="Number of classes")


class DeepSample(BaseModel):
    """A training sample."""

    x: list[float] = Field(description="Input features")
    y: int = Field(description="Label")


class DeepStateResponse(BaseModel):
    """Response model for deep learning state."""

    dataset: str = Field(description="Dataset name")
    dataset_info: DeepDatasetInfo = Field(description="Dataset information")
    sample_count: int = Field(description="Number of samples")
    idx: int = Field(description="Current sample index")
    epoch: int = Field(description="Current epoch")
    total_steps: int = Field(description="Total training steps")
    lr: float = Field(description="Learning rate")
    architecture: DeepArchitecture = Field(description="Model architecture")
    metrics: DeepMetrics = Field(description="Current metrics")
    samples: list[DeepSample] = Field(description="Dataset samples")


class DeepStepInfo(BaseModel):
    """Step-specific training information."""

    batch_size: int = Field(description="Batch size used")
    last_loss: float = Field(description="Loss for last sample")
    last_correct: bool = Field(description="Whether last prediction was correct")
    last_prediction: int = Field(description="Predicted class")
    grad_norm: float = Field(description="Gradient norm")


class DeepStepResponse(DeepStateResponse):
    """Response model for deep learning step."""

    step_info: DeepStepInfo = Field(description="Step information")


class DeepRegionsResponse(BaseModel):
    """Response model for region counting."""

    count: int = Field(description="Actual region count")
    theoretical_max: int = Field(description="Theoretical maximum regions")
    efficiency: float = Field(description="Ratio of actual to theoretical")


class DeepBoundaryResponse(BaseModel):
    """Response model for boundary visualization."""

    resolution: int = Field(description="Grid resolution")
    predictions: list[list[int]] = Field(description="Predicted class grid")
    region_ids: list[list[int]] = Field(description="Region ID grid")
    region_count: int = Field(description="Number of unique regions")
    theoretical_max: int = Field(description="Theoretical maximum")


class DeepHistoryEntry(BaseModel):
    """A single training history entry."""

    step: int = Field(description="Step number")
    epoch: int = Field(description="Epoch number")
    loss: float = Field(description="Loss at this step")
    accuracy: float = Field(description="Accuracy at this step")


class DeepHistoryResponse(BaseModel):
    """Response model for training history."""

    history: list[DeepHistoryEntry] = Field(description="Training history")


class DeepComparisonEntry(BaseModel):
    """An architecture comparison entry."""

    depth: int = Field(description="Number of hidden layers")
    width: int = Field(description="Maximum hidden layer width")
    hidden_dims: list[int] = Field(description="Hidden layer dimensions")
    param_count: int = Field(description="Parameter count")
    actual_regions: int = Field(description="Actual region count")
    theoretical_max: int = Field(description="Theoretical maximum")
    accuracy: float = Field(description="Final accuracy")
    loss: float = Field(description="Final loss")
    total_steps: int = Field(description="Total training steps")


class DeepComparisonResponse(BaseModel):
    """Response model for comparison table."""

    table: list[DeepComparisonEntry] = Field(description="Comparison entries")


class ClearedResponse(BaseModel):
    """Generic response for clear operations."""

    cleared: bool = Field(default=True, description="Whether clearing succeeded")


# ==============================================================================
# Glass Box Models (Chapters 7 & 8)
# ==============================================================================


class AttentionResult(BaseModel):
    """Response model for attention patterns."""

    text: str = Field(description="Input text")
    tokens: list[str] = Field(description="Tokenized text")
    n_layers: int = Field(description="Number of layers")
    n_heads: int = Field(description="Number of attention heads")
    seq_len: int = Field(description="Sequence length")
    model_name: str = Field(description="Model name")
    attentions: list[list[list[list[float]]]] = Field(
        description="Attention weights [layers][heads][seq][seq]"
    )
    explanation: str = Field(description="Educational explanation")


class LogitLensPrediction(BaseModel):
    """A single logit lens prediction."""

    token: str = Field(description="Predicted token")
    token_id: int = Field(description="Token ID")
    probability: float = Field(description="Probability")


class LogitLensLayer(BaseModel):
    """Logit lens predictions for a single layer."""

    layer: int = Field(description="Layer index")
    layer_name: str = Field(description="Layer name")
    predictions: list[LogitLensPrediction] = Field(description="Top predictions")


class LogitLensResult(BaseModel):
    """Response model for logit lens."""

    text: str = Field(description="Input text")
    tokens: list[str] = Field(description="Tokenized text")
    n_layers: int = Field(description="Number of layers")
    top_k: int = Field(description="Number of top predictions")
    model_name: str = Field(description="Model name")
    layers: list[LogitLensLayer] = Field(description="Per-layer predictions")
    explanation: str = Field(description="Educational explanation")


class KVCacheConfig(BaseModel):
    """KV cache configuration parameters."""

    context_length: int = Field(description="Context/sequence length")
    n_layers: int = Field(description="Number of layers")
    d_model: int = Field(description="Model dimension")
    n_heads: int = Field(description="Number of attention heads")
    d_head: int = Field(description="Dimension per head")
    gqa_groups: int = Field(description="GQA groups")
    mla_latent_dim: int = Field(description="MLA latent dimension")
    bytes_per_param: int = Field(description="Bytes per parameter")


class KVCacheArchitecture(BaseModel):
    """KV cache memory for an architecture."""

    name: str = Field(description="Architecture abbreviation")
    full_name: str = Field(description="Full architecture name")
    description: str = Field(description="Architecture description")
    memory_bytes: int = Field(description="Memory usage in bytes")
    memory_formatted: str = Field(description="Human-readable memory")
    ratio_to_mha: float = Field(description="Ratio compared to MHA")
    kv_heads: int | None = Field(None, description="Number of KV heads")
    latent_dim: int | None = Field(None, description="Latent dimension (MLA)")


class KVCacheResult(BaseModel):
    """Response model for KV cache comparison."""

    config: KVCacheConfig = Field(description="Configuration used")
    architectures: list[KVCacheArchitecture] = Field(description="Architectures")
    mha_to_mla_savings: float = Field(description="MHA to MLA savings ratio")
    explanation: str = Field(description="Educational explanation")
