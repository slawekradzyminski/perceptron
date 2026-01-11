"""Transformer visualization routes for Chapter 5."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.api.deps import scale_service, transformer_service

router = APIRouter(prefix="/transformer")


@router.get("/state")
def transformer_state() -> dict[str, Any]:
    """Get current transformer service state."""
    return transformer_service.state()


@router.post("/tokenize")
def transformer_tokenize(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Tokenize text and return detailed token information.

    Body:
        text: str - Text to tokenize
    """
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string")
    if len(text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")

    return transformer_service.tokenize(text)


@router.post("/embed")
def transformer_embed(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Get embedding information for tokens.

    Body:
        text: str - Text to embed (optional, uses cached if not provided)
    """
    text = body.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="'text' must be a string")
        if len(text) > 10000:
            raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")

    return transformer_service.get_embedding_info(text)


@router.post("/trace")
def transformer_trace(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Show how the matrix flows through transformer blocks.

    Body:
        text: str - Text to process (optional, uses cached if not provided)
        n_blocks: int - Number of blocks to show (default: 12)
    """
    text = body.get("text")
    n_blocks = body.get("n_blocks", 12)

    if text is not None:
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="'text' must be a string")
        if len(text) > 10000:
            raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")

    if not isinstance(n_blocks, int) or n_blocks < 1 or n_blocks > 96:
        raise HTTPException(status_code=400, detail="n_blocks must be an integer between 1 and 96")

    return transformer_service.get_block_trace(text, n_blocks)


@router.post("/chat")
def transformer_chat(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Chat with the LLM.

    Body:
        messages: list - Array of {role, content} message objects
    """
    messages = body.get("messages")

    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' field")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")
    if len(messages) > 50:
        raise HTTPException(status_code=400, detail="Too many messages (max 50)")

    # Validate message format
    for msg in messages:
        if not isinstance(msg, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object")
        if "role" not in msg or "content" not in msg:
            raise HTTPException(status_code=400, detail="Each message must have 'role' and 'content'")

    return transformer_service.chat(messages)


@router.post("/chat/stream")
def transformer_chat_stream(body: dict[str, Any] = Body(...)) -> StreamingResponse:
    """Stream chat response from the LLM token by token.

    Body:
        messages: list - Array of {role, content} message objects
    """
    messages = body.get("messages")

    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' field")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")
    if len(messages) > 50:
        raise HTTPException(status_code=400, detail="Too many messages (max 50)")

    # Validate message format
    for msg in messages:
        if not isinstance(msg, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object")
        if "role" not in msg or "content" not in msg:
            raise HTTPException(status_code=400, detail="Each message must have 'role' and 'content'")

    def generate():
        for chunk in transformer_service.chat_stream(messages):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/ollama-status")
def transformer_ollama_status() -> dict[str, Any]:
    """Get Ollama status for generation tab."""
    return transformer_service.ollama_status()


@router.get("/models")
def transformer_models() -> dict[str, Any]:
    """Get GPT model comparison data."""
    return transformer_service.get_model_comparison()


# Scale comparison endpoints (shared between AlexNet and Transformer pages)
@router.get("/scale")
def scale_all_models() -> dict[str, Any]:
    """Get all models for scale comparison."""
    return scale_service.get_all_models()


@router.get("/scale/cnns")
def scale_cnns() -> dict[str, Any]:
    """Get CNN models only."""
    return scale_service.get_cnns()


@router.get("/scale/transformers")
def scale_transformers() -> dict[str, Any]:
    """Get Transformer models only."""
    return scale_service.get_transformers()


@router.get("/scale/compare")
def scale_compare(
    model1: str = Query(..., description="First model name"),
    model2: str = Query(..., description="Second model name"),
) -> dict[str, Any]:
    """Compare two models by name."""
    try:
        return scale_service.get_comparison(model1, model2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/scale/growth")
def scale_growth() -> dict[str, Any]:
    """Get exponential growth data for visualization."""
    return scale_service.get_growth_data()


# Chapter 7 & 8: Glass Box endpoints
@router.post("/attention")
def transformer_attention(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Get attention patterns for text (Chapter 8).

    Body:
        text: str - Text to analyze
    """
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string")
    if len(text) > 500:
        raise HTTPException(
            status_code=400,
            detail="Text too long (max 500 chars for attention visualization)",
        )

    return transformer_service.get_attention_patterns(text)


@router.post("/logit-lens")
def transformer_logit_lens(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Get Logit Lens predictions for text (Chapter 7).

    The Logit Lens shows what the model "believes" at each layer.

    Body:
        text: str - Text to analyze
        top_k: int - Number of top predictions per layer (default: 5)
    """
    text = body.get("text")
    top_k = body.get("top_k", 5)

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string")
    if len(text) > 500:
        raise HTTPException(
            status_code=400,
            detail="Text too long (max 500 chars for logit lens)",
        )
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be an integer between 1 and 20")

    return transformer_service.get_logit_lens(text, top_k)


@router.post("/kv-cache")
def transformer_kv_cache(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Calculate KV cache memory comparison (Chapter 8).

    Compares MHA, MQA, GQA, and MLA memory usage.

    Body:
        context_length: int - Sequence length (default: 4096)
        n_layers: int - Number of layers (default: 32)
        d_model: int - Hidden dimension (default: 4096)
        n_heads: int - Number of attention heads (default: 32)
        gqa_groups: int - Heads per KV group for GQA (default: 8)
        mla_latent_dim: int - Latent dimension for MLA (default: 512)
    """
    context_length = body.get("context_length", 4096)
    n_layers = body.get("n_layers", 32)
    d_model = body.get("d_model", 4096)
    n_heads = body.get("n_heads", 32)
    gqa_groups = body.get("gqa_groups", 8)
    mla_latent_dim = body.get("mla_latent_dim", 512)

    # Validation
    if not isinstance(context_length, int) or context_length < 1 or context_length > 1_000_000:
        raise HTTPException(status_code=400, detail="context_length must be 1-1,000,000")
    if not isinstance(n_layers, int) or n_layers < 1 or n_layers > 200:
        raise HTTPException(status_code=400, detail="n_layers must be 1-200")
    if not isinstance(d_model, int) or d_model < 64 or d_model > 65536:
        raise HTTPException(status_code=400, detail="d_model must be 64-65536")
    if not isinstance(n_heads, int) or n_heads < 1 or n_heads > 256:
        raise HTTPException(status_code=400, detail="n_heads must be 1-256")
    if not isinstance(gqa_groups, int) or gqa_groups < 1 or gqa_groups > n_heads:
        raise HTTPException(status_code=400, detail="gqa_groups must be 1 to n_heads")
    if not isinstance(mla_latent_dim, int) or mla_latent_dim < 1 or mla_latent_dim > d_model:
        raise HTTPException(status_code=400, detail="mla_latent_dim must be 1 to d_model")

    return transformer_service.get_kv_cache_comparison(
        context_length=context_length,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        gqa_groups=gqa_groups,
        mla_latent_dim=mla_latent_dim,
    )

