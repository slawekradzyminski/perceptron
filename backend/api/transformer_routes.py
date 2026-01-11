"""Transformer visualization routes for Chapter 5."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.api.deps import scale_service, transformer_service
from backend.schemas.request import (
    AttentionRequest,
    ChatRequest,
    EmbedRequest,
    KVCacheRequest,
    LogitLensRequest,
    TokenizeRequest,
    TraceRequest,
)

router = APIRouter(prefix="/transformer")


@router.get("/state")
def transformer_state() -> dict[str, Any]:
    """Get current transformer service state."""
    return transformer_service.state()


@router.post("/tokenize")
def transformer_tokenize(body: TokenizeRequest) -> dict[str, Any]:
    """Tokenize text and return detailed token information."""
    return transformer_service.tokenize(body.text)


@router.post("/embed")
def transformer_embed(body: EmbedRequest) -> dict[str, Any]:
    """Get embedding information for tokens."""
    return transformer_service.get_embedding_info(body.text)


@router.post("/trace")
def transformer_trace(body: TraceRequest) -> dict[str, Any]:
    """Show how the matrix flows through transformer blocks."""
    return transformer_service.get_block_trace(body.text, body.n_blocks)


@router.post("/chat")
def transformer_chat(body: ChatRequest) -> dict[str, Any]:
    """Chat with the LLM."""
    messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]
    return transformer_service.chat(messages)


@router.post("/chat/stream")
def transformer_chat_stream(body: ChatRequest) -> StreamingResponse:
    """Stream chat response from the LLM token by token."""
    messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]

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
def transformer_attention(body: AttentionRequest) -> dict[str, Any]:
    """Get attention patterns for text (Chapter 8)."""
    return transformer_service.get_attention_patterns(body.text)


@router.post("/logit-lens")
def transformer_logit_lens(body: LogitLensRequest) -> dict[str, Any]:
    """Get Logit Lens predictions for text (Chapter 7)."""
    return transformer_service.get_logit_lens(body.text, body.top_k)


@router.post("/kv-cache")
def transformer_kv_cache(body: KVCacheRequest) -> dict[str, Any]:
    """Calculate KV cache memory comparison (Chapter 8)."""
    return transformer_service.get_kv_cache_comparison(
        context_length=body.context_length,
        n_layers=body.n_layers,
        d_model=body.d_model,
        n_heads=body.n_heads,
        gqa_groups=body.gqa_groups,
        mla_latent_dim=body.mla_latent_dim,
    )

