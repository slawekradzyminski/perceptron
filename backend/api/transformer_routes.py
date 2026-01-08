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

