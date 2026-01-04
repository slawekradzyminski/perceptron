from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter

from backend.api.deps import gd_service

router = APIRouter(prefix="/gd")


@router.get("/loss-curves")
def gd_loss_curves() -> Dict[str, Any]:
    return gd_service.loss_curves()


@router.get("/token-losses")
def gd_token_losses(
    source: Optional[str] = None,
    example_id: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return gd_service.token_loss_examples(
        source_override=source,
        example_id=example_id,
        prompt=prompt,
    )


@router.get("/next-token-logprobs")
def gd_next_token_logprobs(prompt: str, limit: Optional[int] = None) -> Dict[str, Any]:
    return gd_service.next_token_logprobs(prompt=prompt, limit=limit)


@router.get("/ollama-status")
def gd_ollama_status() -> Dict[str, Any]:
    return gd_service.ollama_status()
