from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.api.deps import gd_service

router = APIRouter(prefix="/gd")


@router.get("/loss-curves")
def gd_loss_curves() -> dict[str, Any]:
    return gd_service.loss_curves()


@router.get("/token-losses")
def gd_token_losses(
    source: str | None = None,
    example_id: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    return gd_service.token_loss_examples(
        source_override=source,
        example_id=example_id,
        prompt=prompt,
    )


@router.get("/next-token-logprobs")
def gd_next_token_logprobs(prompt: str, limit: int | None = None) -> dict[str, Any]:
    return gd_service.next_token_logprobs(prompt=prompt, limit=limit)


@router.get("/ollama-status")
def gd_ollama_status() -> dict[str, Any]:
    return gd_service.ollama_status()
