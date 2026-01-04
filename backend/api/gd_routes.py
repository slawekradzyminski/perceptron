from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.api.deps import gd_service

router = APIRouter(prefix="/gd")


@router.get("/loss-curves")
def gd_loss_curves() -> Dict[str, Any]:
    return gd_service.loss_curves()


@router.get("/token-losses")
def gd_token_losses() -> Dict[str, Any]:
    return gd_service.token_loss_examples()
