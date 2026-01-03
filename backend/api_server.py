"""Backward-compatible entrypoint for perceptron API (FastAPI)."""

from backend.api_app import app, run

__all__ = ["app", "run"]
