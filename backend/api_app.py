"""FastAPI server for perceptron + diagnostics + LMS."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import diagnostics_router, lms_router, perceptron_router

app = FastAPI(title="Perceptron Visual Lab API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(perceptron_router)
app.include_router(diagnostics_router)
app.include_router(lms_router)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("backend.api_app:app", host=host, port=port, log_level="info")
