"""FastAPI server for perceptron + diagnostics + LMS."""

from __future__ import annotations

import logging
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.api import (
    alexnet_router,
    backprop_router,
    conv_router,
    deep_router,
    diagnostics_router,
    gd_router,
    lms_router,
    mlp_router,
    perceptron_router,
    transformer_router,
)

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

# Create logger
logger = logging.getLogger("perceptron_api")
logger.setLevel(logging.DEBUG)

# File handler - detailed logs
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_format)

# Console handler - info level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(levelname)-8s | %(message)s")
console_handler.setFormatter(console_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = FastAPI(title="Perceptron Visual Lab API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Response]) -> Response:
    """Log all requests and responses."""
    request_id = f"{time.time():.0f}"
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""

    # Log request
    logger.info(f"[{request_id}] --> {method} {path} {query}")

    # Log request body for POST (excluding file uploads for size reasons)
    if method == "POST":
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            logger.debug(f"[{request_id}] Body: [file upload]")
        else:
            try:
                body = await request.body()
                if body:
                    logger.debug(f"[{request_id}] Body: {body[:500].decode('utf-8', errors='replace')}")
            except Exception:
                pass

    start_time = time.time()

    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000

        # Log response
        logger.info(f"[{request_id}] <-- {response.status_code} ({duration:.1f}ms)")

        return response

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] <-- ERROR ({duration:.1f}ms): {e}")
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        raise


app.include_router(perceptron_router)
app.include_router(diagnostics_router)
app.include_router(lms_router)
app.include_router(mlp_router)
app.include_router(gd_router)
app.include_router(backprop_router)
app.include_router(deep_router)
app.include_router(alexnet_router)
app.include_router(transformer_router)
app.include_router(conv_router)

logger.info(f"API logging enabled. Log file: {LOG_FILE.absolute()}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("backend.api_app:app", host=host, port=port, log_level="info")
