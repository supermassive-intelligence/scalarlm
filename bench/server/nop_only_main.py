"""Benchmark-only FastAPI entrypoint with just the nop + request_id surface.

Used by bench/scenarios/nop_sweep.sh to measure the FastAPI ceiling in
isolation from vLLM and everything else in the full app. Start with::

    python -m bench.server.nop_only_main

Exposes ``GET /v1/health`` (cheap 200) and ``GET /v1/bench/nop`` (empty
object). X-Request-Id middleware is applied, so the number you measure
is the cost of routing + middleware + nop handler — nothing more.

Kept out of the production graph on purpose: main.py carries the full
app; this script is a standalone benchmark oracle.
"""

from __future__ import annotations

import os
import sys

# Make the bundled packages importable when running from the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for subdir in ("infra",):
    path = os.path.join(_ROOT, subdir)
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI

from cray_infra.api.fastapi.middleware.request_id import (
    request_id_and_log_middleware,
)
from cray_infra.api.fastapi.routers.bench_router import bench_router


app = FastAPI()
app.middleware("http")(request_id_and_log_middleware)
app.include_router(bench_router, prefix="/v1")


@app.get("/v1/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("NOP_HOST", "0.0.0.0")
    port = int(os.environ.get("NOP_PORT", "8000"))
    workers = int(os.environ.get("NOP_WORKERS", "1"))
    uvicorn.run(
        "bench.server.nop_only_main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="warning",
    )
