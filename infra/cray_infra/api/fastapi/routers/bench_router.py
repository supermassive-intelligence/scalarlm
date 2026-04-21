"""Benchmark-only endpoints.

Exists so the benchmark plan in ``enhance-openai-api.md`` can separate
the FastAPI / uvicorn / middleware ceiling from the inference-layer
ceiling. Only enabled when ``bench_endpoints_enabled`` is set in config
so production builds don't accidentally expose a no-op that bypasses
auth / rate limits.
"""

from fastapi import APIRouter

bench_router = APIRouter()


@bench_router.get("/bench/nop")
async def nop():
    """Returns an empty object as cheaply as the framework allows.

    Intentionally does no work: no config read, no session creation, no
    logging beyond the request_id middleware. The number you measure
    against this endpoint is the cost of FastAPI routing + the
    X-Request-Id middleware on this platform, nothing more.
    """
    return {}
