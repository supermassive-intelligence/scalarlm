"""Request-ID + structured request-log middleware.

Lives in its own module so unit tests can exercise the middleware without
importing the full `cray_infra.api.fastapi.main` (which transitively pulls
in vLLM and the whole app graph).
"""

import json
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def request_id_and_log_middleware(request: Request, call_next):
    """Assign an ``X-Request-Id`` to every request and emit structured logs.

    Honours a caller-supplied ``X-Request-Id`` header when present so upstream
    correlation IDs survive; otherwise generates a fresh UUID4 hex. The id is
    stashed on ``request.state.request_id`` for handlers and echoed on the
    response so clients (and the SDK's retry layer) can quote it when
    reporting issues.
    """
    start_time = time.time()

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id

    trace_id = None
    traceparent = request.headers.get("traceparent", "")
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 2:
            trace_id = parts[1]

    logger.info(json.dumps({
        "event": "request_start",
        "request_id": request_id,
        "trace_id": trace_id,
        "method": request.method,
        "path": str(request.url.path),
        "client": request.client.host if request.client else None,
        "timestamp": start_time,
    }))

    try:
        response = await call_next(request)
    except Exception as exc:
        # When a handler raises, we still owe the client a response carrying
        # X-Request-Id — that's precisely the scenario where ops wants to
        # pivot from the response to the logs.
        duration = time.time() - start_time
        logger.exception(json.dumps({
            "event": "request_error",
            "request_id": request_id,
            "trace_id": trace_id,
            "method": request.method,
            "path": str(request.url.path),
            "error": str(exc),
            "duration_seconds": round(duration, 3),
            "timestamp": time.time(),
        }))
        response = JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "request_id": request_id},
        )

    response.headers["X-Request-Id"] = request_id

    duration = time.time() - start_time
    logger.info(json.dumps({
        "event": "request_end",
        "request_id": request_id,
        "trace_id": trace_id,
        "method": request.method,
        "path": str(request.url.path),
        "status_code": response.status_code,
        "duration_seconds": round(duration, 3),
        "timestamp": time.time(),
    }))

    return response
