"""Deprecation-log dependency for the client-facing Path A endpoints.

Phase 4 of the enhancement plan. The long-term direction is a single
OpenAI-compatible surface; Path A (``/v1/generate``, ``/v1/generate/
get_results``, ``/v1/generate/upload``, ``/v1/generate/download``) is
frozen and will be removed once telemetry shows no one is using it.

This module emits one structured log line per hit to those endpoints,
carrying ``User-Agent`` and client IP so we can see who to migrate
before pulling the plug. It is **not** attached to the worker-facing
endpoints (``get_work``, ``finish_work``, ``get_adaptors``,
``clear_queue``, ``metrics``) — those are ScalarLM-internal and survive
the Path A deprecation.
"""

from __future__ import annotations

import json
import logging

from fastapi import Request

logger = logging.getLogger(__name__)


_MIGRATION_HINTS = {
    "/v1/generate": "POST /v1/chat/completions (chat) or /v1/completions with an array prompt (bulk completions)",
    "/v1/generate/get_results": "batches: GET /v1/batches/{id}",
    "/v1/generate/upload": "POST /v1/batches with JSONL input",
    "/v1/generate/download": "GET /v1/batches/{id}/output_file_content",
}


async def log_path_a_deprecation(request: Request) -> None:
    path = request.url.path
    logger.warning(json.dumps({
        "event": "path_a_deprecation",
        "path": path,
        "user_agent": request.headers.get("user-agent"),
        "client": request.client.host if request.client else None,
        "request_id": getattr(request.state, "request_id", None),
        "migration_hint": _MIGRATION_HINTS.get(path),
    }))
