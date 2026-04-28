"""
OpenAI v1 API Router - Standard OpenAI-compatible endpoints.
These endpoints are exposed directly under /v1/ to match OpenAI API spec.

Both /v1/chat/completions and /v1/completions proxy to vLLM. They also feed
the same `Metrics` counter that the queue-based /v1/generate path updates, so
the metrics card on /app/metrics reflects ALL inference traffic — not just
queue-routed requests. See _wrap_with_metrics below.
"""

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _CHAT_ALLOWED_KEYS,
    _COMPLETION_ALLOWED_KEYS,
    _USAGE_SCAN_TAIL_BYTES,
    _ensure_usage_reported,
    _extract_token_count,
    _filter_params,
    _read_total_tokens,
)
from cray_infra.generate.metrics import get_metrics
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

import logging
from typing import Optional

logger = logging.getLogger(__name__)

openai_v1_router = APIRouter()


@openai_v1_router.get("/models")
async def list_models():
    """List available models - proxy to vLLM server."""
    session = get_global_session()
    config = get_config()
    async with session.get(config["vllm_api_url"] + "/v1/models") as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            return JSONResponse(
                content={"error": f"Failed to fetch models: {resp.status}"},
                status_code=resp.status,
            )


@openai_v1_router.post("/completions")
async def create_completions(request: CompletionRequest, raw_request: Request):
    """Create completions - proxy to vLLM server."""
    config = get_config()
    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _COMPLETION_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    logger.info("Received completions request: %s", params)
    return _proxy_streaming(
        upstream_url=config["vllm_api_url"] + "/v1/completions",
        params=params,
        endpoint_label="completions",
    )


@openai_v1_router.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    Create chat completions.

    Streaming requests (stream=True) keep the existing direct-to-vLLM
    SSE proxy — SSE keeps the connection alive natively, so it doesn't
    need the queue-and-heartbeat machinery. Non-streaming requests go
    through the new queue path: admission control → coalescer →
    SQLite InferenceWorkQueue → worker → result router → chunked-JSON
    heartbeat response. See docs/openai-chat-completions-queue.md §3.1.
    """
    if getattr(request, "stream", False):
        config = get_config()
        params = _filter_params(request.model_dump(mode="json", exclude_none=True), _CHAT_ALLOWED_KEYS)
        _ensure_usage_reported(params)
        logger.info("Received streaming chat completions request: %s", params)
        return _proxy_streaming(
            upstream_url=config["vllm_api_url"] + "/v1/chat/completions",
            params=params,
            endpoint_label="chat completions",
        )

    # Non-streaming path: queue-backed with admission control and
    # whitespace-heartbeat transport.
    from cray_infra.api.fastapi.chat_completions.handler import (
        chat_completions_via_queue,
    )

    return await chat_completions_via_queue(request)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _proxy_streaming(
    *,
    upstream_url: str,
    params: dict,
    endpoint_label: str,
) -> StreamingResponse:
    session = get_global_session()

    async def upstream():
        async with session.post(upstream_url, json=params) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(
                    "vLLM %s error (%s): %s", endpoint_label, resp.status, error_text,
                )
                yield (
                    f'data: {{"error": "Failed to create {endpoint_label}: {error_text}"}}\n\n'
                ).encode("utf-8")
                return
            async for chunk in resp.content.iter_any():
                yield chunk

    return StreamingResponse(
        content=_wrap_with_metrics(upstream()),
        media_type="text/event-stream",
    )


async def _wrap_with_metrics(source):
    """Pass chunks through verbatim while keeping a sliding-window buffer so
    we can extract the terminal `usage.total_tokens` for the metrics counter.

    Two sources of truth handled:
      - Streaming SSE: walk `data: {...}\\n\\n` events; last `usage` wins.
      - Non-streaming single-JSON body: parse the (possibly partial) buffer
        once at the end.

    The metrics counter is balanced even on errors / client disconnect: the
    finally block always fires record_completed_request to keep queue_depth
    consistent with record_new_request().
    """
    metrics = get_metrics()
    metrics.record_new_request()
    # Separate streaming-only counter so /v1/generate/metrics can show
    # in-flight requests on this path even though it never touches the
    # SQLiteAckQueue. The try/finally balance below guarantees this
    # counter doesn't drift the way Metrics.queue_depth can.
    metrics.record_streaming_start()

    buffer = bytearray()
    try:
        async for chunk in source:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            buffer.extend(chunk)
            if len(buffer) > _USAGE_SCAN_TAIL_BYTES:
                # Drop the head; the terminal usage event is guaranteed to
                # land in the last 64 KB.
                buffer = bytearray(buffer[-_USAGE_SCAN_TAIL_BYTES:])
            yield chunk
    finally:
        token_count = _extract_token_count(bytes(buffer))
        metrics.record_completed_request(
            token_count=token_count if token_count is not None else 0,
            flop_count=None,
        )
        metrics.record_streaming_end()


