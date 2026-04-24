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
    _OPENAI_CACHE_ENABLED,
    _OPENAI_CACHE_KEYS,
    _QUEUE_ROUTE_THRESHOLD,
    _USAGE_SCAN_TAIL_BYTES,
    _cache_dir,
    _cache_key,
    _cache_key_from_request,
    _cache_lookup,
    _cache_lookup_by_key,
    _cache_store,
    _ensure_usage_reported,
    _extract_token_count,
    _filter_params,
    _read_total_tokens,
    _should_route_via_queue_fast,
)
from cray_infra.generate.metrics import get_metrics
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def _route_via_scalarlm_queue(request, config: dict) -> JSONResponse:
    """Dispatch a bulk array /v1/completions request through the existing
    /v1/generate queue worker. The worker calls create_completion on a
    controlled `get_batch_size()`-sized slice, `asyncio.gather`s the
    sub-calls, and posts results back — keeping the engine saturated
    without fanning N AsyncStream futures onto the APIServer event loop.

    The worker's queue layer caches by content hash too, so repeat calls
    short-circuit at that layer as well (in addition to openai's own
    cache above). We deliberately do not double-store in the openai
    cache here — the queue's `{hash}_response.json` already holds it.

    Limitations (deliberate, documented in enhance-openai-api.md):
    - no streaming (queue is batch-oriented)
    - `usage.*_tokens` returns 0 (GenerateResponse.Result doesn't carry
      token_count today — follow-up item to plumb through)
    - no logprobs
    """
    # Imported lazily so modules that import this router at startup don't
    # force-import the /v1/generate pipeline.
    from cray_infra.api.fastapi.generate.generate import generate as _scalarlm_generate
    from cray_infra.api.fastapi.routers.request_types.generate_request import (
        GenerateRequest,
    )

    metrics = get_metrics()
    metrics.record_new_request()

    gen_req = GenerateRequest(
        model=request.model,
        prompts=list(request.prompt),
        max_tokens=request.max_tokens or 16,
        temperature=request.temperature or 0.0,
        tools=getattr(request, "tools", None),
        tool_choice=getattr(request, "tool_choice", None),
    )
    gen_resp = await _scalarlm_generate(gen_req)

    # Per-item errors bubble up as a 500 with an OpenAI-shaped body.
    # scalarlm's /v1/generate supports partial success; /v1/completions
    # does not (the OpenAI `finish_reason` enum has no "error" value,
    # so clients would fail validation on a mixed-success batch).
    # Fail the whole request instead.
    failed = [(i, r.error) for i, r in enumerate(gen_resp.results) if r.error]
    if failed:
        first_i, first_err = failed[0]
        metrics.record_completed_request(token_count=0, flop_count=None)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": (
                        f"inference failed for prompt index {first_i}: "
                        f"{first_err}" + (
                            f" (and {len(failed) - 1} other prompt(s))"
                            if len(failed) > 1 else ""
                        )
                    ),
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    import time as _t
    choices = [
        {
            "index": i,
            "text": r.response or "",
            "finish_reason": "stop",
            "logprobs": None,
        }
        for i, r in enumerate(gen_resp.results)
    ]
    group_id = (
        gen_resp.results[0].request_id.split("_")[0][:12]
        if gen_resp.results else "queue"
    )
    body = {
        "id": f"cmpl-queue-{group_id}",
        "object": "text_completion",
        "created": int(_t.time()),
        "model": request.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    metrics.record_completed_request(token_count=0, flop_count=None)
    return JSONResponse(content=body)

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

    # Bulk-route fast path (Phase 31). Fires BEFORE model_dump so bulk
    # requests don't pay the O(N) pydantic walk.
    if _should_route_via_queue_fast(request):
        if _OPENAI_CACHE_ENABLED:
            hit = _cache_lookup_by_key(_cache_key_from_request(request), config)
            if hit is not None:
                return JSONResponse(content=hit)
        logger.info(
            "completions via queue: model=%s prompts=%d",
            request.model, len(request.prompt),
        )
        return await _route_via_scalarlm_queue(request, config)

    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _COMPLETION_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    cached = _cache_lookup(params, config)
    if cached is not None:
        return JSONResponse(content=cached)
    logger.info("Received completions request: %s", params)
    if _OPENAI_CACHE_ENABLED and not params.get("stream"):
        return await _proxy_nonstreaming_cached(
            upstream_url=config["vllm_api_url"] + "/v1/completions",
            params=params,
            endpoint_label="completions",
            config=config,
        )
    return _proxy_streaming(
        upstream_url=config["vllm_api_url"] + "/v1/completions",
        params=params,
        endpoint_label="completions",
    )


@openai_v1_router.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Create chat completions - proxy to vLLM server."""
    config = get_config()
    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _CHAT_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    cached = _cache_lookup(params, config)
    if cached is not None:
        return JSONResponse(content=cached)
    logger.info("Received chat completions request: %s", params)
    if _OPENAI_CACHE_ENABLED and not params.get("stream"):
        return await _proxy_nonstreaming_cached(
            upstream_url=config["vllm_api_url"] + "/v1/chat/completions",
            params=params,
            endpoint_label="chat completions",
            config=config,
        )
    return _proxy_streaming(
        upstream_url=config["vllm_api_url"] + "/v1/chat/completions",
        params=params,
        endpoint_label="chat completions",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _proxy_nonstreaming_cached(
    *,
    upstream_url: str,
    params: dict,
    endpoint_label: str,
    config: dict,
) -> JSONResponse:
    """Collect the upstream body in full, store it in the openai-cache,
    then return it to the caller. Used for non-streaming requests when
    `SCALARLM_OPENAI_CACHE=1`; streaming requests still go through
    `_proxy_streaming` since the cache is batch-granular.
    """
    session = get_global_session()
    metrics = get_metrics()
    metrics.record_new_request()
    try:
        async with session.post(upstream_url, json=params) as resp:
            text = await resp.text()
            if resp.status != 200:
                logger.error(
                    "vLLM %s error (%s): %s", endpoint_label, resp.status, text,
                )
                metrics.record_completed_request(token_count=0, flop_count=None)
                return JSONResponse(
                    content={"error": f"Failed to create {endpoint_label}: {text}"},
                    status_code=resp.status,
                )
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                metrics.record_completed_request(token_count=0, flop_count=None)
                return JSONResponse(
                    content={"error": f"vLLM returned non-JSON body for {endpoint_label}"},
                    status_code=502,
                )
    except Exception:  # noqa: BLE001
        metrics.record_completed_request(token_count=0, flop_count=None)
        raise

    token_count = 0
    usage = body.get("usage") if isinstance(body, dict) else None
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)):
            token_count = int(total)
    metrics.record_completed_request(token_count=token_count, flop_count=None)
    _cache_store(params, body, config)
    return JSONResponse(content=body)


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


# _extract_token_count, _read_total_tokens, _filter_params,
# _ensure_usage_reported are re-imported from openai_v1_helpers above.
