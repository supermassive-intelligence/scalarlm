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
from cray_infra.api.fastapi.routers.openai_lora import ensure_adapter_loaded
from cray_infra.api.fastapi.routers.openai_prompts import count_prompts
from cray_infra.api.fastapi.routers.openai_queue import (
    QueueFull,
    get_openai_limiter,
)
from cray_infra.generate.flop_count import compute_flops_per_token
from cray_infra.generate.metrics import get_metrics
from cray_infra.one_server.vllm_registry import get_vllm_servings
from cray_infra.training.vllm_model_manager import get_vllm_model_manager
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Phase 8a — scatter-gather A/B. Set SCALARLM_SCATTER_THRESHOLD=N to make
# /v1/completions calls with a list-prompt of length >= N fan out into N
# single-prompt sub-requests dispatched in parallel via asyncio.gather.
# Unset / 0 = pre-Phase-8 monolithic dispatch (vLLM scatters internally
# via merge_async_iterators). The N=1000 profile showed merge_async_iterators
# at 13.4 % in-stack — this experiment tests whether moving the scatter
# one layer up (asyncio.gather + N create_completion calls) is faster.
_SCATTER_THRESHOLD = int(os.environ.get("SCALARLM_SCATTER_THRESHOLD", "0") or 0)

# Phase 8a v2 — bounded scatter. The plain scatter (above) creates all N
# tasks instantly. scalarlm's worker instead pulls batch_size requests,
# fans them out, waits for them to drain, pulls the next batch. The
# hypothesis is that vLLM's scheduler does O(pending_requests) work per
# iteration and gets slower when 1000 are pending vs 16 (max_num_seqs).
# SCALARLM_SCATTER_MAX_INFLIGHT bounds the number of sub-requests in
# flight at once via a Semaphore — set to max_num_seqs to mimic scalarlm.
# 0 / unset = no in-flight cap (pure asyncio.gather, the v1 behaviour).
_SCATTER_MAX_INFLIGHT = int(os.environ.get("SCALARLM_SCATTER_MAX_INFLIGHT", "0") or 0)

openai_v1_router = APIRouter()

# Tail-window for sniffing the upstream payload for `usage`. The terminal
# usage event in an OpenAI SSE stream is on the order of a few hundred bytes
# and always sits at the very end; 64 KB is more than enough headroom while
# bounding memory for very long completions.
_USAGE_SCAN_TAIL_BYTES = 64 * 1024

# Allowed keys on requests forwarded to vLLM. `stream_options` is included so
# that callers can opt into usage reporting; we also force it on for streaming
# requests below so we can count tokens server-side.
_COMPLETION_ALLOWED_KEYS = (
    "model",
    "temperature",
    "prompt",
    "max_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "response_format",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
)

_CHAT_ALLOWED_KEYS = (
    "model",
    "temperature",
    "messages",
    "max_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "response_format",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
)


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
    """Create completions - proxy to vLLM server.

    Accepts the OpenAI-spec ``prompt`` shape: ``str``, ``list[str]``,
    ``list[int]``, or ``list[list[int]]``. Array prompts pass through to
    vLLM unchanged and come back as a ``choices`` array of matching length.
    """
    config = get_config()
    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _COMPLETION_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    await _ensure_model_available(params.get("model"), config)
    slot = await _acquire_queue_slot(config["model"])
    logger.info(
        "completions request: model=%s prompts=%d stream=%s",
        params.get("model"),
        count_prompts(params.get("prompt")),
        bool(params.get("stream")),
    )
    return await _dispatch(
        endpoint="completions",
        request=request,
        raw_request=raw_request,
        params=params,
        config=config,
        queue_slot=slot,
    )


@openai_v1_router.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Create chat completions - proxy to vLLM server."""
    config = get_config()
    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _CHAT_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    await _ensure_model_available(params.get("model"), config)
    slot = await _acquire_queue_slot(config["model"])
    logger.info("Received chat completions request: %s", params)
    return await _dispatch(
        endpoint="chat",
        request=request,
        raw_request=raw_request,
        params=params,
        config=config,
        queue_slot=slot,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _filter_params(raw: dict, allowed: tuple) -> dict:
    return {k: v for k, v in raw.items() if v is not None and k in allowed}


async def _ensure_model_available(model_name: Optional[str], config: dict) -> None:
    """Load a requested LoRA/tokenformer adapter into vLLM on first use.

    Wraps ``ensure_adapter_loaded`` with the dependencies the handler needs
    to supply. Any failure is surfaced as a 503 so the caller sees a clean
    HTTP error instead of a hung stream.
    """
    try:
        await ensure_adapter_loaded(
            session=get_global_session(),
            vllm_api_url=config["vllm_api_url"],
            model_name=model_name,
            model_manager=get_vllm_model_manager(),
            training_job_directory=config["training_job_directory"],
            base_model=config["model"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


async def _acquire_queue_slot(model: str):
    """Reserve a concurrency slot for this request. Translates ``QueueFull``
    into a 503 + ``Retry-After`` so OpenAI clients follow their standard
    retry path instead of hanging on a saturated vLLM instance.
    """
    limiter = await get_openai_limiter()
    try:
        return await limiter.acquire(model=model)
    except QueueFull as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        )


def _ensure_usage_reported(params: dict) -> None:
    """For streaming requests, force vLLM to emit a final `usage` event so we
    can count tokens. OpenAI-compatible clients tolerate the extra field; the
    ScalarLM chat UI specifically reads it and surfaces tokens-per-message.
    Non-streaming responses always include usage in the final JSON body, so
    no opt-in is needed there.
    """
    if not params.get("stream"):
        return
    opts = dict(params.get("stream_options") or {})
    opts.setdefault("include_usage", True)
    params["stream_options"] = opts


async def _dispatch(
    *,
    endpoint: str,  # "completions" or "chat"
    request,
    raw_request: Request,
    params: dict,
    config: dict,
    queue_slot,
):
    """Route a prepared request to the in-process or HTTP-proxy backend.

    Phase 6. The in-process path is preferred when:
      (a) vLLM has registered its servings with `vllm_registry` (i.e. this
          process is running the `one_server` bundle, not a remote vLLM),
      (b) ``openai_inprocess_enabled`` is true in config (the default).

    Falls back to the HTTP proxy otherwise — identical observable behaviour,
    the only difference is the transport to vLLM.
    """
    servings = get_vllm_servings()
    if servings is not None and config.get("openai_inprocess_enabled", True):
        # Phase 8a A/B: if scatter threshold is set and the request is an
        # array-prompt /v1/completions call past the threshold, fan out into
        # single-prompt sub-requests rather than letting vLLM's internal
        # merge_async_iterators do the scatter. See _scatter_gather_completions.
        if (
            endpoint == "completions"
            and _SCATTER_THRESHOLD > 0
            and isinstance(getattr(request, "prompt", None), list)
            and len(request.prompt) >= _SCATTER_THRESHOLD
            and all(isinstance(p, str) for p in request.prompt)
        ):
            return await _scatter_gather_completions(
                request=request,
                raw_request=raw_request,
                servings=servings,
                base_model_name=config["model"],
                queue_slot=queue_slot,
            )
        return await _call_inprocess(
            endpoint=endpoint,
            request=request,
            raw_request=raw_request,
            servings=servings,
            base_model_name=config["model"],
            queue_slot=queue_slot,
        )
    path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    label = "chat completions" if endpoint == "chat" else "completions"
    return _proxy_streaming(
        upstream_url=config["vllm_api_url"] + path,
        params=params,
        endpoint_label=label,
        base_model_name=config["model"],
        queue_slot=queue_slot,
    )


async def _call_inprocess(
    *,
    endpoint: str,
    request,
    raw_request: Request,
    servings,
    base_model_name: Optional[str],
    queue_slot,
):
    """Direct Python-API call into vLLM's OpenAI serving classes.

    Skips the localhost-HTTP hop to vLLM's FastAPI on ``vllm_api_url``.
    vLLM's ``create_completion`` / ``create_chat_completion`` return one of:
    an async-iterator for SSE streaming, a Pydantic response model for
    non-streaming, or an ``ErrorResponse`` for failures. Each is wrapped
    so the Phase 3d queue slot gets released exactly once and the FLOP/
    token metrics fire the same way they did on the HTTP path.
    """
    if endpoint == "chat":
        result = await servings.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
    else:
        result = await servings.openai_serving_completion.create_completion(
            request, raw_request
        )

    # ErrorResponse from vLLM: release slot, forward status + body.
    if hasattr(result, "error") and hasattr(result.error, "code"):
        status_code = getattr(result.error, "code", 500)
        body = result.model_dump() if hasattr(result, "model_dump") else {"error": str(result)}
        if queue_slot is not None:
            await queue_slot.release()
        return JSONResponse(content=body, status_code=status_code)

    # Streaming — async iterator of str/bytes chunks
    if hasattr(result, "__aiter__"):
        return StreamingResponse(
            content=_wrap_with_metrics(
                result, base_model_name=base_model_name, queue_slot=queue_slot,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming pydantic response. Record metrics synchronously from the
    # usage block so dashboards stay in sync with the streaming path.
    body = result.model_dump() if hasattr(result, "model_dump") else result
    token_count = 0
    if isinstance(body, dict):
        usage = body.get("usage") or {}
        token_count = usage.get("total_tokens") or 0
    flop_count: Optional[int] = None
    if token_count and base_model_name:
        per_token = compute_flops_per_token(base_model_name)
        if per_token:
            flop_count = per_token * token_count
    metrics = get_metrics()
    metrics.record_new_request()
    metrics.record_completed_request(token_count=token_count, flop_count=flop_count)
    if queue_slot is not None:
        try:
            await queue_slot.release()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to release OpenAI proxy queue slot")
    return JSONResponse(content=body)


async def _scatter_gather_completions(
    *,
    request,            # CompletionRequest with list[str] prompt
    raw_request: Request,
    servings,
    base_model_name: Optional[str],
    queue_slot,
):
    """Phase 8a: dispatch each prompt as its own create_completion call,
    gather sub-responses, return a merged CompletionResponse.

    Sub-requests share the parent's queue_slot — we never acquire N slots
    for one logical call, otherwise the OpenAIConcurrencyLimiter (default
    semaphore=16) would be exhausted by a single 1 000-prompt request.
    Per Gemini's correctness note in the amended plan.

    Streaming and non-string prompts (e.g. token-id arrays) are not
    handled — _dispatch only routes string-list prompts here.
    """
    prompts = request.prompt
    sub_requests = [request.model_copy(update={"prompt": p}) for p in prompts]

    if _SCATTER_MAX_INFLIGHT > 0:
        # Bounded fan-out — only K sub-requests in flight at once. Mimics
        # scalarlm's queue-driven batch_size pull pattern. Tests whether
        # vLLM scheduler scales poorly with many pending requests.
        sem = asyncio.Semaphore(_SCATTER_MAX_INFLIGHT)
        async def _bounded(r):
            async with sem:
                return await servings.openai_serving_completion.create_completion(r, raw_request)
        sub_results = await asyncio.gather(*[_bounded(r) for r in sub_requests])
    else:
        sub_results = await asyncio.gather(*[
            servings.openai_serving_completion.create_completion(r, raw_request)
            for r in sub_requests
        ])

    # Per-sub error short-circuits the whole call (matches array-prompt
    # behaviour today — vLLM aborts the batch if any one prompt errors).
    for r in sub_results:
        if hasattr(r, "error") and hasattr(r.error, "code"):
            status_code = getattr(r.error, "code", 500)
            body = r.model_dump() if hasattr(r, "model_dump") else {"error": str(r)}
            if queue_slot is not None:
                await queue_slot.release()
            return JSONResponse(content=body, status_code=status_code)

    # Merge: concatenate choices arrays with index renumbered to original
    # array position; sum usage; first sub's id/created/model/system_fp.
    merged = _merge_completion_responses(sub_results)

    # Run the same metric-recording the non-streaming inprocess path runs,
    # ONCE for the whole logical request — not N times. Otherwise per-request
    # Prometheus labels and the FLOP counter would each fire N times.
    token_count = 0
    if isinstance(merged, dict):
        token_count = (merged.get("usage") or {}).get("total_tokens") or 0
    flop_count: Optional[int] = None
    if token_count and base_model_name:
        per_token = compute_flops_per_token(base_model_name)
        if per_token:
            flop_count = per_token * token_count
    metrics = get_metrics()
    metrics.record_new_request()
    metrics.record_completed_request(token_count=token_count, flop_count=flop_count)

    if queue_slot is not None:
        try:
            await queue_slot.release()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to release OpenAI proxy queue slot (scatter)")
    return JSONResponse(content=merged)


def _merge_completion_responses(sub_results) -> dict:
    """Merge N single-prompt CompletionResponse objects into one.

    Choices: concatenate, renumber `index` by original array position so
    the response shape matches what an unscattered call would have returned.
    Usage: sum prompt/completion/total tokens. id/created/model/system_fp:
    take the first sub's value (one logical request → one logical id).
    """
    merged_choices = []
    prompt_tokens = completion_tokens = total_tokens = 0

    for original_idx, sub in enumerate(sub_results):
        body = sub.model_dump() if hasattr(sub, "model_dump") else dict(sub)
        for choice in body.get("choices", []):
            choice = dict(choice)
            choice["index"] = original_idx
            merged_choices.append(choice)
        usage = body.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0) or 0
        completion_tokens += usage.get("completion_tokens", 0) or 0
        total_tokens += usage.get("total_tokens", 0) or 0

    first = sub_results[0]
    first_body = first.model_dump() if hasattr(first, "model_dump") else dict(first)
    return {
        "id": first_body.get("id"),
        "object": first_body.get("object", "text_completion"),
        "created": first_body.get("created"),
        "model": first_body.get("model"),
        "system_fingerprint": first_body.get("system_fingerprint"),
        "choices": merged_choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _proxy_streaming(
    *,
    upstream_url: str,
    params: dict,
    endpoint_label: str,
    base_model_name: Optional[str] = None,
    queue_slot=None,
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
        content=_wrap_with_metrics(
            upstream(), base_model_name=base_model_name, queue_slot=queue_slot
        ),
        media_type="text/event-stream",
    )


async def _wrap_with_metrics(
    source,
    base_model_name: Optional[str] = None,
    queue_slot=None,
):
    """Pass chunks through verbatim while keeping a sliding-window buffer so
    we can extract the terminal `usage.total_tokens` for the metrics counter.

    Two sources of truth handled:
      - Streaming SSE: walk `data: {...}\\n\\n` events; last `usage` wins.
      - Non-streaming single-JSON body: parse the (possibly partial) buffer
        once at the end.

    ``base_model_name`` drives the FLOP estimate. When the model is known we
    multiply token_count by per-token FLOPs (cached by
    ``compute_flops_per_token``) so the observability counter matches Path
    A's worker-side reporting.

    ``queue_slot`` (when provided) is released in the finally so the
    concurrency limiter frees the slot whether the stream completed
    normally, errored, or the client disconnected.

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
        token_count = _extract_token_count(bytes(buffer)) or 0
        flop_count: Optional[int] = None
        if token_count and base_model_name:
            per_token = compute_flops_per_token(base_model_name)
            if per_token:
                flop_count = per_token * token_count
        metrics.record_completed_request(
            token_count=token_count,
            flop_count=flop_count,
        )
        if queue_slot is not None:
            try:
                await queue_slot.release()
            except Exception:  # noqa: BLE001 — releasing must never mask the real error
                logger.exception("Failed to release OpenAI proxy queue slot")


def _extract_token_count(payload: bytes) -> Optional[int]:
    """Best-effort scan for `usage.total_tokens` in either an SSE stream tail
    or a single JSON response body. Returns None if not found."""
    if not payload:
        return None
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception:
        return None

    # SSE path — look for the last `data: {...}` event with a usage field.
    if "data:" in text:
        last: Optional[int] = None
        for event in text.split("\n\n"):
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].lstrip()
                if not body or body == "[DONE]":
                    continue
                tokens = _read_total_tokens(body)
                if tokens is not None:
                    last = tokens
        if last is not None:
            return last

    # Non-streaming path — try parsing the whole tail as JSON.
    return _read_total_tokens(text)


def _read_total_tokens(json_text: str) -> Optional[int]:
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    return int(total) if isinstance(total, (int, float)) else None
