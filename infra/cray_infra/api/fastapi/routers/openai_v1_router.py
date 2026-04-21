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
from cray_infra.generate.flop_count import compute_flops_per_token
from cray_infra.generate.metrics import get_metrics
from cray_infra.training.vllm_model_manager import get_vllm_model_manager
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
    """Create completions - proxy to vLLM server."""
    config = get_config()
    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _COMPLETION_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    await _ensure_model_available(params.get("model"), config)
    logger.info("Received completions request: %s", params)
    return _proxy_streaming(
        upstream_url=config["vllm_api_url"] + "/v1/completions",
        params=params,
        endpoint_label="completions",
        base_model_name=config["model"],
    )


@openai_v1_router.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Create chat completions - proxy to vLLM server."""
    config = get_config()
    params = _filter_params(request.model_dump(mode="json", exclude_none=True), _CHAT_ALLOWED_KEYS)
    _ensure_usage_reported(params)
    await _ensure_model_available(params.get("model"), config)
    logger.info("Received chat completions request: %s", params)
    return _proxy_streaming(
        upstream_url=config["vllm_api_url"] + "/v1/chat/completions",
        params=params,
        endpoint_label="chat completions",
        base_model_name=config["model"],
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


def _proxy_streaming(
    *,
    upstream_url: str,
    params: dict,
    endpoint_label: str,
    base_model_name: Optional[str] = None,
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
        content=_wrap_with_metrics(upstream(), base_model_name=base_model_name),
        media_type="text/event-stream",
    )


async def _wrap_with_metrics(source, base_model_name: Optional[str] = None):
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
