"""Pure-logic helpers for the Phase 30 cache and Phase 31 queue-route
fast-path.

Kept free of vllm / fastapi / aiohttp imports so the logic stays unit-
testable without the full inference stack. ``openai_v1_router``
re-exports these names.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_env_bool(name: str, default: bool = False) -> bool:
    """Robust boolean parse for scalarlm env flags. Accepts common forms
    (``1`` / ``true`` / ``yes`` / ``on`` — case-insensitive). Anything
    else, or unset, returns ``default``. Prevents the ``bool(int(...))``
    pattern from crashing module import when a user sets
    ``SCALARLM_OPENAI_CACHE=true``.
    """
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# --- Phase 30: response cache ---------------------------------------------

_OPENAI_CACHE_ENABLED = _parse_env_bool("SCALARLM_OPENAI_CACHE")
_OPENAI_CACHE_KEYS = (
    "model", "prompt", "messages", "max_tokens", "temperature",
    "top_p", "stop", "n", "tools", "tool_choice",
)

# --- Phase 31: bulk queue-route fast-path ---------------------------------

_QUEUE_ROUTE_THRESHOLD = int(os.environ.get("SCALARLM_QUEUE_ROUTE_THRESHOLD", "0") or 0)

# Fields that the queue-route translation does NOT forward to the worker.
# If any of these is set to a non-default value on a bulk request, keep
# the request on the direct proxy path instead of silently dropping them.
# Map of field name → default value that we treat as "unset".
_QUEUE_ROUTE_UNSUPPORTED = {
    "top_p": None,
    "stop": None,
    "seed": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "response_format": None,
    "logprobs": None,
    "logit_bias": None,
    "suffix": None,
    "echo": False,
    "best_of": None,
}


def _cache_dir(config: dict) -> str:
    base = config.get("upload_base_path") or "/app/cray/inference_requests"
    path = os.path.join(base, "openai_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_key(params: dict) -> str:
    payload = {k: params.get(k) for k in _OPENAI_CACHE_KEYS if k in params}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_key_from_request(request) -> str:
    """Build the cache key without walking the full pydantic object via
    ``model_dump``. On the queue-route fast-path this saves ~20-30 ms at
    N=1000 (pydantic re-encodes every element of the prompt list)."""
    payload = {}
    for k in _OPENAI_CACHE_KEYS:
        v = getattr(request, k, None)
        if v is None:
            continue
        payload[k] = v
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_lookup(params: dict, config: dict) -> Optional[dict]:
    if not _OPENAI_CACHE_ENABLED or params.get("stream"):
        return None
    path = os.path.join(_cache_dir(config), _cache_key(params) + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_lookup_by_key(key: str, config: dict) -> Optional[dict]:
    path = os.path.join(_cache_dir(config), key + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_store(params: dict, body: dict, config: dict) -> None:
    if not _OPENAI_CACHE_ENABLED or params.get("stream"):
        return
    if not isinstance(body, dict) or "choices" not in body:
        return
    key = _cache_key(params)
    cache_dir = _cache_dir(config)
    path = os.path.join(cache_dir, key + ".json")
    # Unique tmp path so two concurrent writers of the same key don't
    # clobber each other's tmp file and leave a half-written .json on
    # the rename. mkstemp creates the file atomically with a unique
    # suffix; we os.fdopen its fd so we never open() a path twice.
    try:
        fd, tmp = tempfile.mkstemp(
            prefix=key + ".", suffix=".json.tmp", dir=cache_dir,
        )
    except OSError:
        logger.exception("openai cache store mkstemp failed under %s", cache_dir)
        return
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(body, fh)
        os.replace(tmp, path)
    except OSError:
        logger.exception("openai cache store failed at %s", path)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _should_route_via_queue_fast(request) -> bool:
    if _QUEUE_ROUTE_THRESHOLD <= 0:
        return False
    if getattr(request, "stream", False):
        return False
    prompt = getattr(request, "prompt", None)
    if not isinstance(prompt, list) or len(prompt) < _QUEUE_ROUTE_THRESHOLD:
        return False
    if not all(isinstance(p, str) for p in prompt):
        return False
    # The queue-route translation only forwards model / prompts /
    # max_tokens / temperature / tools / tool_choice to the worker.
    # If the caller set any other OpenAI completion param, keep the
    # request on the direct proxy so the param actually takes effect.
    for field, default in _QUEUE_ROUTE_UNSUPPORTED.items():
        if getattr(request, field, default) != default:
            return False
    # n defaults to 1 for /v1/completions; reject n > 1 because the
    # queue worker only returns one completion per prompt.
    n = getattr(request, "n", 1)
    if n is not None and n != 1:
        return False
    return True


# --- Router-level pure helpers (kept here for unit-testability) -----------

# Tail-window for sniffing the upstream SSE payload for `usage`. The
# terminal usage event is on the order of a few hundred bytes and always
# sits at the very end; 64 KB is more than enough headroom while bounding
# memory for very long completions.
_USAGE_SCAN_TAIL_BYTES = 64 * 1024

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


def _filter_params(raw: dict, allowed: tuple) -> dict:
    return {k: v for k, v in raw.items() if v is not None and k in allowed}


def _ensure_usage_reported(params: dict) -> None:
    """For streaming requests, force vLLM to emit a final ``usage`` event so we
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


def _extract_token_count(payload: bytes) -> Optional[int]:
    """Best-effort scan for ``usage.total_tokens`` in either an SSE stream tail
    or a single JSON response body. Returns None if not found."""
    if not payload:
        return None
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception:
        return None

    # SSE path — look for the last event that contains a `usage` field.
    # Per the SSE spec:
    #   - events are separated by an empty line (\n\n, \r\n\r\n, or \r\r)
    #   - a single event can contain multiple `data:` lines whose values
    #     are concatenated with "\n" to form one decoded value
    # Normalize line endings to LF up-front so split("\n\n") handles all
    # separator forms; parse per-event (not per-line) so multi-line JSON
    # still resolves to a single object.
    if "data:" in text:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        last: Optional[int] = None
        for event in normalized.split("\n\n"):
            data_lines = [
                line[5:].lstrip() for line in event.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            body = "\n".join(data_lines)
            if not body or body == "[DONE]":
                continue
            tokens = _read_total_tokens(body)
            if tokens is not None:
                last = tokens
        if last is not None:
            return last

    # Non-streaming path — try parsing the whole tail as JSON.
    return _read_total_tokens(text)
