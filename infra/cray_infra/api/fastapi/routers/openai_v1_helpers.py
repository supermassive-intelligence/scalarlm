"""Pure-logic helpers for the OpenAI proxy router.

Kept free of vllm / fastapi / aiohttp imports so the logic stays
unit-testable without the full inference stack. ``openai_v1_router``
re-exports these names.
"""

from __future__ import annotations

import json
from typing import Optional


# Tail-window for sniffing the upstream payload for `usage`. The terminal
# usage event in an OpenAI SSE stream is on the order of a few hundred
# bytes and always sits at the very end; 64 KB is more than enough
# headroom while bounding memory for very long completions.
_USAGE_SCAN_TAIL_BYTES = 64 * 1024

# Allowed keys on requests forwarded to vLLM. ``stream_options`` is
# included so callers can opt into usage reporting; we also force it on
# for streaming requests below so we can count tokens server-side.
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
    """For streaming requests, force vLLM to emit a final ``usage`` event
    so we can count tokens. OpenAI-compatible clients tolerate the extra
    field; the ScalarLM chat UI specifically reads it and surfaces
    tokens-per-message. Non-streaming responses always include usage in
    the final JSON body, so no opt-in is needed there.
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
    """Best-effort scan for ``usage.total_tokens`` in either an SSE
    stream tail or a single JSON response body. Returns None if not
    found.
    """
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
