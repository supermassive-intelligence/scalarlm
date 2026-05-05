"""
Resolve `max_model_length` from vLLM's runtime config rather than from
a stale operator-configured value.

The cray-config.yaml's `max_model_length` is easy to misconfigure
(default is 256 in default_config.py — way smaller than any real
deployment), and an operator-set value also drifts the moment vLLM
is restarted with a different `--max-model-len`. vLLM itself knows
the right number for each loaded model (it's in the `/v1/models`
response shape: `{data: [{id: "...", max_model_len: 65536, ...}]}`),
so we ask it directly.

Per-model cache: different LoRA adapters share the base model's
context window, but different base models can differ. Keying by
model name keeps multi-model serving honest.

Failure handling: the resolver is best-effort. If vLLM hasn't booted,
the HTTP call fails, the response shape isn't what we expect, or the
model isn't in the list, we fall back to `config["max_model_length"]`
with a logged warning. The chat handler treats `<= 0` as "no cap",
which is the safe behavior — better to admit the request and let
the worker-side 400 trap catch it than to spuriously reject every
request when vLLM is briefly unreachable.
"""

import asyncio
import logging
from typing import Any

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


_cache: dict[str, int] = {}
_cache_lock = asyncio.Lock()


async def resolve_max_model_length(model: str) -> int:
    """
    Return the max sequence length vLLM reports for `model`. Cached
    per model so the handler hot path pays the HTTP round-trip at
    most once per model per process. Returns 0 (handler treats as
    "no cap") when vLLM is unreachable or the model isn't listed.
    """
    cached = _cache.get(model)
    if cached is not None:
        return cached

    async with _cache_lock:
        # Double-check under the lock — concurrent first-callers
        # would otherwise both pay the HTTP round-trip.
        cached = _cache.get(model)
        if cached is not None:
            return cached

        resolved = await _query_vllm_for_max_model_len(model)
        if resolved is None:
            resolved = _fallback_from_config()
        _cache[model] = resolved
        return resolved


def reset_cache_for_tests() -> None:
    """Tests can reuse the same model name across cases."""
    _cache.clear()


async def _query_vllm_for_max_model_len(model: str) -> int | None:
    """Hit vLLM's `/v1/models`, find the entry, return its
    `max_model_len`. None on any failure — caller falls back."""
    config = get_config()
    url = config["vllm_api_url"] + "/v1/models"
    try:
        session = get_global_session()
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning(
                    "resolve_max_model_length: vLLM /v1/models returned %s; "
                    "falling back to config",
                    resp.status,
                )
                return None
            payload = await resp.json()
    except Exception as exc:
        logger.warning(
            "resolve_max_model_length: HTTP error querying %s: %s; "
            "falling back to config",
            url,
            exc,
        )
        return None

    return _extract_max_model_len(payload, model)


def _extract_max_model_len(payload: Any, model: str) -> int | None:
    """Walk vLLM's models-list response and pull `max_model_len` for
    the requested id. Tolerates shape drift: missing keys, alternate
    field names (`max_position_embeddings`), non-int values."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") != model:
            continue
        # Try the canonical field first, then the underlying HF name.
        for field in ("max_model_len", "max_position_embeddings"):
            value = entry.get(field)
            if isinstance(value, int) and value > 0:
                return value
            # Some vLLM versions stringify; coerce if we can.
            try:
                coerced = int(value)
                if coerced > 0:
                    return coerced
            except (TypeError, ValueError):
                continue
        return None
    return None


def _fallback_from_config() -> int:
    """Last-resort value when vLLM is unavailable. The chat handler
    treats <= 0 as "no cap" so an unset/zero config disables the
    pre-admission check rather than rejecting every request."""
    try:
        return int(get_config().get("max_model_length", 0))
    except (TypeError, ValueError):
        return 0
