"""Pure-logic helpers for the Phase 30 response cache.

Kept free of vllm / fastapi / aiohttp imports so the logic stays unit-
testable without the full inference stack. ``openai_v1_router``
re-exports these names.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# --- Phase 30: response cache ---------------------------------------------

_OPENAI_CACHE_ENABLED = bool(int(os.environ.get("SCALARLM_OPENAI_CACHE", "0") or 0))
_OPENAI_CACHE_KEYS = (
    "model", "prompt", "messages", "max_tokens", "temperature",
    "top_p", "stop", "n", "tools", "tool_choice",
)


def _cache_dir(config: dict) -> str:
    base = config.get("upload_base_path") or "/app/cray/inference_requests"
    path = os.path.join(base, "openai_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_key(params: dict) -> str:
    payload = {k: params.get(k) for k in _OPENAI_CACHE_KEYS if k in params}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def _cache_store(params: dict, body: dict, config: dict) -> None:
    if not _OPENAI_CACHE_ENABLED or params.get("stream"):
        return
    if not isinstance(body, dict) or "choices" not in body:
        return
    path = os.path.join(_cache_dir(config), _cache_key(params) + ".json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(body, fh)
        os.replace(tmp, path)
    except OSError:
        logger.exception("openai cache store failed at %s", path)
