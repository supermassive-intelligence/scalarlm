"""LoRA-adapter routing for the OpenAI-compatible proxy.

Path A (``/v1/generate`` + worker) loads adapters in-process via vLLM's
Python API. Path B (``/v1/chat/completions``, ``/v1/completions``) runs in
the API process and must therefore call vLLM's HTTP endpoint
``POST /v1/load_lora_adapter`` before forwarding any request that targets
an unloaded adapter.

vLLM itself is idempotent-by-name and serializes concurrent loads of the
same adapter (``lora_resolver_lock`` in vllm-fork's serving.py), so the
proxy-side state here exists only to skip the HTTP round-trip once an
adapter is known to be loaded. The ``asyncio.Event`` per name coalesces a
thundering herd of first-use callers into a single upstream load.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class _AdapterLoadState:
    """In-process cache of adapter load state, keyed by adapter name.

    A name whose Event is ``set()`` is loaded upstream. A name whose Event
    exists but isn't set has a load in flight — concurrent callers await
    that Event. A name absent from the map has never been attempted.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_event(self, name: str) -> tuple[asyncio.Event, bool]:
        """Return (event, is_first_caller). The first caller is responsible
        for performing the upstream load; others should ``await event.wait()``.
        """
        async with self._lock:
            existing = self._events.get(name)
            if existing is not None:
                return existing, False
            event = asyncio.Event()
            self._events[name] = event
            return event, True

    async def mark_failed(self, name: str) -> None:
        """Remove the failed entry so a subsequent request can retry."""
        async with self._lock:
            event = self._events.pop(name, None)
        if event is not None:
            # Wake any waiters so they can raise / retry themselves instead
            # of hanging forever.
            event.set()

    def clear(self) -> None:
        """Test hook — production code should not call this."""
        self._events.clear()


_state = _AdapterLoadState()


async def ensure_adapter_loaded(
    session,
    vllm_api_url: str,
    model_name: Optional[str],
    model_manager,
    training_job_directory: str,
    base_model: str,
) -> None:
    """Ensure ``model_name`` is loaded in the upstream vLLM server.

    No-op when:
      - ``model_name`` is falsy,
      - ``model_manager.find_model`` returns ``None`` (unknown — let vLLM
        respond with its own 400/404 and surface the error cleanly),
      - the resolved name matches ``base_model`` (no adapter needed),
      - the adapter is already known-loaded in this process.

    Raises ``RuntimeError`` when the upstream load fails, with the status
    code and body so callers can translate to a client-facing 5xx.
    """
    if not model_name:
        return

    resolved = model_manager.find_model(model_name)
    if resolved is None:
        return

    if resolved == base_model:
        return

    event, is_first = await _state.get_or_create_event(resolved)
    if not is_first:
        await event.wait()
        return

    lora_path = os.path.join(training_job_directory, resolved)
    payload = {"lora_name": resolved, "lora_path": lora_path}
    load_url = vllm_api_url.rstrip("/") + "/v1/load_lora_adapter"

    try:
        async with session.post(load_url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(
                    "vLLM load_lora_adapter failed for %s (status=%s): %s",
                    resolved,
                    resp.status,
                    body,
                )
                await _state.mark_failed(resolved)
                raise RuntimeError(
                    f"Failed to load adapter {resolved}: status={resp.status} body={body}"
                )
    except Exception:
        # Also unmark on transport errors so retries aren't permanently blocked.
        await _state.mark_failed(resolved)
        raise

    logger.info("Loaded adapter '%s' on upstream vLLM", resolved)
    event.set()


def _reset_state_for_tests() -> None:
    """Test hook — production code should not call this."""
    _state.clear()
