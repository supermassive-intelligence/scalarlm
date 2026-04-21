"""Bounded-concurrency limiter for the OpenAI-compatible proxy.

Phase 3d of the OpenAI-API enhancement plan. Path A has a SQLite-backed
work queue with depth/wait observability. Path B historically proxied
straight through — which meant callers could overwhelm vLLM's KV cache
and degrade every other in-flight request. This module sits in front of
the proxy handlers and caps concurrency:

- At most ``concurrency`` calls run against vLLM simultaneously.
- A caller that arrives while the slot is full waits on the semaphore.
- Once total (waiting + in-flight) reaches ``max_depth`` new arrivals are
  rejected with a ``QueueFull`` sentinel so the handler can return
  ``503 Service Unavailable`` with ``Retry-After`` — exactly the load-
  shedding contract OpenAI clients already retry on.

Metrics piggyback on the shared Prometheus gauges that Path A emits
(``scalarlm_queue_depth``, ``scalarlm_queue_wait_time_seconds``) so the
observability dashboards merge cleanly across both surfaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    """Raised when the total depth would exceed ``max_depth``.

    Carries ``retry_after`` seconds so the HTTP layer can build a
    ``Retry-After`` header without hard-coding a value here.
    """

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"OpenAI proxy queue full; retry after {retry_after}s")
        self.retry_after = retry_after


class OpenAIConcurrencyLimiter:
    """Acquire-per-request slot with wait-time + depth observability.

    Use via ``acquire(model=...)`` at handler entry; the returned object is
    a callable that must be invoked once (and only once) when the request
    has finished — typically from the ``finally`` of the streaming
    generator. A ``QueueFull`` from ``acquire`` is the handler's cue to
    translate to ``HTTPException(status_code=503, …)``.
    """

    def __init__(
        self,
        *,
        concurrency: int,
        max_depth: int,
        retry_after_seconds: int = 1,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if max_depth < concurrency:
            raise ValueError("max_depth must be >= concurrency")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._concurrency = concurrency
        self._max_depth = max_depth
        self._retry_after_seconds = retry_after_seconds
        self._in_flight = 0
        self._waiting = 0
        self._state_lock = asyncio.Lock()

    @property
    def depth(self) -> int:
        return self._in_flight + self._waiting

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self, *, model: Optional[str] = None) -> "_Slot":
        """Reserve a slot. Raises :class:`QueueFull` when the proxy is
        already past the overflow threshold.
        """
        async with self._state_lock:
            if self.depth >= self._max_depth:
                raise QueueFull(retry_after=self._retry_after_seconds)
            self._waiting += 1
        self._record_depth(model)

        started = time.monotonic()
        try:
            await self._semaphore.acquire()
        except BaseException:
            async with self._state_lock:
                self._waiting -= 1
            self._record_depth(model)
            raise
        wait = time.monotonic() - started

        async with self._state_lock:
            self._waiting -= 1
            self._in_flight += 1
        self._record_depth(model)
        self._record_wait(model, wait)

        return _Slot(limiter=self, model=model, wait_seconds=wait)

    async def _release(self, *, model: Optional[str]) -> None:
        async with self._state_lock:
            if self._in_flight <= 0:
                logger.warning("OpenAIConcurrencyLimiter release with no in-flight slot")
                return
            self._in_flight -= 1
        self._semaphore.release()
        self._record_depth(model)

    def _record_depth(self, model: Optional[str]) -> None:
        try:
            from cray_infra.observability.prometheus_metrics import queue_depth
        except Exception:  # noqa: BLE001 — observability must never break the request path
            return
        try:
            queue_depth.labels(model=model or "unknown").set(self.depth)
        except Exception:  # noqa: BLE001
            pass

    def _record_wait(self, model: Optional[str], seconds: float) -> None:
        try:
            from cray_infra.observability.prometheus_metrics import (
                queue_wait_time_seconds,
            )
        except Exception:  # noqa: BLE001
            return
        try:
            queue_wait_time_seconds.labels(model=model or "unknown").observe(seconds)
        except Exception:  # noqa: BLE001
            pass


class _Slot:
    """Handle returned by :meth:`OpenAIConcurrencyLimiter.acquire`. Must be
    released exactly once. Prefer the handler→generator pattern: handler
    acquires, passes the slot to the streaming wrapper, wrapper releases
    in its ``finally`` so normal completion and error both release.
    """

    __slots__ = ("_limiter", "_model", "_released", "wait_seconds")

    def __init__(
        self,
        *,
        limiter: OpenAIConcurrencyLimiter,
        model: Optional[str],
        wait_seconds: float,
    ) -> None:
        self._limiter = limiter
        self._model = model
        self._released = False
        self.wait_seconds = wait_seconds

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._limiter._release(model=self._model)


_shared_limiter: Optional[OpenAIConcurrencyLimiter] = None
_shared_limiter_lock = asyncio.Lock()


async def get_openai_limiter() -> OpenAIConcurrencyLimiter:
    """Process-wide limiter, lazily built from config on first use."""
    global _shared_limiter
    if _shared_limiter is not None:
        return _shared_limiter
    async with _shared_limiter_lock:
        if _shared_limiter is None:
            from cray_infra.util.get_config import get_config

            config = get_config()
            _shared_limiter = OpenAIConcurrencyLimiter(
                concurrency=int(config["openai_queue_concurrency"]),
                max_depth=int(config["openai_queue_max_depth"]),
                retry_after_seconds=int(config["openai_queue_retry_after_seconds"]),
            )
    return _shared_limiter


def _reset_for_tests() -> None:
    """Test hook — production code should not call this."""
    global _shared_limiter
    _shared_limiter = None
