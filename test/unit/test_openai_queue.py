"""Unit tests for the OpenAI-proxy concurrency limiter.

Phase 3d of the OpenAI-API enhancement plan. The limiter sits in front of
``/v1/chat/completions`` / ``/v1/completions`` and:

- Caps concurrent calls to vLLM at ``concurrency``.
- Queues extras up to ``max_depth`` total (waiting + in-flight).
- Raises :class:`QueueFull` when the total would overflow, so the handler
  can return 503 + Retry-After.

These tests target the class directly rather than the FastAPI handlers —
the handlers' integration is covered once the test harness stands up a
full in-process app (deferred to integration tests).
"""

from __future__ import annotations

import asyncio

import pytest

from cray_infra.api.fastapi.routers.openai_queue import (
    OpenAIConcurrencyLimiter,
    QueueFull,
)


@pytest.mark.asyncio
async def test_acquire_returns_slot_with_wait_seconds():
    limiter = OpenAIConcurrencyLimiter(concurrency=2, max_depth=4)

    slot = await limiter.acquire(model="m")

    assert slot.wait_seconds >= 0.0
    assert limiter.in_flight == 1
    await slot.release()
    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_release_is_idempotent():
    limiter = OpenAIConcurrencyLimiter(concurrency=2, max_depth=4)
    slot = await limiter.acquire(model="m")

    await slot.release()
    await slot.release()  # no-op

    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_third_caller_waits_when_concurrency_is_two():
    limiter = OpenAIConcurrencyLimiter(concurrency=2, max_depth=4)
    slot_a = await limiter.acquire(model="m")
    slot_b = await limiter.acquire(model="m")

    # Start a third caller — it must NOT complete until a slot frees.
    third = asyncio.create_task(limiter.acquire(model="m"))
    await asyncio.sleep(0)  # let it attempt
    assert not third.done()
    assert limiter.waiting == 1
    assert limiter.in_flight == 2

    # Free a slot; the third caller proceeds.
    await slot_a.release()
    slot_c = await asyncio.wait_for(third, timeout=1.0)
    assert slot_c.wait_seconds > 0.0  # it actually waited
    assert limiter.in_flight == 2

    await slot_b.release()
    await slot_c.release()
    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_overflow_raises_queue_full():
    # concurrency=1, max_depth=2 → at most one in-flight plus one waiting.
    limiter = OpenAIConcurrencyLimiter(concurrency=1, max_depth=2)
    slot_a = await limiter.acquire(model="m")

    # Start one waiter (legal — brings total depth to 2).
    waiter = asyncio.create_task(limiter.acquire(model="m"))
    await asyncio.sleep(0)
    assert limiter.waiting == 1

    # A third caller would push depth to 3 — rejected.
    with pytest.raises(QueueFull) as exc_info:
        await limiter.acquire(model="m")
    assert exc_info.value.retry_after == 1

    await slot_a.release()
    slot_b = await waiter
    await slot_b.release()


@pytest.mark.asyncio
async def test_queue_full_carries_configurable_retry_after():
    limiter = OpenAIConcurrencyLimiter(
        concurrency=1, max_depth=1, retry_after_seconds=7
    )
    slot = await limiter.acquire(model="m")

    with pytest.raises(QueueFull) as exc_info:
        await limiter.acquire(model="m")
    assert exc_info.value.retry_after == 7

    await slot.release()


@pytest.mark.asyncio
async def test_rejects_bad_config():
    with pytest.raises(ValueError):
        OpenAIConcurrencyLimiter(concurrency=0, max_depth=10)
    with pytest.raises(ValueError):
        OpenAIConcurrencyLimiter(concurrency=5, max_depth=2)  # depth < concurrency


@pytest.mark.asyncio
async def test_cancelled_acquire_frees_its_waiting_slot():
    # If a caller is cancelled while waiting, the waiting counter must drop
    # so the queue can accept the next arrival.
    limiter = OpenAIConcurrencyLimiter(concurrency=1, max_depth=3)
    held = await limiter.acquire(model="m")

    waiter = asyncio.create_task(limiter.acquire(model="m"))
    await asyncio.sleep(0)
    assert limiter.waiting == 1

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert limiter.waiting == 0

    await held.release()


@pytest.mark.asyncio
async def test_many_concurrent_callers_never_exceed_concurrency():
    """Soak test: 50 concurrent callers against concurrency=4 must never
    have more than four slots in flight simultaneously.
    """
    limiter = OpenAIConcurrencyLimiter(concurrency=4, max_depth=100)
    observed_max = 0

    async def work():
        nonlocal observed_max
        slot = await limiter.acquire(model="m")
        try:
            observed_max = max(observed_max, limiter.in_flight)
            await asyncio.sleep(0)  # yield so others can race
        finally:
            await slot.release()

    await asyncio.gather(*(work() for _ in range(50)))

    assert observed_max <= 4
    assert limiter.in_flight == 0
    assert limiter.waiting == 0
