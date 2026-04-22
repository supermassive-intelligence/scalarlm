"""Unit tests for Phase 6's in-process vLLM call path.

Phase 6 replaces the localhost-HTTP hop to vLLM's FastAPI server with a
direct call into ``OpenAIServingCompletion.create_completion`` /
``OpenAIServingChat.create_chat_completion``. The helper wraps vLLM's
three possible return shapes — async iterator (streaming SSE), Pydantic
response (non-streaming), ``ErrorResponse`` — so the existing Phase 3d
queue slot is released exactly once and the shared metrics counter is
updated regardless of which shape comes back.

These tests exercise ``_call_inprocess`` directly; the dispatch wrapper
is a thin ``if`` on top of it.
"""

from __future__ import annotations

import pytest
from unittest import mock


# ---- helpers to produce vLLM-shaped return values --------------------------


class _Pydantic:
    """Minimal stand-in for a vLLM CompletionResponse. Has model_dump()
    and is not an async iterator."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _Err:
    """Minimal stand-in for vLLM's ErrorResponse. Has .error.code + model_dump."""

    class _Inner:
        code = 400

    def __init__(self):
        self.error = self._Inner()

    def model_dump(self):
        return {"error": {"code": 400, "message": "bad"}}


async def _sse_stream(lines):
    for line in lines:
        yield line


class _FakeSlot:
    def __init__(self):
        self.released = 0

    async def release(self):
        self.released += 1


def _make_servings(result_for_completion=None, result_for_chat=None):
    servings = mock.MagicMock()
    if result_for_completion is not None:
        servings.openai_serving_completion.create_completion = mock.AsyncMock(
            return_value=result_for_completion
        )
    if result_for_chat is not None:
        servings.openai_serving_chat.create_chat_completion = mock.AsyncMock(
            return_value=result_for_chat
        )
    return servings


# ---- tests -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_releases_slot_and_records_metrics():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    body = {"choices": [{"text": "hi"}], "usage": {"total_tokens": 7}}
    servings = _make_servings(result_for_completion=_Pydantic(body))
    slot = _FakeSlot()

    with mock.patch.object(mod, "compute_flops_per_token", return_value=11):
        metrics = mod.get_metrics()
        before = metrics.total_completed_tokens
        resp = await mod._call_inprocess(
            endpoint="completions",
            request=mock.Mock(),
            raw_request=mock.Mock(),
            servings=servings,
            base_model_name="acme/base",
            queue_slot=slot,
        )

    assert resp.status_code == 200
    assert slot.released == 1
    # token count propagated through the shared metrics counter
    assert metrics.total_completed_tokens == before + 7
    # flop_count = per_token × tokens = 11 × 7
    assert metrics.total_completed_flops >= 77


@pytest.mark.asyncio
async def test_error_response_forwards_status_and_releases_slot():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    servings = _make_servings(result_for_chat=_Err())
    slot = _FakeSlot()

    resp = await mod._call_inprocess(
        endpoint="chat",
        request=mock.Mock(),
        raw_request=mock.Mock(),
        servings=servings,
        base_model_name="acme/base",
        queue_slot=slot,
    )

    assert resp.status_code == 400
    assert slot.released == 1


@pytest.mark.asyncio
async def test_streaming_returns_streaming_response_and_releases_on_drain():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    sse = _sse_stream([
        b'data: {"choices":[{"text":"hi"}]}\n\n',
        b'data: {"usage":{"total_tokens":3}}\n\n',
    ])
    servings = _make_servings(result_for_completion=sse)
    slot = _FakeSlot()

    resp = await mod._call_inprocess(
        endpoint="completions",
        request=mock.Mock(),
        raw_request=mock.Mock(),
        servings=servings,
        base_model_name="acme/base",
        queue_slot=slot,
    )

    # Drain the streaming body like Starlette would.
    assert slot.released == 0  # not yet — slot releases when stream ends
    async for _ in resp.body_iterator:
        pass
    assert slot.released == 1


@pytest.mark.asyncio
async def test_non_streaming_skips_flop_when_token_count_zero():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    body = {"choices": [], "usage": {"total_tokens": 0}}
    servings = _make_servings(result_for_completion=_Pydantic(body))
    slot = _FakeSlot()

    with mock.patch.object(mod, "compute_flops_per_token", return_value=11) as cf:
        await mod._call_inprocess(
            endpoint="completions",
            request=mock.Mock(),
            raw_request=mock.Mock(),
            servings=servings,
            base_model_name="acme/base",
            queue_slot=slot,
        )

    # No tokens → no reason to resolve flops
    cf.assert_not_called()
    assert slot.released == 1


@pytest.mark.asyncio
async def test_dispatcher_picks_inprocess_when_servings_registered():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    servings = _make_servings(
        result_for_completion=_Pydantic({"choices": [], "usage": {"total_tokens": 1}})
    )
    slot = _FakeSlot()

    with mock.patch.object(mod, "get_vllm_servings", return_value=servings), \
         mock.patch.object(mod, "compute_flops_per_token", return_value=0), \
         mock.patch.object(mod, "_proxy_streaming") as proxy:
        await mod._dispatch(
            endpoint="completions",
            request=mock.Mock(),
            raw_request=mock.Mock(),
            params={"model": "acme/base"},
            config={"model": "acme/base", "vllm_api_url": "http://vllm", "openai_inprocess_enabled": True},
            queue_slot=slot,
        )

    proxy.assert_not_called()
    servings.openai_serving_completion.create_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_falls_back_to_http_when_servings_missing():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    with mock.patch.object(mod, "get_vllm_servings", return_value=None), \
         mock.patch.object(mod, "_proxy_streaming") as proxy:
        proxy.return_value = mock.Mock()
        await mod._dispatch(
            endpoint="completions",
            request=mock.Mock(),
            raw_request=mock.Mock(),
            params={"model": "acme/base"},
            config={"model": "acme/base", "vllm_api_url": "http://vllm", "openai_inprocess_enabled": True},
            queue_slot=None,
        )

    proxy.assert_called_once()


@pytest.mark.asyncio
async def test_dispatcher_respects_disabled_flag():
    from cray_infra.api.fastapi.routers import openai_v1_router as mod

    servings = _make_servings(
        result_for_completion=_Pydantic({"choices": [], "usage": {}})
    )

    with mock.patch.object(mod, "get_vllm_servings", return_value=servings), \
         mock.patch.object(mod, "_proxy_streaming") as proxy:
        proxy.return_value = mock.Mock()
        await mod._dispatch(
            endpoint="completions",
            request=mock.Mock(),
            raw_request=mock.Mock(),
            params={"model": "acme/base"},
            config={"model": "acme/base", "vllm_api_url": "http://vllm", "openai_inprocess_enabled": False},
            queue_slot=None,
        )

    proxy.assert_called_once()
    servings.openai_serving_completion.create_completion.assert_not_called()
