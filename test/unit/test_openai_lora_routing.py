"""Unit tests for Path B LoRA-adapter routing.

Phase 3c of the OpenAI-API enhancement plan: when a caller targets an
adapter by ``model`` name on ``/v1/chat/completions`` or ``/v1/completions``,
the proxy must load the adapter into the upstream vLLM server exactly
once, coalescing concurrent first-use callers.

The tests exercise ``ensure_adapter_loaded`` directly — they would otherwise
have to stand up the full FastAPI app, which pulls in vLLM. They inject a
minimal fake aiohttp-style session and the stock ``StubVLLMModelManager``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from cray_infra.api.fastapi.routers import openai_lora


class _FakeResponse:
    def __init__(self, status: int = 200, body: str = ""):
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    """Minimal stand-in for aiohttp.ClientSession with just ``post``."""

    def __init__(self, response_factory=None):
        self.calls: list[tuple[str, Any]] = []
        self._response_factory = response_factory or (lambda: _FakeResponse(200, "{}"))

    def post(self, url: str, *, json):
        self.calls.append((url, json))
        return self._response_factory()


class _StubModelManager:
    """Collapsed version of StubVLLMModelManager carrying just the surface
    we need — base vs adapter resolution.
    """

    BASE = "base-model"

    def __init__(self, known_adapters=()):
        self._adapters = set(known_adapters)

    def find_model(self, name):
        if name == self.BASE:
            return self.BASE
        if name in self._adapters:
            return name
        return None


@pytest.fixture(autouse=True)
def _reset_state():
    openai_lora._reset_state_for_tests()


def _kwargs(session, manager=None, model_name="my-finetune"):
    return dict(
        session=session,
        vllm_api_url="http://vllm",
        model_name=model_name,
        model_manager=manager or _StubModelManager(),
        training_job_directory="/tmp/jobs",
        base_model=_StubModelManager.BASE,
    )


@pytest.mark.asyncio
async def test_base_model_request_does_not_call_load_lora():
    session = _FakeSession()
    await openai_lora.ensure_adapter_loaded(
        **_kwargs(session, model_name=_StubModelManager.BASE)
    )

    assert session.calls == []


@pytest.mark.asyncio
async def test_unknown_model_does_not_call_load_lora():
    # Unknown names are forwarded unchanged so vLLM returns its own
    # 400/404 — the proxy should not try to load a bogus path.
    session = _FakeSession()
    await openai_lora.ensure_adapter_loaded(
        **_kwargs(session, model_name="not-a-model")
    )

    assert session.calls == []


@pytest.mark.asyncio
async def test_adapter_loads_once_on_first_request():
    session = _FakeSession()
    manager = _StubModelManager(known_adapters={"my-finetune"})

    await openai_lora.ensure_adapter_loaded(**_kwargs(session, manager))

    assert len(session.calls) == 1
    url, payload = session.calls[0]
    assert url == "http://vllm/v1/load_lora_adapter"
    assert payload == {"lora_name": "my-finetune", "lora_path": "/tmp/jobs/my-finetune"}


@pytest.mark.asyncio
async def test_adapter_does_not_reload_on_second_request():
    session = _FakeSession()
    manager = _StubModelManager(known_adapters={"my-finetune"})

    for _ in range(3):
        await openai_lora.ensure_adapter_loaded(**_kwargs(session, manager))

    assert len(session.calls) == 1  # coalesced after first success


@pytest.mark.asyncio
async def test_concurrent_first_use_coalesces_into_one_call():
    """Thundering-herd check: 10 concurrent callers for the same not-yet-
    loaded adapter must produce exactly one upstream HTTP call.
    """
    release = asyncio.Event()

    class _SlowResponse(_FakeResponse):
        async def __aenter__(self):
            await release.wait()
            return self

    session = _FakeSession(response_factory=lambda: _SlowResponse(200, "{}"))
    manager = _StubModelManager(known_adapters={"my-finetune"})

    async def call():
        await openai_lora.ensure_adapter_loaded(**_kwargs(session, manager))

    tasks = [asyncio.create_task(call()) for _ in range(10)]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*tasks)

    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_failed_load_surfaces_as_runtime_error_and_allows_retry():
    attempts = {"count": 0}

    def factory():
        attempts["count"] += 1
        return _FakeResponse(status=500, body='{"error":"kaboom"}')

    failing_session = _FakeSession(response_factory=factory)
    manager = _StubModelManager(known_adapters={"my-finetune"})

    with pytest.raises(RuntimeError, match="Failed to load adapter my-finetune"):
        await openai_lora.ensure_adapter_loaded(**_kwargs(failing_session, manager))

    # State must be cleared so the next caller retries rather than hanging.
    good_session = _FakeSession()
    await openai_lora.ensure_adapter_loaded(**_kwargs(good_session, manager))

    assert attempts["count"] == 1  # first attempt failed
    assert len(good_session.calls) == 1  # retry issued


@pytest.mark.asyncio
async def test_none_model_is_noop():
    session = _FakeSession()
    await openai_lora.ensure_adapter_loaded(**_kwargs(session, model_name=None))

    assert session.calls == []
