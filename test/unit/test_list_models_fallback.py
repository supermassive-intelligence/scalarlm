"""
Pin the contract for the ``GET /v1/models`` handler (``list_models``).

It proxies vLLM's ``/v1/models`` when reachable, and otherwise falls back to an
OpenAI-shaped list built from ``config["model"]`` so model discovery keeps
working while vLLM's endpoint is unavailable (e.g. its prometheus instrumentator
500s, or vLLM is still starting). Mirrors the proxy-then-config fallback already
used by ``resolve_max_model_length``.

The fallback is scoped to *expected* vLLM-unavailability failures (connection
errors, timeouts, malformed bodies); unexpected errors are intentionally left to
propagate rather than being masked as a healthy discovery response.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from cray_infra.api.fastapi.routers import openai_v1_router as r


def _fake_session(payload=None, status=200, raises=None, json_raises=None, seen=None):
    """Mimic aiohttp's ``session.get(...)`` async context manager.

    ``status`` non-200 simulates vLLM error responses; ``raises`` an exception
    on the request itself; ``json_raises`` an exception while decoding the body;
    ``seen`` (a dict) captures the requested URL.
    """
    response = MagicMock()
    response.status = status
    if json_raises is not None:
        response.json = AsyncMock(side_effect=json_raises)
    else:
        response.json = AsyncMock(return_value=payload)

    @asynccontextmanager
    async def fake_get(url, **kwargs):
        if seen is not None:
            seen["url"] = url
            seen["kwargs"] = kwargs
        if raises is not None:
            raise raises
        yield response

    session = MagicMock()
    session.get = fake_get
    return session


def _patches(session, *, model="cfg-model", vllm_url="http://vllm:8001"):
    return (
        patch.object(r, "get_global_session", return_value=session),
        patch.object(
            r,
            "get_config",
            return_value={"vllm_api_url": vllm_url, "model": model},
        ),
    )


def _assert_config_fallback(result, model):
    assert result["object"] == "list"
    assert [m["id"] for m in result["data"]] == [model]
    assert result["data"][0]["object"] == "model"
    assert result["data"][0]["owned_by"] == "scalarlm"
    assert isinstance(result["data"][0]["created"], int)


@pytest.mark.asyncio
async def test_proxies_vllm_when_healthy():
    payload = {"object": "list", "data": [{"id": "a"}, {"id": "b"}]}
    seen: dict = {}
    session = _fake_session(payload, status=200, seen=seen)
    p1, p2 = _patches(session)
    with p1, p2:
        result = await r.list_models()
    assert result == payload
    assert seen["url"] == "http://vllm:8001/v1/models"
    # the discovery call must be bounded by a short timeout (guards regressions)
    assert seen["kwargs"]["timeout"].total == 5


@pytest.mark.asyncio
async def test_falls_back_on_non_200():
    session = _fake_session({}, status=503)
    p1, p2 = _patches(session, model="m")
    with p1, p2:
        result = await r.list_models()
    _assert_config_fallback(result, "m")


@pytest.mark.asyncio
async def test_falls_back_on_client_error():
    session = _fake_session(raises=aiohttp.ClientConnectionError("refused"))
    p1, p2 = _patches(session, model="m")
    with p1, p2:
        result = await r.list_models()
    _assert_config_fallback(result, "m")


@pytest.mark.asyncio
async def test_falls_back_on_timeout():
    session = _fake_session(raises=asyncio.TimeoutError())
    p1, p2 = _patches(session, model="m")
    with p1, p2:
        result = await r.list_models()
    _assert_config_fallback(result, "m")


@pytest.mark.asyncio
async def test_falls_back_on_malformed_json():
    session = _fake_session(status=200, json_raises=json.JSONDecodeError("x", "", 0))
    p1, p2 = _patches(session, model="m")
    with p1, p2:
        result = await r.list_models()
    _assert_config_fallback(result, "m")


@pytest.mark.asyncio
async def test_unexpected_error_is_not_masked():
    # A non-availability error (e.g. a real bug) must propagate rather than be
    # silently turned into a healthy-looking config fallback.
    session = _fake_session(raises=ValueError("boom"))
    p1, p2 = _patches(session, model="m")
    with p1, p2, pytest.raises(ValueError):
        await r.list_models()
