"""Unit tests for the request-id + request-log middleware.

Covers Phase 3a of the OpenAI-API enhancement plan (see enhance-openai-api.md):
every response carries ``X-Request-Id``, client-supplied values are preserved,
and handlers can read the id from ``request.state.request_id``.

These tests deliberately do not import ``cray_infra.api.fastapi.main`` — that
would transitively pull in vLLM and the full app graph. Instead we build a
minimal FastAPI app and attach the middleware directly, which is precisely
what ``main.py`` does.
"""

import re

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from cray_infra.api.fastapi.middleware.request_id import (
    request_id_and_log_middleware,
)


UUID_HEX = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture
def client():
    """Minimal app with the middleware and a handful of echo routes."""
    app = FastAPI()
    app.middleware("http")(request_id_and_log_middleware)

    @app.get("/echo")
    async def echo(request: Request):
        return {"request_id": request.state.request_id}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("deliberate")

    return TestClient(app, raise_server_exceptions=False)


def test_response_carries_x_request_id_header(client):
    resp = client.get("/echo")

    assert resp.status_code == 200
    assert "X-Request-Id" in resp.headers
    assert UUID_HEX.match(resp.headers["X-Request-Id"])


def test_handler_reads_same_request_id(client):
    # The handler echoes request.state.request_id; it must match the header
    # the client receives — otherwise logs and the response drift.
    resp = client.get("/echo")

    assert resp.json()["request_id"] == resp.headers["X-Request-Id"]


def test_client_supplied_header_is_preserved(client):
    # If a caller already has a correlation ID (e.g. from an upstream gateway)
    # we honour it rather than minting a new one, so trace chains survive.
    resp = client.get("/echo", headers={"X-Request-Id": "caller-provided-id"})

    assert resp.headers["X-Request-Id"] == "caller-provided-id"
    assert resp.json()["request_id"] == "caller-provided-id"


def test_distinct_requests_get_distinct_ids(client):
    first = client.get("/echo").headers["X-Request-Id"]
    second = client.get("/echo").headers["X-Request-Id"]

    assert first != second


def test_error_response_still_carries_request_id(client):
    # When the handler raises, the response is 500. The id still has to be on
    # it — that's exactly the scenario where ops wants to correlate logs.
    resp = client.get("/boom")

    assert resp.status_code == 500
    assert "X-Request-Id" in resp.headers
