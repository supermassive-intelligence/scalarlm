"""Unit tests for the Path A deprecation-log dependency.

Phase 4 of the enhancement plan. We need a telemetry signal for every
call to the client-facing Path A endpoints so removal can be timed from
real-usage data rather than guesswork. Worker-facing endpoints
(``get_work``, ``finish_work``, ``get_adaptors``, ``clear_queue``,
``metrics``) must *not* emit the log — they're ScalarLM-internal.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from cray_infra.api.fastapi.generate.deprecation import log_path_a_deprecation


def _build_app():
    app = FastAPI()

    @app.post("/v1/generate", dependencies=[Depends(log_path_a_deprecation)])
    async def generate():
        return {"ok": True}

    @app.post("/v1/generate/get_results", dependencies=[Depends(log_path_a_deprecation)])
    async def get_results():
        return {"ok": True}

    @app.post("/v1/generate/upload", dependencies=[Depends(log_path_a_deprecation)])
    async def upload():
        return {"ok": True}

    @app.post("/v1/generate/download", dependencies=[Depends(log_path_a_deprecation)])
    async def download():
        return {"ok": True}

    # No deprecation dependency — represents a worker-facing endpoint.
    @app.post("/v1/generate/get_work")
    async def get_work():
        return {"ok": True}

    return app


@pytest.fixture
def caplog_warning(caplog):
    caplog.set_level(logging.WARNING, logger="cray_infra.api.fastapi.generate.deprecation")
    return caplog


def _deprecation_records(caplog):
    return [
        r for r in caplog.records
        if r.name == "cray_infra.api.fastapi.generate.deprecation"
    ]


def test_generate_hit_logs_deprecation(caplog_warning):
    client = TestClient(_build_app())

    client.post("/v1/generate", json={}, headers={"User-Agent": "sql-gen/1.0"})

    records = _deprecation_records(caplog_warning)
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["event"] == "path_a_deprecation"
    assert payload["path"] == "/v1/generate"
    assert payload["user_agent"] == "sql-gen/1.0"
    assert "migration_hint" in payload
    assert "chat/completions" in payload["migration_hint"]


def test_get_results_hit_logs_deprecation_with_distinct_hint(caplog_warning):
    client = TestClient(_build_app())

    client.post("/v1/generate/get_results", json={})

    payload = json.loads(_deprecation_records(caplog_warning)[0].message)
    assert payload["path"] == "/v1/generate/get_results"
    assert "batches" in payload["migration_hint"]


def test_upload_and_download_each_log_deprecation(caplog_warning):
    client = TestClient(_build_app())

    client.post("/v1/generate/upload", json={})
    client.post("/v1/generate/download", json={})

    paths = [json.loads(r.message)["path"] for r in _deprecation_records(caplog_warning)]
    assert paths == ["/v1/generate/upload", "/v1/generate/download"]


def test_worker_endpoint_does_not_log_deprecation(caplog_warning):
    client = TestClient(_build_app())

    client.post("/v1/generate/get_work", json={})

    assert _deprecation_records(caplog_warning) == []
