"""Unit tests for the OpenAI Batch API scaffold (Phase 3f).

Covers:
- ``BatchStore`` create/read/transition/append-output semantics.
- Illegal transitions are rejected so bugs in the runner can't drive a
  batch backwards through the state machine.
- ``BatchRunner.run()`` iterates input lines, posts via the injected
  upstream_call, and writes OpenAI-shaped output lines.
- Per-line errors don't abort the batch; the terminal status reflects
  partial success the way OpenAI's spec does.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cray_infra.api.fastapi.routers.openai_batches.runner import BatchRunner
from cray_infra.api.fastapi.routers.openai_batches.store import (
    BatchStore,
    TERMINAL_STATUSES,
)


# ---- BatchStore -----------------------------------------------------------


def test_create_persists_input_and_status(tmp_path: Path):
    store = BatchStore(str(tmp_path))
    lines = [
        json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
        json.dumps({"custom_id": "b", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
    ]
    status = store.create(lines, endpoint="/v1/chat/completions")

    assert status.status == "validating"
    assert status.request_counts == {"total": 2, "completed": 0, "failed": 0}
    assert status.id.startswith("batch_")

    persisted = store.get(status.id)
    assert persisted is not None
    assert persisted.id == status.id

    # input.jsonl round-trips line-for-line.
    read_back = list(store.iter_input_lines(status.id))
    assert read_back == lines


def test_output_file_starts_empty_and_can_be_appended(tmp_path: Path):
    store = BatchStore(str(tmp_path))
    status = store.create(["{}"], endpoint="/v1/chat/completions")

    assert store.read_output(status.id) == ""

    store.append_output(status.id, {"id": "batch_req_1", "custom_id": "a"})
    store.append_output(status.id, {"id": "batch_req_2", "custom_id": "b"})

    lines = [json.loads(ln) for ln in store.read_output(status.id).strip().splitlines()]
    assert [l["custom_id"] for l in lines] == ["a", "b"]


def test_transition_follows_state_machine(tmp_path: Path):
    store = BatchStore(str(tmp_path))
    status = store.create(["{}"], endpoint="/v1/chat/completions")

    store.transition(status.id, "in_progress")
    store.transition(status.id, "completed")

    final = store.get(status.id)
    assert final.status == "completed"
    assert final.completed_at is not None


def test_illegal_transition_raises(tmp_path: Path):
    store = BatchStore(str(tmp_path))
    status = store.create(["{}"], endpoint="/v1/chat/completions")
    store.transition(status.id, "in_progress")
    store.transition(status.id, "completed")

    with pytest.raises(ValueError):
        # Terminal — no outgoing transitions permitted.
        store.transition(status.id, "in_progress")


def test_get_returns_none_for_unknown_batch(tmp_path: Path):
    assert BatchStore(str(tmp_path)).get("batch_missing") is None


def test_bump_counts_is_cumulative(tmp_path: Path):
    store = BatchStore(str(tmp_path))
    status = store.create(["{}", "{}", "{}"], endpoint="/v1/chat/completions")

    store.bump_counts(status.id, completed=1)
    store.bump_counts(status.id, completed=1, failed=1)

    final = store.get(status.id)
    assert final.request_counts == {"total": 3, "completed": 2, "failed": 1}


# ---- BatchRunner ----------------------------------------------------------


def _build_runner(tmp_path: Path, input_lines: list[str], upstream_call):
    store = BatchStore(str(tmp_path))
    status = store.create(input_lines, endpoint="/v1/chat/completions")
    runner = BatchRunner(
        batch_id=status.id,
        store=store,
        upstream_call=upstream_call,
        vllm_api_url="http://vllm",
    )
    return store, status, runner


@pytest.mark.asyncio
async def test_runner_happy_path_writes_output_and_marks_completed(tmp_path: Path):
    calls = []

    async def upstream(*, method, url, body):
        calls.append((method, url, body))
        # Echo the body so tests can assert the wiring.
        return 200, {"echoed": body}

    lines = [
        json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {"n": 1}}),
        json.dumps({"custom_id": "b", "method": "POST", "url": "/v1/chat/completions", "body": {"n": 2}}),
    ]
    store, status, runner = _build_runner(tmp_path, lines, upstream)

    await runner.run()

    final = store.get(status.id)
    assert final.status == "completed"
    assert final.request_counts == {"total": 2, "completed": 2, "failed": 0}

    output = [json.loads(ln) for ln in store.read_output(status.id).strip().splitlines()]
    assert [o["custom_id"] for o in output] == ["a", "b"]
    assert output[0]["response"]["status_code"] == 200
    assert output[0]["response"]["body"] == {"echoed": {"n": 1}}
    assert output[0]["error"] is None

    # Each call is forwarded verbatim to the chosen URL. Stream forcing
    # lives in ``_default_upstream_call`` (to keep the injection seam
    # transport-agnostic), so the runner itself passes the body through.
    assert calls[0][0] == "POST"
    assert calls[0][1] == "http://vllm/v1/chat/completions"
    assert calls[0][2] == {"n": 1}


@pytest.mark.asyncio
async def test_runner_per_line_error_does_not_abort(tmp_path: Path):
    attempt = {"n": 0}

    async def upstream(*, method, url, body):
        attempt["n"] += 1
        if attempt["n"] == 2:
            raise RuntimeError("transient")
        return 200, {"ok": True}

    lines = [
        json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
        json.dumps({"custom_id": "b", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
        json.dumps({"custom_id": "c", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
    ]
    store, status, runner = _build_runner(tmp_path, lines, upstream)

    await runner.run()

    final = store.get(status.id)
    # Matches OpenAI contract: partial success still ends as completed;
    # the per-line error appears in the output file, not as a batch-level
    # failure.
    assert final.status == "completed"
    assert final.request_counts == {"total": 3, "completed": 2, "failed": 1}

    output = [json.loads(ln) for ln in store.read_output(status.id).strip().splitlines()]
    assert output[1]["custom_id"] == "b"
    assert output[1]["response"] is None
    assert output[1]["error"]["code"] == "upstream_error"


@pytest.mark.asyncio
async def test_runner_marks_failed_when_all_lines_fail(tmp_path: Path):
    async def upstream(*, method, url, body):
        raise RuntimeError("everything is broken")

    lines = [
        json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
        json.dumps({"custom_id": "b", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
    ]
    store, status, runner = _build_runner(tmp_path, lines, upstream)

    await runner.run()

    final = store.get(status.id)
    assert final.status == "failed"
    assert final.request_counts["failed"] == 2


@pytest.mark.asyncio
async def test_runner_reports_malformed_input_lines_in_output(tmp_path: Path):
    async def upstream(*, method, url, body):
        raise AssertionError("should not be called for malformed lines")

    lines = [
        "this-is-not-json",  # malformed
        json.dumps({"only": "some fields"}),  # missing required keys
    ]
    store, status, runner = _build_runner(tmp_path, lines, upstream)

    await runner.run()

    final = store.get(status.id)
    assert final.status == "failed"
    output = [json.loads(ln) for ln in store.read_output(status.id).strip().splitlines()]
    assert len(output) == 2
    assert output[0]["error"]["code"] == "invalid_json"
    assert output[1]["error"]["code"] == "missing_fields"


@pytest.mark.asyncio
async def test_runner_cancel_stops_before_next_line(tmp_path: Path):
    progressed = asyncio.Event()

    async def upstream(*, method, url, body):
        progressed.set()
        return 200, {"ok": True}

    lines = [
        json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
        json.dumps({"custom_id": "b", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
        json.dumps({"custom_id": "c", "method": "POST", "url": "/v1/chat/completions", "body": {}}),
    ]
    store, status, runner = _build_runner(tmp_path, lines, upstream)

    # Cancel before anything starts.
    runner.cancel()
    await runner.run()

    final = store.get(status.id)
    assert final.status == "cancelled"
    assert final.status in TERMINAL_STATUSES
