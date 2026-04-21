"""Batch runner — consumes input JSONL, posts each line to the upstream
OpenAI endpoint, and appends results to the output JSONL.

Scope-light on purpose:
- Each batch runs as a single asyncio task, one sub-request at a time.
  Concurrency within a batch is bounded by the Phase 3d proxy limiter, so
  stacking many batches at once doesn't bypass backpressure.
- Per-line errors don't fail the batch; they're captured in the output
  line's ``error`` field, OpenAI-compatible shape. A batch only transitions
  to ``failed`` for structural problems (malformed input line, upstream
  unreachable after retries).
- Cancellation: a ``cancel()`` call sets a flag the runner checks between
  sub-requests; the currently-in-flight call is allowed to complete. No
  mid-call interruption — vLLM doesn't expose one and it'd leave KV cache
  in an ambiguous state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Optional

from .store import BatchStore, TERMINAL_STATUSES

logger = logging.getLogger(__name__)


class BatchRunner:
    """Run one batch end-to-end. Construct per batch, call ``run()``.

    ``upstream_call`` is injected so tests can swap it for a fake without
    mocking aiohttp. Signature: ``async (method: str, url: str, body:
    dict) -> tuple[int, dict]`` returning the upstream status and parsed
    JSON body.
    """

    def __init__(
        self,
        *,
        batch_id: str,
        store: BatchStore,
        upstream_call,
        vllm_api_url: str,
    ) -> None:
        self._batch_id = batch_id
        self._store = store
        self._upstream_call = upstream_call
        self._vllm_api_url = vllm_api_url.rstrip("/")
        self._cancelled = asyncio.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    async def run(self) -> None:
        status = self._store.transition(self._batch_id, "in_progress")

        total = status.request_counts["total"]
        completed = 0
        failed = 0

        for raw in self._store.iter_input_lines(self._batch_id):
            if self._cancelled.is_set():
                self._store.transition(self._batch_id, "cancelled")
                return

            custom_id, parsed_error = _parse_input_line(raw)
            if parsed_error is not None:
                self._store.append_output(
                    self._batch_id,
                    _error_output_line(custom_id, parsed_error),
                )
                failed += 1
                self._store.bump_counts(self._batch_id, failed=1)
                continue

            method = custom_id["method"]
            url_suffix = custom_id["url"]
            body = custom_id["body"]
            cid = custom_id["custom_id"]

            upstream_url = self._vllm_api_url + url_suffix
            try:
                status_code, response_body = await self._upstream_call(
                    method=method, url=upstream_url, body=body
                )
            except Exception as exc:  # noqa: BLE001 — per-line errors must not fail batch
                logger.exception("Batch %s line %s upstream failed", self._batch_id, cid)
                self._store.append_output(
                    self._batch_id,
                    _error_output_line(cid, {"code": "upstream_error", "message": str(exc)}),
                )
                failed += 1
                self._store.bump_counts(self._batch_id, failed=1)
                continue

            if status_code >= 400:
                self._store.append_output(
                    self._batch_id,
                    _response_output_line(cid, status_code, response_body, is_error=True),
                )
                failed += 1
                self._store.bump_counts(self._batch_id, failed=1)
            else:
                self._store.append_output(
                    self._batch_id,
                    _response_output_line(cid, status_code, response_body, is_error=False),
                )
                completed += 1
                self._store.bump_counts(self._batch_id, completed=1)

        # Terminal state. If any line succeeded we mark completed; otherwise
        # failed. OpenAI's Batch API treats partial success as ``completed``
        # (individual failures live in ``errors`` / per-line output) so we
        # match that contract.
        final_status = self._store.get(self._batch_id)
        if final_status and final_status.status in TERMINAL_STATUSES:
            return  # already cancelled
        if completed == 0 and failed > 0:
            self._store.transition(self._batch_id, "failed")
        else:
            self._store.transition(self._batch_id, "completed")


def _parse_input_line(raw: str):
    """Return (parsed, error). ``parsed`` is a dict with validated fields;
    ``error`` is an OpenAI-shaped error dict when the line is malformed.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {"code": "invalid_json", "message": str(exc)}

    if not isinstance(obj, dict):
        return None, {"code": "invalid_input", "message": "line must be a JSON object"}

    missing = [k for k in ("custom_id", "method", "url", "body") if k not in obj]
    if missing:
        return None, {"code": "missing_fields", "message": f"missing: {missing}"}

    return obj, None


def _response_output_line(custom_id: str, status_code: int, body: dict, *, is_error: bool) -> dict:
    line_id = "batch_req_" + uuid.uuid4().hex
    return {
        "id": line_id,
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": line_id,
            "body": body,
        },
        "error": None if not is_error else {"code": "upstream_error", "status_code": status_code},
    }


def _error_output_line(custom_id, error: dict) -> dict:
    # custom_id might be the parsed dict (rare malformed line caught post-parse)
    # or a string from a well-formed one.
    cid = custom_id if isinstance(custom_id, str) else (
        custom_id.get("custom_id") if isinstance(custom_id, dict) else None
    )
    line_id = "batch_req_" + uuid.uuid4().hex
    return {
        "id": line_id,
        "custom_id": cid,
        "response": None,
        "error": error,
    }
