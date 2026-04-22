"""Batch runner — consumes input JSONL, posts each line to the upstream
OpenAI endpoint, and appends results to the output JSONL.

Phase 7: dispatches up to ``concurrency`` lines in parallel via
``asyncio.gather`` guarded by a Semaphore. Pre-Phase-7 behaviour was
sequential per-line awaits, which made the Batch API ~5× slower than
the array-completions endpoint at equal N (the flat 1.9–3.2 p/s curve
in the first-pass results). The Phase 3d proxy limiter still bounds
total in-flight work, so this can't bypass backpressure even with many
batches active at once — set ``batch_runner_concurrency`` no larger
than ``openai_queue_concurrency`` (default 16, same value).

Scope notes:
- OpenAI's Batch API does **not** guarantee output-line order — results
  are addressed by ``custom_id``. Parallel dispatch writes lines in
  completion order, which is the documented contract.
- Per-line errors are captured in the output line's ``error`` field
  (OpenAI-compatible shape). A batch only transitions to ``failed`` for
  structural problems (no successful lines, upstream unreachable, etc.).
- Cancellation: ``cancel()`` sets a flag checked at semaphore-acquire
  time. In-flight calls complete naturally — vLLM doesn't expose mid-call
  interruption and aborting would leave KV cache in an ambiguous state.
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
        concurrency: int = 1,
    ) -> None:
        self._batch_id = batch_id
        self._store = store
        self._upstream_call = upstream_call
        self._vllm_api_url = vllm_api_url.rstrip("/")
        self._cancelled = asyncio.Event()
        # Bounded fan-out for line dispatch. concurrency=1 reproduces the
        # pre-Phase-7 sequential behaviour exactly (semaphore is acquired
        # and released around each line).
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self._semaphore = asyncio.Semaphore(concurrency)
        # bump_counts and append_output share the batch's status file and
        # output JSONL respectively. Output appends are O_APPEND-atomic for
        # our small lines, but the read-modify-write in bump_counts must be
        # serialised under concurrent dispatch.
        self._counts_lock = asyncio.Lock()

    def cancel(self) -> None:
        self._cancelled.set()

    async def run(self) -> None:
        self._store.transition(self._batch_id, "in_progress")

        # Snapshot the lines once so the for-loop doesn't pin a file
        # descriptor open across the gather.
        raw_lines = list(self._store.iter_input_lines(self._batch_id))

        # Each task returns (completed_delta, failed_delta) so the outer
        # totals can decide the terminal state without another pass over
        # the store.
        tasks = [asyncio.create_task(self._dispatch_one(raw)) for raw in raw_lines]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        completed = sum(c for c, _ in results)
        failed = sum(f for _, f in results)

        # Terminal state. If any line succeeded we mark completed; otherwise
        # failed. OpenAI's Batch API treats partial success as ``completed``
        # (individual failures live in ``errors`` / per-line output) so we
        # match that contract.
        final_status = self._store.get(self._batch_id)
        if final_status and final_status.status in TERMINAL_STATUSES:
            return  # already cancelled mid-flight
        if completed == 0 and failed > 0:
            self._store.transition(self._batch_id, "failed")
        else:
            self._store.transition(self._batch_id, "completed")

    async def _dispatch_one(self, raw: str) -> tuple[int, int]:
        """Dispatch one input line. Returns (completed, failed) delta."""
        async with self._semaphore:
            # Re-check cancellation under the semaphore — a cancel that
            # arrived while we were queued should drop us before we hit
            # vLLM. In-flight calls past this point are allowed to finish.
            if self._cancelled.is_set():
                current = self._store.get(self._batch_id)
                if current and current.status not in TERMINAL_STATUSES:
                    self._store.transition(self._batch_id, "cancelled")
                return (0, 0)

            parsed, parsed_error = _parse_input_line(raw)
            if parsed_error is not None:
                self._store.append_output(
                    self._batch_id,
                    _error_output_line(parsed, parsed_error),
                )
                async with self._counts_lock:
                    self._store.bump_counts(self._batch_id, failed=1)
                return (0, 1)

            method = parsed["method"]
            url_suffix = parsed["url"]
            body = parsed["body"]
            cid = parsed["custom_id"]
            upstream_url = self._vllm_api_url + url_suffix

            try:
                status_code, response_body = await self._upstream_call(
                    method=method, url=upstream_url, body=body
                )
            except Exception as exc:  # noqa: BLE001 — per-line errors must not fail batch
                logger.exception(
                    "Batch %s line %s upstream failed", self._batch_id, cid
                )
                self._store.append_output(
                    self._batch_id,
                    _error_output_line(cid, {"code": "upstream_error", "message": str(exc)}),
                )
                async with self._counts_lock:
                    self._store.bump_counts(self._batch_id, failed=1)
                return (0, 1)

            is_error = status_code >= 400
            self._store.append_output(
                self._batch_id,
                _response_output_line(cid, status_code, response_body, is_error=is_error),
            )
            async with self._counts_lock:
                self._store.bump_counts(
                    self._batch_id,
                    completed=0 if is_error else 1,
                    failed=1 if is_error else 0,
                )
            return ((0, 1) if is_error else (1, 0))


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
