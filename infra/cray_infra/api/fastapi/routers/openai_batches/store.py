"""File-backed storage for OpenAI Batch API jobs.

Phase 3f of the enhancement plan. The batch runtime lives in this
package; ``store.py`` is the on-disk layer. Each batch keeps a directory
under ``{upload_base_path}/batches/<batch_id>/`` holding:

- ``input.jsonl``    — the submitted request lines, written once.
- ``output.jsonl``   — result lines, appended as individual sub-requests
                       complete. Partial output is readable during
                       ``in_progress`` so clients can stream-consume.
- ``status.json``    — batch metadata (id, status, counts, timestamps).

Durability is intentionally modest: recovery after a process crash means
re-running in-flight work using ``input.jsonl``. A full SQLite queue
would be overkill here — the Batch API's async-poll semantics already
absorb any restart.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional


# Valid state transitions for a batch. Restrictive on purpose so bugs in
# the runner can't silently move a batch backwards.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "validating": frozenset({"in_progress", "failed", "cancelled"}),
    "in_progress": frozenset({"completed", "failed", "cancelled"}),
    # Terminal states — no outgoing transitions.
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class BatchStatus:
    """Serialisable metadata for one batch. Shape chosen to match OpenAI's
    ``Batch`` object where reasonable; we intentionally drop fields we
    don't implement (input_file_id, output_file_id, endpoint completion
    window) rather than stub them with lies.
    """

    id: str
    object: str = "batch"
    endpoint: str = "/v1/chat/completions"
    status: str = "validating"
    created_at: int = 0
    in_progress_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: dict = field(default_factory=lambda: {"total": 0, "completed": 0, "failed": 0})
    errors: Optional[list] = None


class BatchStore:
    """File-backed batch registry. Thread-safe within a single process as
    long as callers don't mutate the same batch concurrently (which the
    runner guarantees — one task per batch).
    """

    def __init__(self, base_path: str) -> None:
        self._base = os.path.join(base_path, "batches")
        os.makedirs(self._base, exist_ok=True)

    def _dir(self, batch_id: str) -> str:
        return os.path.join(self._base, batch_id)

    def _status_path(self, batch_id: str) -> str:
        return os.path.join(self._dir(batch_id), "status.json")

    def _input_path(self, batch_id: str) -> str:
        return os.path.join(self._dir(batch_id), "input.jsonl")

    def _output_path(self, batch_id: str) -> str:
        return os.path.join(self._dir(batch_id), "output.jsonl")

    # ---- creation & access ------------------------------------------------

    def create(self, input_lines: list[str], *, endpoint: str) -> BatchStatus:
        batch_id = "batch_" + uuid.uuid4().hex
        os.makedirs(self._dir(batch_id), exist_ok=False)

        with open(self._input_path(batch_id), "w", encoding="utf-8") as fh:
            for line in input_lines:
                # Persist the caller's bytes as-is; each line must already
                # be a JSON object terminated by '\n'. We strip only the
                # trailing newline we'll re-add ourselves so the file is
                # canonical regardless of caller quirks.
                fh.write(line.rstrip("\n") + "\n")
        # Create output file eagerly so tail-readers don't 404.
        open(self._output_path(batch_id), "w", encoding="utf-8").close()

        status = BatchStatus(
            id=batch_id,
            endpoint=endpoint,
            status="validating",
            created_at=int(time.time()),
            request_counts={"total": len(input_lines), "completed": 0, "failed": 0},
        )
        self._write_status(status)
        return status

    def get(self, batch_id: str) -> Optional[BatchStatus]:
        path = self._status_path(batch_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return BatchStatus(**raw)

    def iter_input_lines(self, batch_id: str) -> Iterator[str]:
        with open(self._input_path(batch_id), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line

    def read_output(self, batch_id: str) -> str:
        with open(self._output_path(batch_id), "r", encoding="utf-8") as fh:
            return fh.read()

    # ---- mutations --------------------------------------------------------

    def append_output(self, batch_id: str, line: dict) -> None:
        with open(self._output_path(batch_id), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def transition(self, batch_id: str, new_status: str, *, error: Optional[dict] = None) -> BatchStatus:
        status = self.get(batch_id)
        if status is None:
            raise KeyError(batch_id)
        allowed = _ALLOWED_TRANSITIONS[status.status]
        if new_status not in allowed:
            raise ValueError(
                f"Illegal batch transition {status.status!r} → {new_status!r}"
            )

        status.status = new_status
        now = int(time.time())
        if new_status == "in_progress":
            status.in_progress_at = now
        elif new_status == "completed":
            status.completed_at = now
        elif new_status == "failed":
            status.failed_at = now
            if error is not None:
                status.errors = (status.errors or []) + [error]
        elif new_status == "cancelled":
            status.cancelled_at = now

        self._write_status(status)
        return status

    def bump_counts(self, batch_id: str, *, completed: int = 0, failed: int = 0) -> BatchStatus:
        status = self.get(batch_id)
        if status is None:
            raise KeyError(batch_id)
        status.request_counts["completed"] += completed
        status.request_counts["failed"] += failed
        self._write_status(status)
        return status

    # ---- internals --------------------------------------------------------

    def _write_status(self, status: BatchStatus) -> None:
        # Write-then-rename for crash-safety: a partial status.json after a
        # crash would break the server-side reader and the client couldn't
        # tell a dead batch from a fresh one.
        tmp = self._status_path(status.id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(status), fh)
        os.replace(tmp, self._status_path(status.id))
