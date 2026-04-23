"""
Stream a service log file as newline-delimited JSON.

Query contract (see health_router.py):
    GET /v1/health/logs/{service}?starting_line_number=N&tail=M&limit=K

- `starting_line_number` (int, default 0): seek to this line before
  streaming. Used by the UI to resume tailing from wherever it left
  off after an EOF reconnect.
- `tail` (int | None): if set, ignore `starting_line_number` and
  stream the last `tail` lines of the file. Used for the initial
  "jump to end" load so a 100k-line file doesn't dump its whole
  history to the client.
- `limit` (int | None): cap the number of lines yielded from this
  call. Omit for live-tail connections that should run until EOF;
  set for bounded backfill fetches ("show me 2000 lines starting at
  line 50000").

Each yielded record is:
    {"line": "<text>", "line_number": <0-based>}\n
"""

from cray_infra.util.get_config import get_config

import aiofiles
import os
import json

import logging

logger = logging.getLogger(__name__)


def service_logs_generator(
    service_name: str,
    starting_line_number: int = 0,
    tail: int | None = None,
    limit: int | None = None,
):
    service_log_file = get_service_log_file(service_name)

    logger.info(
        "Using log file: %s (start=%s, tail=%s, limit=%s)",
        service_log_file,
        starting_line_number,
        tail,
        limit,
    )

    # Resolve `tail` up front by counting lines, so the async generator
    # that follows only does one pass. `count_lines` is blocking I/O but
    # the request handler is already IO-bound — the alternative is a
    # two-pass aiofiles read which is materially slower on 100k-line
    # files.
    effective_start = starting_line_number
    if tail is not None and tail > 0:
        total = count_lines(service_log_file)
        effective_start = max(0, total - tail)

    async def generate():
        yielded = 0
        line_number = 0
        async with aiofiles.open(service_log_file, mode="r") as f:
            async for line in f:
                if line_number < effective_start:
                    line_number += 1
                    continue
                if limit is not None and yielded >= limit:
                    break

                yield (
                    json.dumps({"line": line.rstrip(), "line_number": line_number})
                    + "\n"
                )
                line_number += 1
                yielded += 1

    return generate()


def count_lines(path: str) -> int:
    # `sum(1 for _ in f)` is the simplest correct count. For our usage
    # (100k lines ≈ a few MB) it reads at memory-bandwidth speed.
    with open(path, "rb", buffering=1024 * 1024) as f:
        return sum(1 for _ in f)


def get_service_log_file(service_name: str) -> str:
    config = get_config()
    log_base_dir = config["log_directory"]

    service_log_file = os.path.join(log_base_dir, f"{service_name}.log")

    if not os.path.isfile(service_log_file):
        raise FileNotFoundError(f"Log file for service '{service_name}' not found.")

    return service_log_file
