"""
Stream a service log file as newline-delimited JSON.

Protocol (see health_router.py):

    GET /v1/health/logs/{service}
        ?tail=N                  # initial jump-to-end: seek to last N lines
        ?starting_byte_offset=B  # resume from a previous record's next_offset
        ?before_byte_offset=B
        &before_count=C          # scrollback: last C lines strictly before B
        ?starting_line_number=N  # (legacy) forward-skip; slow on large N
        ?limit=M                 # cap records in this response

Each yielded record is:

    {
      "line":         "<text>",
      "line_number":  <monotonic label — see below>,
      "byte_offset":  <start of this line in the file>,
      "next_offset":  <byte_offset + len(raw line including newline)>
    }

Positioning is by byte offset. `line_number` is a *label* that increments
across a tailing session but isn't guaranteed to match the file's global
line count — once we seek by byte we no longer know what global line
we landed on. The client drives its own counter.

Why reverse-read: the last-N-lines path used to do a full-file line-count
pass plus a forward `async for` that decoded and discarded every line
until it reached the tail. On an 80k-line log that's two passes of the
whole file, which is both network-slow (aiofiles thread-hops per line)
and wall-clock-slow. `find_tail_offset` reads ~avg_line_length × tail
bytes from the end (a few MB instead of 80+) and streams forward from
there.
"""

from cray_infra.util.get_config import get_config

import aiofiles
import os
import json

import logging

logger = logging.getLogger(__name__)

_NEWLINE = 0x0A  # ord(b"\n")
_BLOCK_SIZE = 64 * 1024


def service_logs_generator(
    service_name: str,
    starting_line_number: int = 0,
    starting_byte_offset: int | None = None,
    tail: int | None = None,
    limit: int | None = None,
    before_byte_offset: int | None = None,
    before_count: int | None = None,
):
    service_log_file = get_service_log_file(service_name)

    start_offset, end_offset = _resolve_range(
        service_log_file,
        starting_line_number=starting_line_number,
        starting_byte_offset=starting_byte_offset,
        tail=tail,
        before_byte_offset=before_byte_offset,
        before_count=before_count,
    )
    line_number_base = max(0, starting_line_number)

    logger.info(
        "Serving log %s from byte %d (end=%s, label_base=%d, limit=%s)",
        service_log_file,
        start_offset,
        end_offset,
        line_number_base,
        limit,
    )

    async def generate():
        yielded = 0
        line_number = line_number_base
        offset = start_offset
        async with aiofiles.open(service_log_file, mode="rb") as f:
            await f.seek(start_offset)
            while True:
                if limit is not None and yielded >= limit:
                    break
                if end_offset is not None and offset >= end_offset:
                    break
                raw = await f.readline()
                if not raw:
                    break
                next_offset = offset + len(raw)
                # before-window requests must not spill past `before` —
                # truncate if we'd cross that boundary.
                if end_offset is not None and next_offset > end_offset:
                    break
                line = raw.rstrip(b"\r\n").decode("utf-8", errors="replace")
                yield (
                    json.dumps(
                        {
                            "line": line,
                            "line_number": line_number,
                            "byte_offset": offset,
                            "next_offset": next_offset,
                        }
                    )
                    + "\n"
                )
                offset = next_offset
                line_number += 1
                yielded += 1

    return generate()


def _resolve_range(
    path: str,
    *,
    starting_line_number: int,
    starting_byte_offset: int | None,
    tail: int | None,
    before_byte_offset: int | None,
    before_count: int | None,
) -> tuple[int, int | None]:
    """Return (start_offset, optional end_offset) picking the most-specific
    request. Precedence: before-window > tail > byte_offset > line_number."""
    if (
        before_byte_offset is not None
        and before_count is not None
        and before_count > 0
    ):
        start = find_before_offset(path, before_byte_offset, before_count)
        return start, before_byte_offset

    if tail is not None and tail > 0:
        return find_tail_offset(path, tail), None

    if starting_byte_offset is not None and starting_byte_offset > 0:
        return starting_byte_offset, None

    if starting_line_number > 0:
        # Legacy forward-skip. Slow on large N; retained so an older UI
        # bundle talking to a newer server still works, and so direct
        # curl consumers aren't surprised.
        return _forward_skip_offset(path, starting_line_number), None

    return 0, None


def find_tail_offset(
    path: str, tail: int, block_size: int = _BLOCK_SIZE
) -> int:
    """Byte offset of the start of the last `tail` lines.

    Reads the file backward in `block_size` chunks, counting newlines,
    stopping as soon as it's found `tail` line boundaries. For a
    100k-line file with `tail=5000`, touches ~1 MB instead of the full
    file.
    """
    if tail <= 0:
        return 0
    end = os.path.getsize(path)
    if end == 0:
        return 0

    # If the file ends in a newline, that newline is the terminator of
    # the last line — not a candidate for "start of a new line." Skip it
    # so the counting below lands on the right boundary.
    with open(path, "rb") as f:
        f.seek(end - 1)
        last = f.read(1)
    scan_end = end - 1 if last == b"\n" else end
    if scan_end <= 0:
        return 0

    return _find_offset_after_nth_newline_backward(
        path, scan_end, tail, block_size
    )


def find_before_offset(
    path: str, before: int, count: int, block_size: int = _BLOCK_SIZE
) -> int:
    """Byte offset of the start of the `count`-th line strictly before `before`.

    `before` is a start-of-line offset (what the client passes as
    `before_byte_offset` = an earlier record's byte_offset). Walking
    back `count + 1` newlines lands at the start of the target line:
    the first newline takes us across the boundary into the previous
    line, and each subsequent one moves one full line further back.
    """
    if count <= 0 or before <= 0:
        return 0
    return _find_offset_after_nth_newline_backward(
        path, before, count + 1, block_size
    )


def _find_offset_after_nth_newline_backward(
    path: str, start_pos: int, k: int, block_size: int
) -> int:
    if k <= 0 or start_pos <= 0:
        return 0
    with open(path, "rb") as f:
        pos = start_pos
        remaining = k
        while pos > 0:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            for i in range(len(chunk) - 1, -1, -1):
                if chunk[i] == _NEWLINE:
                    remaining -= 1
                    if remaining == 0:
                        return pos + i + 1
    return 0


def _forward_skip_offset(path: str, starting_line_number: int) -> int:
    """Legacy path. Walks the file from 0 and returns the byte offset of
    the `starting_line_number`-th line. Only used when neither `tail`
    nor `starting_byte_offset` is provided."""
    if starting_line_number <= 0:
        return 0
    with open(path, "rb", buffering=_BLOCK_SIZE) as f:
        seen = 0
        while seen < starting_line_number:
            line = f.readline()
            if not line:
                return f.tell()
            seen += 1
        return f.tell()


def get_service_log_file(service_name: str) -> str:
    config = get_config()
    log_base_dir = config["log_directory"]

    service_log_file = os.path.join(log_base_dir, f"{service_name}.log")

    if not os.path.isfile(service_log_file):
        raise FileNotFoundError(f"Log file for service '{service_name}' not found.")

    return service_log_file
