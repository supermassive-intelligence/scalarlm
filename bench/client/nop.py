#!/usr/bin/env python3
"""Hammer ``GET /v1/bench/nop`` to measure the FastAPI / uvicorn ceiling.

Decouples transport-layer capacity from the inference layer so the
platform triangle (M5 / Spark / 4× Blackwell) has a baseline number
that isn't entangled with model decode rate. See the "Benchmark plan"
section of ``enhance-openai-api.md``.

Usage::

    bench/client/nop.py --url http://localhost:8000 --concurrency 1000 --duration 10

The runner opens ``--concurrency`` async tasks, each re-issuing requests
on an HTTP/2-capable client for ``--duration`` seconds, then reports
throughput, P50/P95/P99 latency, and HTTP status distribution as JSON.

Target the number of in-flight requests via ``--concurrency`` (distinct
coroutines) and the total TCP pool via ``--max-connections`` — with
HTTP/2 the pool can be much smaller than the concurrency number, which
is precisely the point the plan makes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from statistics import quantiles
from typing import Optional

try:
    import httpx
except ImportError:
    print("httpx not installed. `pip install 'httpx[http2]'` in your bench venv.", file=sys.stderr)
    sys.exit(1)


@dataclass
class Result:
    concurrency: int
    duration_seconds: float
    requests_total: int
    requests_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    status_counts: dict
    errors: int


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    deadline: float,
    latencies_ms: list,
    status_counts: Counter,
    errors: list,
) -> None:
    while time.monotonic() < deadline:
        start = time.monotonic()
        try:
            resp = await client.get(url)
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)
            continue
        latencies_ms.append((time.monotonic() - start) * 1000.0)
        status_counts[resp.status_code] += 1


async def run(
    *,
    url: str,
    concurrency: int,
    duration_seconds: float,
    max_connections: Optional[int],
    http2: bool,
    timeout_seconds: float,
) -> Result:
    limits = httpx.Limits(
        max_connections=max_connections or max(concurrency, 100),
        max_keepalive_connections=max_connections or max(concurrency, 100),
    )
    async with httpx.AsyncClient(
        http2=http2,
        limits=limits,
        timeout=httpx.Timeout(timeout_seconds),
    ) as client:
        latencies_ms: list = []
        status_counts: Counter = Counter()
        errors: list = []

        # Warm the connection pool with a single request before the
        # measurement window; otherwise the first cohort's TCP/TLS setup
        # swamps the P99.
        try:
            await client.get(url)
        except Exception:
            pass

        started = time.monotonic()
        deadline = started + duration_seconds
        tasks = [
            asyncio.create_task(
                _worker(client, url, deadline, latencies_ms, status_counts, errors)
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - started

    sorted_ms = sorted(latencies_ms)
    total = len(sorted_ms)
    if total:
        # statistics.quantiles with n=100 gives percentile cuts at 1..99.
        # Guard against tiny samples where quantiles() would raise.
        if total >= 2:
            qs = quantiles(sorted_ms, n=100, method="inclusive")
            p50 = qs[49]
            p95 = qs[94]
            p99 = qs[98]
        else:
            p50 = p95 = p99 = sorted_ms[0]
        peak = sorted_ms[-1]
    else:
        p50 = p95 = p99 = peak = 0.0

    return Result(
        concurrency=concurrency,
        duration_seconds=elapsed,
        requests_total=total,
        requests_per_second=total / elapsed if elapsed else 0.0,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=peak,
        status_counts=dict(status_counts),
        errors=len(errors),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000/v1/bench/nop")
    p.add_argument("--concurrency", type=int, default=100,
                   help="Number of in-flight async requests")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Seconds to sustain load")
    p.add_argument("--max-connections", type=int, default=None,
                   help="TCP pool ceiling (default: match concurrency)")
    p.add_argument("--no-http2", action="store_true",
                   help="Use HTTP/1.1 (default is HTTP/2 if server supports it)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-request timeout in seconds")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(
        url=args.url,
        concurrency=args.concurrency,
        duration_seconds=args.duration,
        max_connections=args.max_connections,
        http2=not args.no_http2,
        timeout_seconds=args.timeout,
    ))
    json.dump(asdict(result), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
