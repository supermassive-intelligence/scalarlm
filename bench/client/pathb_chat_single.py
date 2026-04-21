#!/usr/bin/env python3
"""Per-request chat completions — the interactive-streaming-chat shape.

One OpenAI-style ``POST /v1/chat/completions`` per coroutine, N coroutines
in flight. Measures the Path B / OpenAI-compatible surface's per-request
cost: LoRA preflight, queue slot acquire, upstream vLLM round-trip, usage
scan, queue slot release. This is the shape Path A explicitly cannot do
(no streaming), so there's no Path A baseline to compare it against.

Reports P50/P95/P99 wall-clock latency, throughput, HTTP status
distribution, and the count of 503 Retry-After responses (the Phase 3d
graceful-overload signal).

Usage::

    bench/client/pathb_chat_single.py \\
        --url http://localhost:8000 \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --concurrency 100 \\
        --duration 30

Payload is a short user message so per-request inference time is kept
small; for bigger completions bump ``--max-tokens``.
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
    retry_after_count: int
    errors: int


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    deadline: float,
    latencies_ms: list,
    status_counts: Counter,
    retry_after_count: list,
    errors: list,
) -> None:
    while time.monotonic() < deadline:
        start = time.monotonic()
        try:
            resp = await client.post(url, json=body)
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)
            continue
        latencies_ms.append((time.monotonic() - start) * 1000.0)
        status_counts[resp.status_code] += 1
        if resp.status_code == 503 and "retry-after" in {k.lower() for k in resp.headers}:
            retry_after_count.append(1)


async def run(args) -> Result:
    body = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    url = args.url.rstrip("/") + "/v1/chat/completions"
    limits = httpx.Limits(
        max_connections=args.max_connections or max(args.concurrency, 100),
        max_keepalive_connections=args.max_connections or max(args.concurrency, 100),
    )
    async with httpx.AsyncClient(
        http2=not args.no_http2,
        limits=limits,
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        latencies_ms: list = []
        status_counts: Counter = Counter()
        retry_after: list = []
        errors: list = []

        try:
            await client.post(url, json=body)  # warm pool
        except Exception:
            pass

        started = time.monotonic()
        deadline = started + args.duration
        tasks = [
            asyncio.create_task(
                _worker(client, url, body, deadline, latencies_ms,
                        status_counts, retry_after, errors)
            )
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - started

    sorted_ms = sorted(latencies_ms)
    total = len(sorted_ms)
    if total >= 2:
        qs = quantiles(sorted_ms, n=100, method="inclusive")
        p50, p95, p99 = qs[49], qs[94], qs[98]
    elif total:
        p50 = p95 = p99 = sorted_ms[0]
    else:
        p50 = p95 = p99 = 0.0
    peak = sorted_ms[-1] if sorted_ms else 0.0

    return Result(
        concurrency=args.concurrency,
        duration_seconds=elapsed,
        requests_total=total,
        requests_per_second=total / elapsed if elapsed else 0.0,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=peak,
        status_counts=dict(status_counts),
        retry_after_count=len(retry_after),
        errors=len(errors),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--prompt", default="Say hello in five words.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-connections", type=int, default=None)
    p.add_argument("--no-http2", action="store_true")
    p.add_argument("--timeout", type=float, default=120.0)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    json.dump(asdict(result), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
