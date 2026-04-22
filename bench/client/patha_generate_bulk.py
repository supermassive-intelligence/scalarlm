#!/usr/bin/env python3
"""Path A wire-batched /v1/generate — the throughput baseline.

One call with ``prompts=[p1, ..., pN]`` to Path A. The deprecation log
that now fires for every hit is expected; the benchmark is specifically
measuring the deprecated path to establish the number Path B's array
/v1/completions scenario has to stay within 5 % of.

Usage::

    bench/client/patha_generate_bulk.py \\
        --url http://localhost:8000 \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --prompt-count 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import httpx
except ImportError:
    print("httpx not installed. `pip install 'httpx[http2]'` in your bench venv.", file=sys.stderr)
    sys.exit(1)


@dataclass
class Result:
    prompt_count: int
    duration_seconds: float
    status_code: int
    prompts_per_second: float
    results_returned: int


async def run(args) -> Result:
    if args.distinct_prompts:
        prompts = [f"req {i:06d}: {args.prompt}" for i in range(args.prompt_count)]
    else:
        prompts = [args.prompt] * args.prompt_count
    body = {
        "model": args.model,
        "prompts": prompts,
        "max_tokens": args.max_tokens,
    }
    url = args.url.rstrip("/") + "/v1/generate"
    async with httpx.AsyncClient(
        http2=not args.no_http2,
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        start = time.monotonic()
        resp = await client.post(url, json=body)
        elapsed = time.monotonic() - start

    results_returned = 0
    if resp.status_code == 200:
        try:
            payload = resp.json()
            results_returned = len(payload.get("results", []))
        except Exception:  # noqa: BLE001
            pass

    return Result(
        prompt_count=args.prompt_count,
        duration_seconds=elapsed,
        status_code=resp.status_code,
        prompts_per_second=args.prompt_count / elapsed if elapsed else 0.0,
        results_returned=results_returned,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-count", type=int, required=True)
    p.add_argument("--prompt", default="Say hello in five words.")
    p.add_argument("--distinct-prompts", action="store_true",
                   help="Generate N unique prompts instead of N copies.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--no-http2", action="store_true")
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    json.dump(asdict(result), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
