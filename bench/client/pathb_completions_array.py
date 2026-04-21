#!/usr/bin/env python3
"""Array-prompt /v1/completions — the wire-batched Path B equivalent.

One HTTP call carries ``--prompt-count`` prompts in ``prompt=[...]``;
vLLM continues-batches them internally. This is the apples-to-apples
comparison point for Path A's bulk ``POST /v1/generate`` call: equal N
on this scenario and Path A should produce within 5 % wall-clock time
per the plan's parity threshold.

Usage::

    bench/client/pathb_completions_array.py \\
        --url http://localhost:8000 \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --prompt-count 1000

One call per invocation — the script is called once per prompt-count
level by the sweep driver.
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
    total_tokens: Optional[int]
    tokens_per_second: Optional[float]


async def run(args) -> Result:
    body = {
        "model": args.model,
        "prompt": [args.prompt] * args.prompt_count,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    url = args.url.rstrip("/") + "/v1/completions"
    async with httpx.AsyncClient(
        http2=not args.no_http2,
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        start = time.monotonic()
        resp = await client.post(url, json=body)
        elapsed = time.monotonic() - start

    total_tokens: Optional[int] = None
    if resp.status_code == 200:
        try:
            payload = resp.json()
            total_tokens = payload.get("usage", {}).get("total_tokens")
        except Exception:  # noqa: BLE001
            pass

    return Result(
        prompt_count=args.prompt_count,
        duration_seconds=elapsed,
        status_code=resp.status_code,
        prompts_per_second=args.prompt_count / elapsed if elapsed else 0.0,
        total_tokens=total_tokens,
        tokens_per_second=(total_tokens / elapsed) if total_tokens and elapsed else None,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-count", type=int, required=True)
    p.add_argument("--prompt", default="Hello. Say hi back in five words.")
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
