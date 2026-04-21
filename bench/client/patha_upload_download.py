#!/usr/bin/env python3
"""Path A upload+download — the 1M-prompt shape the colleague cited.

Uploads a JSONL of prompts via ``POST /v1/generate/upload``, polls
``POST /v1/generate/download`` until results are ready. Equivalent to
the Batch API on Path B but using the deprecated wire format. The
deprecation log that fires for every hit is expected.

Usage::

    bench/client/patha_upload_download.py \\
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
    upload_seconds: float
    poll_seconds: float
    total_seconds: float
    prompts_per_second: float
    results_returned: int


async def run(args) -> Result:
    base = args.url.rstrip("/")
    requests = [
        {"prompt": args.prompt, "model": args.model, "max_tokens": args.max_tokens}
        for _ in range(args.prompt_count)
    ]
    upload_body = {"requests": requests}

    async with httpx.AsyncClient(
        http2=not args.no_http2,
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        up_start = time.monotonic()
        up_resp = await client.post(f"{base}/v1/generate/upload", json=upload_body)
        up_t = time.monotonic() - up_start
        up_resp.raise_for_status()
        up_payload = up_resp.json()
        request_ids = [r["request_id"] for r in up_payload.get("results", [])]

        poll_start = time.monotonic()
        results = []
        while True:
            await asyncio.sleep(args.poll_interval)
            r = await client.post(f"{base}/v1/generate/download", json={"request_ids": request_ids})
            r.raise_for_status()
            payload = r.json()
            results = payload.get("results", [])
            if all(item.get("response") is not None or item.get("error") is not None for item in results):
                break
        poll_t = time.monotonic() - poll_start

    total_t = up_t + poll_t
    return Result(
        prompt_count=args.prompt_count,
        upload_seconds=up_t,
        poll_seconds=poll_t,
        total_seconds=total_t,
        prompts_per_second=args.prompt_count / total_t if total_t else 0.0,
        results_returned=len(results),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-count", type=int, required=True)
    p.add_argument("--prompt", default="Say hello in five words.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--poll-interval", type=float, default=0.5)
    p.add_argument("--no-http2", action="store_true")
    p.add_argument("--timeout", type=float, default=3600.0)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    json.dump(asdict(result), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
