#!/usr/bin/env python3
"""OpenAI Batch API — async bulk chat, Path A's upload/download replacement.

Submits an N-line JSONL of chat-completion requests, polls the batch
until it reaches a terminal status, fetches the output JSONL. Reports
end-to-end wall-clock (submit through last result available) and the
per-prompt throughput so this scenario's numbers land on the same y-axis
as the other bulk scenarios.

Usage::

    bench/client/pathb_batches.py \\
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


TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class Result:
    prompt_count: int
    submit_seconds: float
    poll_seconds: float
    fetch_seconds: float
    total_seconds: float
    status: str
    prompts_per_second: float
    completed_count: int
    failed_count: int


def _build_input_jsonl(prompt_count: int, prompt: str, model: str) -> str:
    lines = []
    for i in range(prompt_count):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0.0,
            },
        }))
    return "\n".join(lines)


async def run(args) -> Result:
    base = args.url.rstrip("/")
    payload = {
        "input": _build_input_jsonl(args.prompt_count, args.prompt, args.model),
        "endpoint": "/v1/chat/completions",
    }

    async with httpx.AsyncClient(
        http2=not args.no_http2,
        timeout=httpx.Timeout(args.timeout),
    ) as client:
        start = time.monotonic()
        submit_resp = await client.post(f"{base}/v1/batches", json=payload)
        submit_t = time.monotonic() - start
        submit_resp.raise_for_status()
        batch = submit_resp.json()

        poll_start = time.monotonic()
        status = batch.get("status")
        while status not in TERMINAL:
            await asyncio.sleep(args.poll_interval)
            r = await client.get(f"{base}/v1/batches/{batch['id']}")
            r.raise_for_status()
            batch = r.json()
            status = batch.get("status")
        poll_t = time.monotonic() - poll_start

        fetch_start = time.monotonic()
        out_resp = await client.get(f"{base}/v1/batches/{batch['id']}/output_file_content")
        out_resp.raise_for_status()
        _ = out_resp.text
        fetch_t = time.monotonic() - fetch_start

    counts = batch.get("request_counts", {})
    total_t = submit_t + poll_t + fetch_t
    return Result(
        prompt_count=args.prompt_count,
        submit_seconds=submit_t,
        poll_seconds=poll_t,
        fetch_seconds=fetch_t,
        total_seconds=total_t,
        status=status,
        prompts_per_second=args.prompt_count / total_t if total_t else 0.0,
        completed_count=counts.get("completed", 0),
        failed_count=counts.get("failed", 0),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-count", type=int, required=True)
    p.add_argument("--prompt", default="Say hello in five words.")
    p.add_argument("--poll-interval", type=float, default=0.5)
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
