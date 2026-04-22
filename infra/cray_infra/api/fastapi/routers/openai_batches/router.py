"""FastAPI endpoints for the OpenAI-compatible Batch API.

Phase 3f of the enhancement plan. Subset of OpenAI's spec:

- ``POST   /v1/batches``                          — submit inline JSONL
- ``GET    /v1/batches/{batch_id}``               — status poll
- ``GET    /v1/batches/{batch_id}/output_file_content`` — JSONL results
- ``DELETE /v1/batches/{batch_id}``               — cancel

We intentionally skip the ``/v1/files`` indirection: the request body on
``POST /v1/batches`` carries the JSONL directly as either an ``input``
string field or the raw request body (``application/x-ndjson``). Both are
accepted to keep the surface ergonomic for callers that generate JSONL
programmatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .runner import BatchRunner
from .store import BatchStatus, BatchStore

logger = logging.getLogger(__name__)

openai_batches_router = APIRouter()

# Live runners, keyed by batch id — used only for cancellation. Terminal
# batches drop out here naturally once their task completes.
_runners: dict[str, BatchRunner] = {}
_runners_lock = asyncio.Lock()


def _store() -> BatchStore:
    from cray_infra.util.get_config import get_config

    config = get_config()
    return BatchStore(base_path=config["upload_base_path"])


async def _default_upstream_call(*, method: str, url: str, body: dict):
    """Issue one non-streaming call to the upstream OpenAI-compatible
    endpoint. Returns ``(status_code, parsed_body)``.
    """
    from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

    session = get_global_session()
    # Force non-streaming — batch lines always want a single JSON body.
    body = {**body, "stream": False}
    async with session.request(method, url, json=body) as resp:
        text = await resp.text()
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed = {"error": {"code": "non_json_response", "message": text[:500]}}
        return resp.status, parsed


async def _extract_input_lines(request: Request) -> tuple[list[str], str]:
    """Pull JSONL lines and the target endpoint out of the incoming request.

    Two accepted shapes:
    - JSON body: ``{"input": "<JSONL string>", "endpoint": "/v1/chat/completions"}``.
    - Raw body with ``Content-Type: application/x-ndjson``; endpoint derived
      from the first line's ``url`` field (must be uniform across lines).
    """
    content_type = request.headers.get("content-type", "")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty batch body")

    if "application/json" in content_type and not content_type.startswith("application/x-ndjson"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        input_str = payload.get("input")
        if not isinstance(input_str, str):
            raise HTTPException(status_code=400, detail="missing 'input' (JSONL string)")
        endpoint = payload.get("endpoint") or "/v1/chat/completions"
        lines = [ln for ln in input_str.splitlines() if ln.strip()]
        return lines, endpoint

    # Raw NDJSON fallback.
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="no JSONL lines in body")
    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"first line not JSON: {exc}")
    endpoint = first.get("url", "/v1/chat/completions")
    return lines, endpoint


@openai_batches_router.post("/batches")
async def create_batch(request: Request):
    lines, endpoint = await _extract_input_lines(request)
    store = _store()
    status = store.create(lines, endpoint=endpoint)

    from cray_infra.util.get_config import get_config

    cfg = get_config()
    runner = BatchRunner(
        batch_id=status.id,
        store=store,
        upstream_call=_default_upstream_call,
        vllm_api_url=cfg["vllm_api_url"],
        concurrency=int(cfg.get("batch_runner_concurrency", 16)),
    )
    async with _runners_lock:
        _runners[status.id] = runner

    async def _run_and_cleanup():
        try:
            await runner.run()
        finally:
            async with _runners_lock:
                _runners.pop(status.id, None)

    asyncio.create_task(_run_and_cleanup())
    return asdict(status)


@openai_batches_router.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    status = _store().get(batch_id)
    if status is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return asdict(status)


@openai_batches_router.get("/batches/{batch_id}/output_file_content")
async def get_batch_output(batch_id: str):
    store = _store()
    status = store.get(batch_id)
    if status is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return PlainTextResponse(store.read_output(batch_id), media_type="application/x-ndjson")


@openai_batches_router.delete("/batches/{batch_id}")
async def cancel_batch(batch_id: str):
    store = _store()
    status = store.get(batch_id)
    if status is None:
        raise HTTPException(status_code=404, detail="batch not found")

    async with _runners_lock:
        runner = _runners.get(batch_id)

    # Reflect the decision to cancel synchronously on disk so the DELETE
    # response shows status=cancelled even when the runner is mid-subcall
    # (it'll drain the in-flight call and then observe the terminal state
    # on its next check — see BatchRunner.run's final-status guard).
    if status.status not in ("completed", "failed", "cancelled"):
        store.transition(batch_id, "cancelled")
    if runner is not None:
        runner.cancel()
    return asdict(store.get(batch_id) or status)
