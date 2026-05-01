"""
FastAPI handler for /v1/chat/completions (non-streaming, queue-backed).

Stream=True requests are delegated to the existing direct-to-vLLM
proxy (see openai_v1_router.py); only stream=False flows through the
queue. See docs/openai-chat-completions-queue.md §3.1.

The handler is a thin orchestrator over four foundation pieces:

  1. render_chat_template — turns `messages: [...]` into a prompt str
  2. is_over_high_water + WaitEstimator — admission decision and 429
     Retry-After hint
  3. ResultRouter — registers a correlation_id before submission so
     the worker has somewhere to deliver the result
  4. Coalescer — accumulates admitted requests; the configured flush
     callback (`enqueue_coalesced_batch`) writes one batch as one
     SQLite row.

The HTTP response is the heartbeat-padded chunked-JSON streamer from
`heartbeat.py`; the body generator wraps it in a try/finally that
unregisters the correlation id on completion or client disconnect.
"""

import logging
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cray_infra.api.fastapi.chat_completions.admission import (
    WaitEstimator,
    is_over_high_water,
)
from cray_infra.api.fastapi.chat_completions.build_chat_completion_response import (
    build_chat_completion_response,
)
from cray_infra.api.fastapi.chat_completions.coalescer import Coalescer
from cray_infra.api.fastapi.chat_completions.heartbeat import (
    stream_with_heartbeat,
)
from cray_infra.api.fastapi.chat_completions.render_chat_template import (
    render_chat_template,
)
from cray_infra.api.fastapi.chat_completions.result_router import (
    ResultRouter,
    get_result_router,
)
from cray_infra.generate.metrics import get_metrics
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


async def chat_completions_via_queue(request: Any) -> StreamingResponse:
    """
    Non-streaming chat completions handler. The router (see
    openai_v1_router.py) is responsible for branching on
    `request.stream`; this function never sees a streaming request.

    `request` is an OpenAI-compatible ChatCompletionRequest
    (vllm.entrypoints.openai.protocol or equivalent). The shape is
    duck-typed: we read `model`, `messages`, `max_tokens`, and
    `temperature`.
    """
    # Admission check first: rejecting an overload request before
    # rendering the chat template avoids paying the (possibly first-
    # use, multi-second) tokenizer load on requests we're going to
    # 429 anyway.
    config = get_config()
    queue_depth = get_queue_depth()
    router = get_result_router()

    if is_over_high_water(
        queue_depth=queue_depth,
        in_flight_count=router.in_flight_count,
        max_num_seqs=config["max_num_seqs"],
        admit_factor=config["chat_admit_factor"],
    ):
        wait = get_wait_estimator().estimate_wait_seconds(
            queue_depth=queue_depth,
            max_num_seqs=config["max_num_seqs"],
        )
        get_metrics().record_chat_rejected_429()
        raise HTTPException(
            status_code=429,
            detail="Server is over capacity; retry after the indicated interval.",
            headers={"Retry-After": str(max(1, int(wait)))},
        )

    rendered_prompt = render_chat_template(
        model=request.model,
        messages=request.messages,
        prompt=None,
    )

    correlation_id = str(uuid4())
    future = router.register(correlation_id)
    get_metrics().record_chat_admitted(correlation_id)

    backend_request = {
        "prompt": rendered_prompt,
        "model": request.model,
        "max_tokens": getattr(request, "max_tokens", None),
        "temperature": getattr(request, "temperature", None),
        # The worker's dispatcher (`async_generate_task` in
        # create_generate_worker.py) only recognises "generate". The
        # rendered_prompt is a string, so it routes through
        # `async_completion_task` to /v1/completions in vLLM — which
        # is what we want, because the chat template is already
        # applied. We rewrap the worker's response into a
        # ChatCompletion shape below.
        "request_type": "generate",
        "correlation_id": correlation_id,
    }

    await get_coalescer().submit(backend_request, correlation_id)

    return StreamingResponse(
        _stream_and_unregister(future, correlation_id, router, request.model),
        media_type="application/json",
    )


def _encode_chat_completion(model: str):
    """Encoder closure for stream_with_heartbeat. The worker hands us
    its flat result dict (request_id, response, error, token_count);
    we rewrap into the OpenAI ChatCompletion shape so the SDK can
    parse it without surprises."""
    import json

    def encode(result):
        wrapped = build_chat_completion_response(result=result, model=model)
        return json.dumps(wrapped).encode("utf-8")

    return encode


async def _stream_and_unregister(
    future,
    correlation_id: str,
    router: ResultRouter,
    model: str,
) -> AsyncIterator[bytes]:
    """
    Wrap the heartbeat stream so the cid is always unregistered on
    completion or generator close (the client-disconnect path).
    """
    try:
        async for chunk in stream_with_heartbeat(
            future, encode_body=_encode_chat_completion(model)
        ):
            yield chunk
    finally:
        router.unregister(correlation_id)
        # If the worker resolved already, this is a no-op (the cid's
        # start_time was popped during record_chat_resolved). If the
        # client disconnected before the worker resolved, this is the
        # only place that decrements chat_in_flight.
        get_metrics().record_chat_unregistered(correlation_id)


# ---------------------------------------------------------------------------
# Singletons / wiring. Tests patch these accessors; production wires once.
# ---------------------------------------------------------------------------


_coalescer: Coalescer | None = None
_wait_estimator: WaitEstimator | None = None


def get_coalescer() -> Coalescer:
    global _coalescer
    if _coalescer is None:
        from cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch import (
            enqueue_coalesced_batch,
        )

        config = get_config()
        _coalescer = Coalescer(
            packing_factor=config.get("chat_coalescer_packing_factor", 10),
            window_seconds=config.get("chat_coalescer_window_ms", 50) / 1000.0,
            bypass_threshold=config.get("chat_coalescer_bypass_threshold", 10),
            flush_callback=enqueue_coalesced_batch,
            queue_depth_provider=get_queue_depth,
        )
    return _coalescer


def get_wait_estimator() -> WaitEstimator:
    global _wait_estimator
    if _wait_estimator is None:
        config = get_config()
        _wait_estimator = WaitEstimator(
            default_batch_latency_seconds=config.get(
                "chat_wait_estimator_default_seconds", 5.0
            ),
            padding=config.get("chat_wait_estimator_padding", 1.5),
            sample_size=config.get("chat_wait_estimator_sample_size", 32),
        )
    return _wait_estimator


def get_queue_depth() -> int:
    """
    Current in-flight inference work (queue + processing). Read from
    the existing Metrics singleton, which already tracks this counter
    for /v1/generate.
    """
    from cray_infra.generate.metrics import get_metrics

    return get_metrics().queue_depth


