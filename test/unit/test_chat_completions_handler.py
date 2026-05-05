"""
Unit tests for chat_completions_via_queue.

The handler ties together the four foundation pieces (renderer,
admission, router, coalescer) and returns a StreamingResponse backed
by the heartbeat helper. Tests mock the singletons so the flow can be
exercised without uvicorn / SQLite / a real model.

Contract (see docs/openai-chat-completions-queue.md §5):
- Renders the request via render_chat_template before queueing.
- 429 with Retry-After when admission threshold is exceeded.
- Registers a correlation_id with the result router *before*
  submitting to the coalescer so the worker can never resolve a cid
  the router doesn't yet know about.
- Returns a StreamingResponse with media_type=application/json.
- The streaming generator unregisters the cid on completion or
  cancellation (verified by the disconnect test).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cray_infra.api.fastapi.chat_completions import handler as h
from cray_infra.api.fastapi.chat_completions.admission import WaitEstimator
from cray_infra.api.fastapi.chat_completions.result_router import ResultRouter


def _request(messages=None, prompt_text=None, stream=False, **overrides):
    """Build a minimal ChatCompletionRequest-shaped MagicMock."""
    if messages is None and prompt_text is None:
        messages = [{"role": "user", "content": "hi"}]

    req = MagicMock()
    req.model = overrides.get("model", "test-model")
    req.messages = messages
    req.max_tokens = overrides.get("max_tokens", 64)
    req.temperature = overrides.get("temperature", 0.7)
    req.stream = stream
    return req


@pytest.fixture
def fresh_router():
    return ResultRouter()


@pytest.fixture
def fresh_estimator():
    return WaitEstimator(default_batch_latency_seconds=2.0, padding=1.5)


@pytest.fixture
def patched_components(fresh_router, fresh_estimator):
    """
    Patch the four singleton accessors the handler reaches through.
    Returns the mock coalescer and a "queue depth" you can mutate
    per-test.
    """
    coalescer = MagicMock()
    coalescer.submit = AsyncMock()

    queue_depth_holder = {"value": 0}
    fake_config = {
        "max_num_seqs": 256,
        "chat_admit_factor": 4,
    }

    with patch.object(h, "get_result_router", return_value=fresh_router), \
         patch.object(h, "get_coalescer", return_value=coalescer), \
         patch.object(h, "get_wait_estimator", return_value=fresh_estimator), \
         patch.object(h, "get_queue_depth", side_effect=lambda: queue_depth_holder["value"]), \
         patch.object(h, "get_config", return_value=fake_config), \
         patch.object(h, "_resolve_model", side_effect=lambda req, cfg: req or "test-model"), \
         patch.object(h, "render_chat_template", return_value="rendered-prompt"):
        yield {
            "coalescer": coalescer,
            "queue_depth": queue_depth_holder,
            "router": fresh_router,
            "estimator": fresh_estimator,
            "config": fake_config,
        }


@pytest.mark.asyncio
async def test_returns_streaming_response_under_threshold(patched_components):
    response = await h.chat_completions_via_queue(_request())
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "application/json"


@pytest.mark.asyncio
async def test_renders_messages_through_chat_template(patched_components):
    with patch.object(
        h, "render_chat_template", return_value="USER: hi\nASSISTANT: "
    ) as render:
        await h.chat_completions_via_queue(
            _request(messages=[{"role": "user", "content": "hi"}])
        )
    render.assert_called_once()
    kwargs = render.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["prompt"] is None


@pytest.mark.asyncio
async def test_registers_correlation_id_before_submitting(patched_components):
    """
    Submission must happen *after* router.register so the worker can
    never resolve a cid that doesn't exist yet.
    """
    submit_order: list = []

    coalescer = patched_components["coalescer"]

    async def record_submit(req, cid):
        submit_order.append(("submit", cid, patched_components["router"].in_flight_count))

    coalescer.submit = AsyncMock(side_effect=record_submit)

    with patch.object(h, "render_chat_template", return_value="rendered"):
        await h.chat_completions_via_queue(_request())

    assert len(submit_order) == 1
    _, _, in_flight_at_submit = submit_order[0]
    # The router was incremented before the submit ran.
    assert in_flight_at_submit == 1


@pytest.mark.asyncio
async def test_correlation_id_passed_to_coalescer_matches_request_payload(patched_components):
    """The correlation_id is in the request dict AND the coalescer sees it as the second arg."""
    coalescer = patched_components["coalescer"]

    captured: list = []

    async def capture(req, cid):
        captured.append((req, cid))

    coalescer.submit = AsyncMock(side_effect=capture)

    with patch.object(h, "render_chat_template", return_value="rendered"):
        await h.chat_completions_via_queue(_request())

    assert len(captured) == 1
    req, cid = captured[0]
    assert req["correlation_id"] == cid
    assert req["prompt"] == "rendered"
    # Worker's dispatcher only recognises "generate"; the rendered
    # prompt is a string so it routes through async_completion_task,
    # and the handler rewraps the result into ChatCompletion shape.
    assert req["request_type"] == "generate"


@pytest.mark.asyncio
async def test_429_when_over_high_water(patched_components):
    """queue_depth > 4 × max_num_seqs (1024) trips the threshold."""
    patched_components["queue_depth"]["value"] = 1025

    with pytest.raises(HTTPException) as exc_info:
        await h.chat_completions_via_queue(_request())

    assert exc_info.value.status_code == 429
    retry_after = exc_info.value.headers.get("Retry-After")
    assert retry_after is not None
    assert int(retry_after) >= 1


@pytest.mark.asyncio
async def test_429_does_not_register_correlation_id(patched_components):
    """A rejected request must leave no leak in the router."""
    patched_components["queue_depth"]["value"] = 9999

    with pytest.raises(HTTPException):
        await h.chat_completions_via_queue(_request())

    assert patched_components["router"].in_flight_count == 0


# ---------------------------------------------------------------------------
# Pre-admission length check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_400_when_prompt_plus_max_tokens_exceeds_max_model_length(
    patched_components,
):
    """
    A request whose prompt + max_tokens > max_model_length must be
    rejected up front with HTTP 400. Without this check vLLM queues
    it forever — the production stuck-request symptom.
    """
    patched_components["config"]["max_model_length"] = 100

    with patch.object(h, "count_prompt_tokens", return_value=80):
        with pytest.raises(HTTPException) as exc_info:
            await h.chat_completions_via_queue(_request(max_tokens=50))

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert "80" in detail and "50" in detail and "100" in detail


@pytest.mark.asyncio
async def test_too_long_request_does_not_register_correlation_id(
    patched_components,
):
    """The 400 path must leak nothing into the router or coalescer —
    same contract as the 429 over-capacity path."""
    patched_components["config"]["max_model_length"] = 100
    coalescer = patched_components["coalescer"]

    with patch.object(h, "count_prompt_tokens", return_value=200):
        with pytest.raises(HTTPException):
            await h.chat_completions_via_queue(_request(max_tokens=10))

    assert patched_components["router"].in_flight_count == 0
    coalescer.submit.assert_not_called()


@pytest.mark.asyncio
async def test_length_check_passes_when_within_threshold(patched_components):
    """Boundary case: prompt + max_tokens == max_model_length is fine."""
    patched_components["config"]["max_model_length"] = 100

    with patch.object(h, "count_prompt_tokens", return_value=80):
        response = await h.chat_completions_via_queue(_request(max_tokens=20))

    # No exception → got the StreamingResponse back.
    assert response is not None


@pytest.mark.asyncio
async def test_length_check_skipped_when_no_cap_configured(patched_components):
    """
    The default fixture has no max_model_length. count_prompt_tokens
    must NOT be called on the hot path when there's no cap to enforce
    — saves a tokenizer pass per request on misconfigured pods.
    """
    with patch.object(h, "count_prompt_tokens") as count:
        await h.chat_completions_via_queue(_request())

    count.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_model — None / "latest" / explicit / unknown
# ---------------------------------------------------------------------------


def test_resolve_model_none_falls_back_to_config_default():
    """
    None gets the deployment's default. This was the production crash
    surfaced by the inference browser: a missing-model request used to
    crash inside `AutoTokenizer.from_pretrained(None)` with a
    HuggingFace 401 about a non-existent repo named "None".
    """
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "default-m"
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model(None, cfg) == "default-m"
    fake_manager.find_model.assert_called_once_with("default-m")


def test_resolve_model_empty_string_falls_back_to_config_default():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "default-m"
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model("", cfg) == "default-m"


def test_resolve_model_latest_uses_get_latest_model():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "training-job-abc"
    with patch(
        "cray_infra.training.get_latest_model.get_latest_model",
        return_value="training-job-abc",
    ), patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model("latest", cfg) == "training-job-abc"


def test_resolve_model_explicit_validates_against_manager():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "explicit-m"
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model("explicit-m", cfg) == "explicit-m"
    fake_manager.find_model.assert_called_once_with("explicit-m")


def test_resolve_model_unknown_raises_404():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = None
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        with pytest.raises(HTTPException) as exc:
            h._resolve_model("nope-not-here", cfg)
    assert exc.value.status_code == 404
    assert "nope-not-here" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_disconnect_during_stream_unregisters_cid(patched_components):
    """
    The streaming generator's `finally` must unregister the cid even
    if the generator is closed early (client disconnect → FastAPI
    cancels the body iterator).
    """
    with patch.object(h, "render_chat_template", return_value="rendered"):
        response = await h.chat_completions_via_queue(_request())

    router = patched_components["router"]
    assert router.in_flight_count == 1

    # Drive the generator partially then close it (mimics client
    # disconnect: FastAPI calls `aclose()` on the iterator).
    gen = response.body_iterator
    await gen.__anext__()  # consume one heartbeat
    await gen.aclose()

    assert router.in_flight_count == 0
