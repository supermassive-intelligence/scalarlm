"""Unit tests for the Phase 31 bulk queue-route fast-path.

Array /v1/completions requests with >= SCALARLM_QUEUE_ROUTE_THRESHOLD
string prompts are routed through the /v1/generate queue worker. These
tests verify the routing logic and cache key consistency for complex
request shapes.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def router_threshold_100(monkeypatch):
    monkeypatch.setenv("SCALARLM_QUEUE_ROUTE_THRESHOLD", "100")
    from cray_infra.api.fastapi.routers import openai_v1_helpers as r
    importlib.reload(r)
    return r


@pytest.fixture
def router_threshold_off(monkeypatch):
    monkeypatch.delenv("SCALARLM_QUEUE_ROUTE_THRESHOLD", raising=False)
    from cray_infra.api.fastapi.routers import openai_v1_helpers as r
    importlib.reload(r)
    return r


def _req(**kwargs):
    """Minimal stand-in for CompletionRequest. The fast-path only reads
    attributes, so a plain namespace suffices."""
    defaults = {
        "prompt": ["hi"] * 100,
        "stream": False,
        "model": "m",
        "max_tokens": 16,
        "temperature": 0.0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---- _should_route_via_queue_fast: accept ---------------------------------


def test_accept_string_list_at_threshold(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt=["p"] * 100)
    assert r._should_route_via_queue_fast(req) is True


def test_accept_large_string_list(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt=["p"] * 1000)
    assert r._should_route_via_queue_fast(req) is True


# ---- _should_route_via_queue_fast: reject ---------------------------------


def test_reject_when_threshold_unset(router_threshold_off):
    r = router_threshold_off
    req = _req(prompt=["p"] * 1000)
    assert r._should_route_via_queue_fast(req) is False


def test_reject_when_streaming(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt=["p"] * 1000, stream=True)
    assert r._should_route_via_queue_fast(req) is False


def test_reject_scalar_string_prompt(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt="one prompt")
    assert r._should_route_via_queue_fast(req) is False


def test_reject_empty_list(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt=[])
    assert r._should_route_via_queue_fast(req) is False


def test_reject_shorter_than_threshold(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt=["p"] * 99)
    assert r._should_route_via_queue_fast(req) is False


def test_reject_token_id_list(router_threshold_100):
    """list[int] is the OpenAI token-id prompt form — the queue worker
    can't handle it."""
    r = router_threshold_100
    req = _req(prompt=list(range(1000)))
    assert r._should_route_via_queue_fast(req) is False


def test_reject_list_of_token_lists(router_threshold_100):
    """list[list[int]] is the multi-prompt token-id form."""
    r = router_threshold_100
    req = _req(prompt=[[1, 2, 3]] * 200)
    assert r._should_route_via_queue_fast(req) is False


def test_reject_mixed_list(router_threshold_100):
    r = router_threshold_100
    req = _req(prompt=["valid"] * 500 + [[1, 2, 3]])
    assert r._should_route_via_queue_fast(req) is False


# ---- reject on params the queue-route translation drops ------------------


def test_reject_when_top_p_set(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(top_p=0.9)) is False


def test_reject_when_stop_set(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(stop=["<|end|>"])) is False


def test_reject_when_seed_set(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(seed=42)) is False


def test_reject_when_presence_penalty_nonzero(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(presence_penalty=0.5)) is False


def test_reject_when_frequency_penalty_nonzero(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(frequency_penalty=0.5)) is False


def test_accept_when_penalties_at_default(router_threshold_100):
    """Default 0.0 penalties must not trigger a rejection — that's the
    pydantic default, not a user-set value."""
    r = router_threshold_100
    assert (
        r._should_route_via_queue_fast(
            _req(presence_penalty=0.0, frequency_penalty=0.0)
        ) is True
    )


def test_reject_when_response_format_set(router_threshold_100):
    r = router_threshold_100
    assert (
        r._should_route_via_queue_fast(
            _req(response_format={"type": "json_object"})
        ) is False
    )


def test_reject_when_n_greater_than_1(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(n=3)) is False


def test_accept_when_n_is_1(router_threshold_100):
    r = router_threshold_100
    assert r._should_route_via_queue_fast(_req(n=1)) is True


# ---- Cache Key Consistency: _cache_key_from_request vs _cache_key(params) --


def test_cache_key_consistency_simple(router_threshold_100):
    """Ensure fast-path key matches direct-path key for simple params."""
    r = router_threshold_100
    req = _req(prompt=["p1", "p2"], max_tokens=32, temperature=0.7)
    
    # Simulate what _filter_params(request.model_dump()) would produce
    params = {
        "model": req.model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    
    assert r._cache_key_from_request(req) == r._cache_key(params)


def test_cache_key_consistency_complex_fields(router_threshold_100):
    """Verify consistency when complex fields like tools are present."""
    r = router_threshold_100
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"location": {"type": "string"}}
            }
        }
    ]
    req = _req(tools=tools, tool_choice="auto")
    
    params = {
        "model": req.model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "tools": tools,
        "tool_choice": "auto"
    }
    
    assert r._cache_key_from_request(req) == r._cache_key(params)


def test_cache_key_ignores_untracked_request_attributes(router_threshold_100):
    """Attributes on the request object that aren't in _OPENAI_CACHE_KEYS
    should not affect the cache key produced by the fast-path."""
    r = router_threshold_100
    req_a = _req(prompt=["p"])
    req_b = _req(prompt=["p"])
    req_b.internal_debug_flag = True
    
    assert r._cache_key_from_request(req_a) == r._cache_key_from_request(req_b)
