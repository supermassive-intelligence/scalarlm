"""Unit tests for the Phase 30 response cache on /v1/completions and
/v1/chat/completions.

The cache hashes the filtered params dict and persists the response JSON
to {upload_base_path}/openai_cache/{sha256}.json. These tests exercise
the helpers directly rather than standing up the FastAPI app.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def router_cache_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SCALARLM_OPENAI_CACHE", "1")
    from cray_infra.api.fastapi.routers import openai_v1_helpers as r
    importlib.reload(r)
    return r, {"upload_base_path": str(tmp_path)}


@pytest.fixture
def router_cache_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("SCALARLM_OPENAI_CACHE", raising=False)
    from cray_infra.api.fastapi.routers import openai_v1_helpers as r
    importlib.reload(r)
    return r, {"upload_base_path": str(tmp_path)}


# ---- _cache_key deterministic + key-sensitive -----------------------------


def test_cache_key_deterministic_across_dict_order(router_cache_enabled):
    r, _ = router_cache_enabled
    a = {"model": "m", "prompt": ["p1", "p2"], "max_tokens": 16, "temperature": 0.0}
    b = {"temperature": 0.0, "max_tokens": 16, "prompt": ["p1", "p2"], "model": "m"}
    assert r._cache_key(a) == r._cache_key(b)


def test_cache_key_changes_when_any_keyed_field_changes(router_cache_enabled):
    r, _ = router_cache_enabled
    base = {"model": "m", "prompt": ["p"], "max_tokens": 16, "temperature": 0.0}
    for field, new in [
        ("model", "m2"),
        ("prompt", ["p2"]),
        ("max_tokens", 17),
        ("temperature", 0.1),
    ]:
        mutated = dict(base)
        mutated[field] = new
        assert r._cache_key(base) != r._cache_key(mutated), (
            f"hash did not change when {field} changed"
        )


def test_cache_key_ignores_non_cache_fields(router_cache_enabled):
    """Fields outside _OPENAI_CACHE_KEYS (e.g. stream, seed) must not
    influence the hash — otherwise a stream-off/stream-on pair would
    mint different keys for the same inference."""
    r, _ = router_cache_enabled
    a = {"model": "m", "prompt": ["p"], "max_tokens": 16}
    b = {"model": "m", "prompt": ["p"], "max_tokens": 16, "stream": True}
    assert r._cache_key(a) == r._cache_key(b)


# ---- _cache_lookup gating --------------------------------------------------


def test_cache_lookup_returns_none_when_disabled(router_cache_disabled, tmp_path):
    """Even if a matching file is on disk, lookup should return None when
    SCALARLM_OPENAI_CACHE is unset."""
    r, config = router_cache_disabled
    cache_dir = tmp_path / "openai_cache"
    cache_dir.mkdir()
    params = {"model": "m", "prompt": ["p"]}
    key = r._cache_key(params)
    (cache_dir / f"{key}.json").write_text('{"choices": []}')
    assert r._cache_lookup(params, config) is None


def test_cache_lookup_returns_none_for_streaming(router_cache_enabled):
    """Streaming requests bypass the cache entirely."""
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["p"], "stream": True}
    nonstream = dict(params)
    nonstream.pop("stream")
    r._cache_store(nonstream, {"choices": [{"index": 0}]}, config)
    assert r._cache_lookup(params, config) is None


def test_cache_lookup_miss_returns_none(router_cache_enabled):
    r, config = router_cache_enabled
    assert r._cache_lookup({"model": "m", "prompt": ["nope"]}, config) is None


# ---- _cache_lookup error-path handling ------------------------------------


def test_cache_lookup_handles_corrupted_json(router_cache_enabled, tmp_path):
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["p"]}
    key = r._cache_key(params)
    cache_file = Path(config["upload_base_path"]) / "openai_cache" / f"{key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{ not json }")
    assert r._cache_lookup(params, config) is None


def test_cache_lookup_handles_file_io_errors(router_cache_enabled, tmp_path):
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["p"]}
    key = r._cache_key(params)
    cache_file = Path(config["upload_base_path"]) / "openai_cache" / f"{key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"choices": []}')
    with patch("builtins.open", side_effect=PermissionError("no read")):
        assert r._cache_lookup(params, config) is None


# ---- _cache_store roundtrip + refusal cases -------------------------------


def test_cache_store_then_lookup_roundtrip(router_cache_enabled):
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["hi"], "max_tokens": 16}
    body = {
        "id": "cmpl-abc",
        "object": "text_completion",
        "choices": [{"index": 0, "text": "hello", "finish_reason": "stop"}],
        "usage": {"total_tokens": 3},
    }
    r._cache_store(params, body, config)
    hit = r._cache_lookup(params, config)
    assert hit == body


def test_cache_store_is_atomic(router_cache_enabled):
    """The store must use os.replace for atomicity so a crash mid-write
    can't leave a partial file that a concurrent reader would see."""
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["p"]}
    body = {"choices": [{"text": "ok"}]}
    with patch("os.replace") as mock_replace:
        r._cache_store(params, body, config)
        assert mock_replace.called
        src, dst = mock_replace.call_args[0]
        assert src.endswith(".tmp")
        assert dst.endswith(".json")


def test_cache_store_refuses_non_dict_bodies(router_cache_enabled):
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["p"]}
    r._cache_store(params, "not a dict", config)
    assert r._cache_lookup(params, config) is None


def test_cache_store_refuses_bodies_without_choices(router_cache_enabled):
    """Error-shaped responses (no ``choices``) must not be cached."""
    r, config = router_cache_enabled
    params = {"model": "m", "prompt": ["hi"]}
    r._cache_store(params, {"error": {"message": "boom"}}, config)
    assert r._cache_lookup(params, config) is None


def test_cache_store_no_op_when_disabled(router_cache_disabled, tmp_path):
    r, config = router_cache_disabled
    params = {"model": "m", "prompt": ["hi"]}
    r._cache_store(params, {"choices": [{"index": 0}]}, config)
    cache_dir = tmp_path / "openai_cache"
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


# ---- _parse_env_bool robust env parsing ----------------------------------


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("True", True), ("TRUE", True),
    ("yes", True), ("YES", True), ("on", True), ("ON", True),
    ("  true  ", True),  # surrounding whitespace tolerated
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("", False), ("garbage", False), ("2", False),
])
def test_parse_env_bool_accepts_common_forms(val, expected, monkeypatch):
    """The old ``bool(int(os.environ.get(...)))`` pattern crashed on
    SCALARLM_OPENAI_CACHE=true (ValueError from int("true")). The
    robust parser accepts any common boolean form and falls back to
    default on anything else, matching how most config libraries handle
    env-var booleans."""
    monkeypatch.setenv("_TEST_ENV_BOOL", val)
    from cray_infra.api.fastapi.routers import openai_v1_helpers as r
    assert r._parse_env_bool("_TEST_ENV_BOOL") is expected


def test_parse_env_bool_unset_uses_default(monkeypatch):
    monkeypatch.delenv("_TEST_ENV_BOOL", raising=False)
    from cray_infra.api.fastapi.routers import openai_v1_helpers as r
    assert r._parse_env_bool("_TEST_ENV_BOOL") is False
    assert r._parse_env_bool("_TEST_ENV_BOOL", default=True) is True


# ---- _cache_key_from_request consistency ---------------------------------


def test_cache_key_from_request_consistency(router_cache_enabled):
    """The fast-path `_cache_key_from_request` must agree with the
    model-dump-based `_cache_key` — otherwise a cold direct-path call
    and a warm queue-routed call on the same params would mint
    different keys and silently miss the cache."""
    r, _ = router_cache_enabled
    req = SimpleNamespace(
        model="gpt-4",
        prompt=["hello"],
        temperature=0.7,
        max_tokens=100,
        tools=[{"type": "function"}],
        ignored_field="secret",
    )
    params = {
        "model": req.model,
        "prompt": req.prompt,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "tools": req.tools,
    }
    assert r._cache_key_from_request(req) == r._cache_key(params)
