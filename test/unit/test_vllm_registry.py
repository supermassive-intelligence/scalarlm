"""Unit tests for the process-wide vLLM serving registry.

The registry is pure Python with no vLLM imports, so it runs in the
lightest test venv. Covers the happy path (set then get) and the
startup race (get before set returns None cleanly so the proxy can fall
back to the HTTP path without raising).
"""

from types import SimpleNamespace

from cray_infra.one_server import vllm_registry


def _fake_app_state(**overrides):
    defaults = dict(
        openai_serving_completion=object(),
        openai_serving_chat=object(),
        engine_client=object(),
        model_config=object(),
        openai_serving_models=object(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_get_before_set_returns_none():
    vllm_registry._reset_for_tests()
    assert vllm_registry.get_vllm_servings() is None


def test_set_then_get_returns_populated_dataclass():
    vllm_registry._reset_for_tests()
    state = _fake_app_state()
    vllm_registry.set_vllm_servings(state)

    servings = vllm_registry.get_vllm_servings()
    assert servings is not None
    assert servings.engine_client is state.engine_client
    assert servings.openai_serving_completion is state.openai_serving_completion
    assert servings.openai_serving_chat is state.openai_serving_chat


def test_set_twice_replaces():
    vllm_registry._reset_for_tests()
    vllm_registry.set_vllm_servings(_fake_app_state())
    second = _fake_app_state()
    vllm_registry.set_vllm_servings(second)

    assert vllm_registry.get_vllm_servings().engine_client is second.engine_client


def test_missing_required_attribute_fails_loudly():
    vllm_registry._reset_for_tests()
    # Drop a REQUIRED serving on purpose — we want a clean AttributeError
    # at startup, not a mysterious failure at first-request time.
    state = _fake_app_state()
    del state.openai_serving_chat

    try:
        vllm_registry.set_vllm_servings(state)
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError when a required serving is missing")


def test_missing_optional_attribute_is_tolerated():
    vllm_registry._reset_for_tests()
    state = _fake_app_state()
    # model_config / engine_client / openai_serving_models are optional —
    # some vLLM fork versions don't set them on app.state.
    del state.model_config
    del state.engine_client

    vllm_registry.set_vllm_servings(state)
    servings = vllm_registry.get_vllm_servings()
    assert servings is not None
    assert servings.openai_serving_completion is state.openai_serving_completion
    assert servings.model_config is None
    assert servings.engine_client is None
