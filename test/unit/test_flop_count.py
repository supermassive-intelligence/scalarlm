"""Unit tests for HuggingFace-config-based FLOP estimation.

Phase 3b of the OpenAI-API enhancement plan: the proxy needs a FLOP estimate
that matches Path A's worker-side formula so the shared metrics counter is
coherent across both surfaces. The module lives in
``cray_infra.generate.flop_count``; these tests pin its formula and cache
behaviour.
"""

from types import SimpleNamespace
from unittest import mock

from cray_infra.generate import flop_count as flops


def _hf_config(**overrides):
    """Build a minimal HF-style config stub. Fields mirror what Llama/Qwen
    put on their config objects — callers only need to override what they
    care about."""
    defaults = dict(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_compute_uses_standard_formula():
    cfg = _hf_config()
    result = flops._compute(cfg)

    # head_size = 4096 / 32 = 128
    # attention_per_layer = q(4096*4096) + kv(4096*32*128*2) + qk(32*128)
    #                     + av(32*128) + o(4096*4096)
    expected_attention = (
        4096 * 4096          # q_proj
        + 4096 * (32 * 128 * 2)  # kv_proj (K and V)
        + 32 * 128           # qk
        + 32 * 128           # av
        + 4096 * 4096        # o_proj
    )
    expected_mlp = 4096 * 11008 + 11008 * 4096
    expected_embedding = 4096 * 32000
    expected_output = 4096 * 32000
    expected = (
        32 * (expected_attention + expected_mlp)
        + expected_embedding
        + expected_output
    )

    assert result == expected


def test_compute_handles_gqa_via_num_key_value_heads():
    # Qwen3/Llama3-style grouped-query attention — kv heads << attention heads.
    full_gqa = flops._compute(_hf_config(num_key_value_heads=32))
    grouped = flops._compute(_hf_config(num_key_value_heads=8))

    # GQA shrinks KV projection, so total FLOPs must drop.
    assert grouped < full_gqa


def test_compute_uses_explicit_head_dim_when_present():
    # Some configs set head_dim explicitly rather than computing from
    # hidden_size / num_attention_heads. The formula must honour it.
    cfg = _hf_config(head_dim=64)
    result = flops._compute(cfg)

    # With head_dim=64 (half of the implicit 128) KV proj and attention
    # products all shrink — total must be strictly smaller than the default.
    baseline = flops._compute(_hf_config())
    assert result < baseline


def test_resolve_text_config_follows_multimodal_wrapper():
    inner = _hf_config()
    outer = SimpleNamespace(text_config=inner)

    assert flops._resolve_text_config(outer) is inner


def test_resolve_text_config_falls_back_to_outer():
    cfg = _hf_config()

    assert flops._resolve_text_config(cfg) is cfg


def test_compute_flops_per_token_caches_by_name():
    flops._clear_cache_for_tests()
    cfg = _hf_config()
    load = mock.Mock(return_value=cfg)

    with mock.patch.object(flops, "_hf_config_loader", load), mock.patch.object(
        flops, "_compute", return_value=123
    ) as compute:
        first = flops.compute_flops_per_token("acme/base")
        second = flops.compute_flops_per_token("acme/base")

    assert first == 123
    assert second == 123
    assert compute.call_count == 1  # cached on second call
    assert load.call_count == 1


def test_compute_flops_per_token_returns_zero_on_load_failure():
    flops._clear_cache_for_tests()
    load = mock.Mock(side_effect=OSError("no such model"))

    with mock.patch.object(flops, "_hf_config_loader", load):
        result = flops.compute_flops_per_token("does-not-exist")

    assert result == 0


def test_compute_flops_per_token_end_to_end_with_real_formula():
    """Smoke test: loader returns a realistic HF-shaped config, function
    multiplies through and caches. Catches integration glitches that the
    component-level tests above would miss.
    """
    flops._clear_cache_for_tests()
    cfg = _hf_config()
    load = mock.Mock(return_value=cfg)

    with mock.patch.object(flops, "_hf_config_loader", load):
        result = flops.compute_flops_per_token("acme/base")

    assert result == flops._compute(cfg)
    assert result > 0
