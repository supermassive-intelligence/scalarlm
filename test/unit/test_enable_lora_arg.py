"""Unit tests for the Phase 31b conditional --enable-lora arg.

Pure-inference pods can opt out via SCALARLM_ENABLE_LORA=false, which
drops the --enable-lora vLLM CLI arg. This avoids wrapping every layer
in a LoRA shim (perf win, plus sidesteps the fused_moe wrapper bug on
some vllm-fork builds).
"""

from __future__ import annotations

from cray_infra.one_server.vllm_cli_args import build_vllm_cli_args


def _base_config(**overrides):
    cfg = {
        "dtype": "auto",
        "gpu_memory_utilization": 0.85,
        "max_log_length": 100,
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": None,
        "enable_lora": True,
    }
    cfg.update(overrides)
    return cfg


def test_enable_lora_true_includes_flag():
    args = build_vllm_cli_args(_base_config(enable_lora=True))
    assert "--enable-lora" in args


def test_enable_lora_false_omits_flag():
    args = build_vllm_cli_args(_base_config(enable_lora=False))
    assert "--enable-lora" not in args


def test_enable_lora_default_true():
    """Missing key in config defaults to True (back-compat with pods
    that haven't re-synced their config yaml)."""
    cfg = _base_config()
    del cfg["enable_lora"]
    args = build_vllm_cli_args(cfg)
    assert "--enable-lora" in args


def test_other_required_flags_always_present():
    """The LoRA toggle must not accidentally drop other required args."""
    args = build_vllm_cli_args(_base_config(enable_lora=False))
    assert "--trust-remote-code" in args
    assert "--enable-auto-tool-choice" in args
    assert "--tool-call-parser=hermes" in args
    assert any(a.startswith("--tensor-parallel-size=") for a in args)
    assert any(a.startswith("--gpu-memory-utilization=") for a in args)


def test_limit_mm_per_prompt_included_when_set():
    args = build_vllm_cli_args(_base_config(limit_mm_per_prompt='{"image":2}'))
    assert "--limit-mm-per-prompt={\"image\":2}" in args


def test_limit_mm_per_prompt_omitted_when_none():
    args = build_vllm_cli_args(_base_config(limit_mm_per_prompt=None))
    assert not any(a.startswith("--limit-mm-per-prompt=") for a in args)
