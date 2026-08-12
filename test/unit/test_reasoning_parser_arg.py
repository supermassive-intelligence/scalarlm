"""Focused tests for vLLM reasoning-parser startup configuration."""

from __future__ import annotations

import pytest

from cray_infra.one_server.vllm_cli_args import (
    build_vllm_cli_args,
    resolve_reasoning_parser,
)


@pytest.fixture(autouse=True)
def _clear_scalarlm_vllm_args_env(monkeypatch):
    monkeypatch.delenv("SCALARLM_VLLM_ARGS", raising=False)


def _base_config(**overrides):
    config = {
        "dtype": "auto",
        "gpu_memory_utilization": 0.85,
        "max_log_length": 100,
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": None,
        "enable_lora": False,
        "enable_tokenformer": True,
        "model": "",
        "reasoning_parser": None,
    }
    config.update(overrides)
    return config


def _reasoning_parser_args(config):
    return [
        arg
        for arg in build_vllm_cli_args(config)
        if arg.startswith("--reasoning-parser=")
    ]


def test_explicit_parser_is_forwarded():
    assert _reasoning_parser_args(_base_config(reasoning_parser="deepseek_r1")) == [
        "--reasoning-parser=deepseek_r1"
    ]


def test_qwen3_checkpoint_is_detected_case_insensitively():
    for model in ("Qwen/Qwen3-32B", "qwen/qWeN3.6-35B-A3B-FP8"):
        assert _reasoning_parser_args(_base_config(model=model)) == [
            "--reasoning-parser=qwen3"
        ]


def test_unrecognized_checkpoint_does_not_guess_parser():
    assert not _reasoning_parser_args(
        _base_config(model="meta-llama/Llama-3.1-8B-Instruct")
    )


def test_explicit_parser_overrides_detection():
    assert _reasoning_parser_args(
        _base_config(model="Qwen/Qwen3-32B", reasoning_parser="qwen3_xml")
    ) == ["--reasoning-parser=qwen3_xml"]


def test_explicit_empty_string_disables_detection():
    assert not _reasoning_parser_args(
        _base_config(model="Qwen/Qwen3-32B", reasoning_parser="")
    )


def test_missing_config_key_still_allows_detection():
    config = _base_config(model="Qwen/Qwen3-32B")
    del config["reasoning_parser"]
    assert _reasoning_parser_args(config) == ["--reasoning-parser=qwen3"]


def test_none_model_is_safe():
    assert not _reasoning_parser_args(_base_config(model=None))


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    ],
)
def test_known_non_reasoning_qwen3_variants_are_excluded(model):
    assert not _reasoning_parser_args(_base_config(model=model))


def test_exclusions_are_scoped_to_checkpoint_name():
    for model in ("coder-team/Qwen3-32B", "/models/next/Qwen3-32B"):
        assert _reasoning_parser_args(_base_config(model=model)) == [
            "--reasoning-parser=qwen3"
        ]


def test_positive_match_is_scoped_to_checkpoint_name():
    for model in ("qwen3-team/Llama-3.1-8B", "/models/qwen3/Llama-3.1-8B"):
        assert not _reasoning_parser_args(_base_config(model=model))


def test_env_override_wins_over_detection(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--reasoning-parser=deepseek_r1")
    assert _reasoning_parser_args(_base_config(model="Qwen/Qwen3-32B")) == [
        "--reasoning-parser=deepseek_r1"
    ]


def test_env_override_wins_over_explicit_config(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--reasoning-parser=deepseek_r1")
    assert _reasoning_parser_args(_base_config(reasoning_parser="qwen3_xml")) == [
        "--reasoning-parser=deepseek_r1"
    ]


def test_unrelated_env_override_does_not_change_parser(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--max-num-seqs=64")
    assert (
        resolve_reasoning_parser(_base_config(reasoning_parser="qwen3_xml"))
        == "qwen3_xml"
    )


def test_space_separated_env_override_is_supported(monkeypatch):
    monkeypatch.setenv(
        "SCALARLM_VLLM_ARGS", "--max-num-seqs=64 --reasoning-parser deepseek_r1"
    )
    assert (
        resolve_reasoning_parser(_base_config(reasoning_parser="qwen3_xml"))
        == "deepseek_r1"
    )


def test_last_valid_env_override_wins(monkeypatch):
    monkeypatch.setenv(
        "SCALARLM_VLLM_ARGS",
        "--reasoning-parser=first --reasoning-parser second",
    )
    assert resolve_reasoning_parser(_base_config()) == "second"


def test_malformed_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--reasoning-parser --max-num-seqs=64")
    assert (
        resolve_reasoning_parser(_base_config(reasoning_parser="qwen3_xml"))
        == "qwen3_xml"
    )
