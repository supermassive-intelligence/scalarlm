import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune_sweep"))

import pytest

from run_finetune_sweep import (
    ADAPTER_NOT_LOADED,
    BAD_CHECKPOINT,
    DEFAULT_MANIFEST,
    NO_MEMORIZATION,
    NON_FAILING_OUTCOMES,
    PASS,
    RESTART_FAILED,
    SKIPPED,
    TRAIN_FAILED,
    TRAIN_TIMEOUT,
    build_dataset,
    checkpoint_lora_keys_ok,
    classify_result,
    filter_models,
    gate_model,
    load_manifest,
)


def test_load_manifest_has_expected_top_level_keys():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert set(manifest) >= {
        "dataset", "golden_prompt", "expected_output",
        "train_args_defaults", "targets", "models",
    }


def test_non_failing_outcomes_set():
    assert NON_FAILING_OUTCOMES == {PASS, SKIPPED, NO_MEMORIZATION}
    assert RESTART_FAILED not in NON_FAILING_OUTCOMES
    assert TRAIN_FAILED not in NON_FAILING_OUTCOMES
    assert TRAIN_TIMEOUT not in NON_FAILING_OUTCOMES
    assert BAD_CHECKPOINT not in NON_FAILING_OUTCOMES
    assert ADAPTER_NOT_LOADED not in NON_FAILING_OUTCOMES


def test_filter_models_no_filter_returns_all():
    models = [{"id": "a"}, {"id": "b"}]
    assert filter_models(models, None) == models


def test_filter_models_subset():
    models = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert filter_models(models, ["b"]) == [{"id": "b"}]


def test_build_dataset_repeats_examples():
    spec = {"examples": [{"input": "x", "output": "y"}], "repeat": 3}
    assert build_dataset(spec) == [{"input": "x", "output": "y"}] * 3


def test_build_dataset_default_repeat_is_one():
    spec = {"examples": [{"input": "x", "output": "y"}]}
    assert build_dataset(spec) == [{"input": "x", "output": "y"}]


@pytest.mark.parametrize("model,target,free_gb,expected_ok", [
    ({"cpu_ok": True}, "cpu", [], True),
    ({}, "cpu", [], False),
    ({"adapters": {"lora": {"gate_gb": 8}}}, "cuda", [10.0], True),
    ({"adapters": {"lora": {"gate_gb": 8}}}, "cuda", [4.0], False),
    ({"adapters": {"lora": {"gate_gb": 8}}}, "cuda", [], False),
])
def test_gate_model(model, target, free_gb, expected_ok):
    ok, _reason = gate_model(model, target, free_gb)
    assert ok is expected_ok


@pytest.mark.parametrize("keys,expected", [
    (None, False),
    ([], False),
    (["model.layers.0.mlp.lora_A.weight"], False),
    (["model.layers.0.mlp.lora_B.weight"], False),
    (["model.layers.0.mlp.lora_A.weight", "model.layers.0.mlp.lora_B.weight"], True),
])
def test_checkpoint_lora_keys_ok(keys, expected):
    assert checkpoint_lora_keys_ok(keys) is expected


LORA_KEYS = ["model.layers.0.mlp.lora_A.weight", "model.layers.0.mlp.lora_B.weight"]


@pytest.mark.parametrize("train_status,checkpoint_keys,adapter_loaded,memorized,expected", [
    ("FAILED", None, False, False, TRAIN_FAILED),
    ("CANCELLED", None, False, False, TRAIN_FAILED),
    ("TIMEOUT", None, False, False, TRAIN_TIMEOUT),
    ("COMPLETED", None, False, False, BAD_CHECKPOINT),
    ("COMPLETED", [], False, False, BAD_CHECKPOINT),
    ("COMPLETED", LORA_KEYS, False, False, ADAPTER_NOT_LOADED),
    ("COMPLETED", LORA_KEYS, True, False, NO_MEMORIZATION),
    ("COMPLETED", LORA_KEYS, True, True, PASS),
])
def test_classify_result(train_status, checkpoint_keys, adapter_loaded, memorized, expected):
    assert classify_result(train_status, checkpoint_keys, adapter_loaded, memorized) == expected
