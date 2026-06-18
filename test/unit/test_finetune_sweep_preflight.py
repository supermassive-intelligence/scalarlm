"""Unit tests for the fine-tune sweep dry-test: the in-sweep ADAPTER_NO_OP
discriminator (run_finetune_sweep.classify_result) and the offline preflight's
torch-free seams (preflight.py). The two-pass normalization fidelity is verified
in-image on the box, not here (no torch/vllm on the host)."""

import importlib.util
import sys
from pathlib import Path

_SWEEP_DIR = Path(__file__).resolve().parents[1] / "finetune_sweep"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SWEEP_DIR / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rfs = _load("run_finetune_sweep")
pf = _load("preflight")


# --- Cycle A: classify_result ADAPTER_NO_OP branch + precedence ---

def test_classify_result_no_op_not_memorized_is_adapter_no_op():
    outcome = rfs.classify_result(
        train_status="COMPLETED",
        checkpoint_keys=["base.layers.0.self_attn.q_proj.lora_A.weight",
                         "base.layers.0.self_attn.q_proj.lora_B.weight"],
        adapter_loaded=True,
        memorized=False,
        adapter_is_noop=True,
    )
    assert outcome == rfs.ADAPTER_NO_OP


def test_classify_result_no_op_but_memorized_is_pass():
    # Byte-identical to baseline yet the golden string is present (degenerate,
    # but precedence must favor the memorization signal): not a no-op failure.
    keys = ["m.lora_A.weight", "m.lora_B.weight"]
    outcome = rfs.classify_result("COMPLETED", keys, adapter_loaded=True,
                                  memorized=True, adapter_is_noop=True)
    assert outcome == rfs.PASS


def test_classify_result_applied_but_not_memorized_is_no_memorization():
    keys = ["m.lora_A.weight", "m.lora_B.weight"]
    outcome = rfs.classify_result("COMPLETED", keys, adapter_loaded=True,
                                  memorized=False, adapter_is_noop=False)
    assert outcome == rfs.NO_MEMORIZATION


# --- Cycle B: synthesize_lora_keys (pure, torch-free) ---

def test_synthesize_lora_keys_one_layer_standard_targets():
    keys = pf.synthesize_lora_keys(n_layers=1)
    assert set(keys) == {
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.self_attn.k_proj.lora_A.default.weight",
        "model.layers.0.self_attn.v_proj.lora_A.default.weight",
        "model.layers.0.self_attn.o_proj.lora_A.default.weight",
        "model.layers.0.mlp.gate_proj.lora_A.default.weight",
        "model.layers.0.mlp.up_proj.lora_A.default.weight",
        "model.layers.0.mlp.down_proj.lora_A.default.weight",
    }


# --- Cycle C: overlap (pure set logic, the permissive n_overlap > 0 rule) ---

def test_overlap_zero_intersection():
    base = {"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"}
    lora = {"layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"}  # prefix mismatch
    assert pf.overlap(base, lora) == 0


def test_overlap_partial_intersection():
    base = {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"}
    lora = {"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"}
    assert pf.overlap(base, lora) == 1


def test_overlap_full_intersection():
    base = {"a", "b", "c"}
    lora = {"a", "b"}
    assert pf.overlap(base, lora) == 2


# --- Cycle D: parse_preflight_output + fail-open semantics ---

import json


def test_parse_preflight_output_ok():
    stdout = "some torch warning\n" + json.dumps({
        "predicted_ok": True, "n_overlap": 7, "n_total": 7,
        "sample_adapter_keys": ["model.layers.0.self_attn.q_proj"],
        "sample_base_modules": ["model.layers.0.self_attn.q_proj"],
    })
    r = pf.parse_preflight_output("Qwen/Qwen2.5-0.5B", stdout)
    assert r.model_id == "Qwen/Qwen2.5-0.5B"
    assert r.predicted_ok is True
    assert r.n_overlap == 7 and r.n_total == 7
    assert r.error == ""
    assert r.predicted_noop is False


def test_parse_preflight_output_real_noop_is_skipped():
    stdout = json.dumps({"predicted_ok": False, "n_overlap": 0, "n_total": 7,
                         "sample_adapter_keys": ["layers.0.self_attn.q_proj"],
                         "sample_base_modules": ["model.layers.0.self_attn.q_proj"]})
    r = pf.parse_preflight_output("m", stdout)
    assert r.predicted_ok is False
    assert r.predicted_noop is True   # zero overlap, no error -> skip the model


def test_parse_preflight_output_error_fails_open():
    stdout = json.dumps({"error": "ImportError: no such model arch"})
    r = pf.parse_preflight_output("m", stdout)
    assert r.error
    assert r.predicted_noop is False  # build crash -> run the model, don't skip


def test_parse_preflight_output_garbage_fails_open():
    r = pf.parse_preflight_output("m", "Traceback (most recent call last): boom")
    assert r.error
    assert r.predicted_noop is False


# --- Cycle F: split_by_preflight partitions run-list vs PRECHECK_NO_OP rows ---

def test_split_by_preflight_skips_predicted_noop_with_hint():
    models = [{"id": "ok-model"}, {"id": "noop-model"}]
    pf_results = {
        "ok-model": pf.PreflightResult("ok-model", predicted_ok=True,
                                       n_overlap=7, n_total=7),
        "noop-model": pf.PreflightResult(
            "noop-model", predicted_ok=False, n_overlap=0, n_total=7,
            sample_adapter_keys=["language_model.model.layers.0.self_attn.q_proj"],
            sample_base_modules=["model.layers.0.self_attn.q_proj"]),
    }
    to_run, skipped = rfs.split_by_preflight(models, pf_results, "cuda-docker")
    assert [m["id"] for m in to_run] == ["ok-model"]
    assert len(skipped) == 1
    s = skipped[0]
    assert s.model == "noop-model"
    assert s.outcome == rfs.PRECHECK_NO_OP
    assert "language_model.model.layers.0.self_attn.q_proj" in s.hint
    assert "model.layers.0.self_attn.q_proj" in s.hint


def test_split_by_preflight_fail_open_runs_model_on_build_error():
    models = [{"id": "crashy"}]
    pf_results = {"crashy": pf.PreflightResult("crashy", error="ImportError: boom")}
    to_run, skipped = rfs.split_by_preflight(models, pf_results, "cuda-docker")
    assert [m["id"] for m in to_run] == ["crashy"]  # build crash -> run, don't skip
    assert skipped == []


def test_split_by_preflight_runs_model_absent_from_results():
    # A model with no preflight entry (e.g. preflight not run for it) runs.
    models = [{"id": "unchecked"}]
    to_run, skipped = rfs.split_by_preflight(models, {}, "cuda-docker")
    assert [m["id"] for m in to_run] == ["unchecked"]
    assert skipped == []


# --- Cycle E: Result.hint surfaced as a report column ---

def test_write_reports_has_hint_column(tmp_path):
    results = [
        rfs.Result(model="m1", target="cpu", outcome=rfs.PRECHECK_NO_OP,
                   hint="adapter keys ['layers.0...'] vs base ['model.layers.0...']"),
        rfs.Result(model="m2", target="cpu", outcome=rfs.PASS),
    ]
    _json_path, md_path = rfs.write_reports(results, "cpu", tmp_path)
    md = md_path.read_text()
    assert "Hint" in md.splitlines()[2]            # header row carries the column
    assert "adapter keys" in md                     # the hint text is rendered
    assert rfs.PRECHECK_NO_OP in md
    # hint also present in the JSON
    assert "adapter keys" in _json_path.read_text()
