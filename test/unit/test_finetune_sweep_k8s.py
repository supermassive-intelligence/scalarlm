import importlib.util
import sys
from pathlib import Path

# Load the script module by path (it is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "run_finetune_sweep",
    Path(__file__).resolve().parents[1] / "finetune_sweep" / "run_finetune_sweep.py",
)
rfs = importlib.util.module_from_spec(_SPEC)
# Register in sys.modules before exec so dataclasses can resolve cls.__module__.
sys.modules[_SPEC.name] = rfs
_SPEC.loader.exec_module(rfs)


def test_k8s_namespace_sanitizes_model_id():
    assert rfs.k8s_namespace("sweep", "Qwen/Qwen2.5-0.5B") == "sweep-qwen-qwen2-5-0-5b"

def test_k8s_namespace_is_rfc1123_label():
    ns = rfs.k8s_namespace("sweep", "masint/tiny-random-llama")
    assert ns == "sweep-masint-tiny-random-llama"
    assert ns == ns.lower() and "/" not in ns and "_" not in ns and "." not in ns
    assert not ns.startswith("-") and not ns.endswith("-")

def test_k8s_namespace_truncates_to_63_chars():
    ns = rfs.k8s_namespace("sweep", "org/" + "a" * 200)
    assert len(ns) <= 63
    assert not ns.endswith("-")
