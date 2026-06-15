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

def _pod(phase, containers):
    # containers: list of (ready: bool, waiting_reason: str | None)
    cs = []
    for ready, waiting in containers:
        state = {"waiting": {"reason": waiting}} if waiting else {"running": {}}
        cs.append({"ready": ready, "state": state})
    return {"status": {"phase": phase, "containerStatuses": cs}}

def test_classify_empty_is_pending():
    assert rfs.classify_pod_status([]) == "pending"

def test_classify_all_running_ready_is_ready():
    pods = [_pod("Running", [(True, None)]), _pod("Running", [(True, None)])]
    assert rfs.classify_pod_status(pods) == "ready"

def test_classify_unschedulable_pending_is_pending():
    # Pending pod with no container statuses yet (scheduler hasn't placed it).
    assert rfs.classify_pod_status([{"status": {"phase": "Pending"}}]) == "pending"

def test_classify_container_not_ready_is_pending():
    pods = [_pod("Running", [(False, None)])]
    assert rfs.classify_pod_status(pods) == "pending"

def test_classify_crashloop_is_failed():
    pods = [_pod("Running", [(False, "CrashLoopBackOff")])]
    assert rfs.classify_pod_status(pods) == "failed"

def test_classify_imagepull_is_failed():
    pods = [_pod("Pending", [(False, "ImagePullBackOff")])]
    assert rfs.classify_pod_status(pods) == "failed"

def test_classify_failed_phase_is_failed():
    assert rfs.classify_pod_status([{"status": {"phase": "Failed"}}]) == "failed"
