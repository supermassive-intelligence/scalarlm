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

CUDA_CFG = {
    "chart_path": "deployment/helm/scalarlm",
    "release": "scalarlm",
    "namespace_prefix": "sweep",
    "megatron_sts": "scalarlm-megatron",
    "api_service": "scalarlm",
    "cache_hostpath": "/root/.cache",
    "gpu_wait_timeout": 7200,
}

def test_is_k8s_target_true_for_chart_path():
    assert rfs.is_k8s_target(CUDA_CFG) is True

def test_is_k8s_target_false_for_compose():
    assert rfs.is_k8s_target({"compose_service": "cray", "restart_cmd": "x"}) is False

def test_helm_install_cmd():
    cmd = rfs.k8s_helm_install_cmd(CUDA_CFG, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert cmd[:5] == ["helm", "upgrade", "--install", "scalarlm", "deployment/helm/scalarlm"]
    assert "-n" in cmd and "sweep-qwen" in cmd and "--create-namespace" in cmd
    assert "model=Qwen/Qwen2.5-0.5B" in cmd
    assert "storage.cache.kind=hostPath" in cmd
    assert "storage.cache.hostPath=/root/.cache" in cmd

def test_helm_install_cmd_omits_jobs_size_by_default():
    # No jobs_size in cfg -> no override, chart default (100Gi) applies.
    cmd = rfs.k8s_helm_install_cmd(CUDA_CFG, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert not any(c.startswith("storage.jobs.size=") for c in cmd)

def test_helm_install_cmd_sets_jobs_size_when_configured():
    cfg = {**CUDA_CFG, "jobs_size": "20Gi"}
    cmd = rfs.k8s_helm_install_cmd(cfg, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert "storage.jobs.size=20Gi" in cmd

def test_delete_namespace_cmd():
    assert rfs.k8s_delete_namespace_cmd("sweep-qwen") == [
        "kubectl", "delete", "namespace", "sweep-qwen", "--ignore-not-found", "--wait"]

def test_port_forward_cmd():
    assert rfs.k8s_port_forward_cmd(CUDA_CFG, "sweep-qwen") == [
        "kubectl", "port-forward", "-n", "sweep-qwen", "svc/scalarlm", "8000:8000"]

def test_exec_checkpoint_cmd_targets_statefulset():
    cmd = rfs.k8s_exec_checkpoint_cmd(CUDA_CFG, "sweep-qwen", "print(1)")
    assert cmd == ["kubectl", "exec", "-n", "sweep-qwen",
                   "statefulset/scalarlm-megatron", "--", "python3", "-c", "print(1)"]

def test_get_pods_cmd():
    assert rfs.k8s_get_pods_cmd("sweep-qwen") == [
        "kubectl", "get", "pods", "-n", "sweep-qwen", "-o", "json"]

def test_wait_for_pods_ready_returns_ready(monkeypatch):
    monkeypatch.setattr(rfs, "kubectl_get_pods",
                        lambda ns, timeout=15: [_pod("Running", [(True, None)])])
    assert rfs.wait_for_pods_ready("sweep-qwen", gpu_wait_timeout=5, poll=0.01) == "ready"

def test_wait_for_pods_ready_fails_fast_on_crash(monkeypatch):
    monkeypatch.setattr(rfs, "kubectl_get_pods",
                        lambda ns, timeout=15: [_pod("Running", [(False, "CrashLoopBackOff")])])
    assert rfs.wait_for_pods_ready("sweep-qwen", gpu_wait_timeout=5, poll=0.01) == "failed"

def test_wait_for_pods_ready_times_out_while_pending(monkeypatch):
    monkeypatch.setattr(rfs, "kubectl_get_pods", lambda ns, timeout=15: [])  # always pending
    assert rfs.wait_for_pods_ready("sweep-qwen", gpu_wait_timeout=0.05, poll=0.01) == "timeout"

def test_kubectl_get_pods_parses_items(monkeypatch):
    class _R:  # fake CompletedProcess
        stdout = '{"items": [{"status": {"phase": "Running"}}]}'
    monkeypatch.setattr(rfs.subprocess, "run", lambda *a, **k: _R())
    assert rfs.kubectl_get_pods("sweep-qwen") == [{"status": {"phase": "Running"}}]

def test_wait_for_all_up_accepts_none_proc(monkeypatch):
    monkeypatch.setattr(rfs, "get_health", lambda url, timeout=5: {"all": "up"})
    # proc=None must not raise and must return True when health is up.
    assert rfs.wait_for_all_up("http://localhost:8000", None, timeout=1) is True

def test_gate_model_k8s_skips_vram_check_when_free_gb_none():
    model = {"id": "m", "adapters": {"lora": {"gate_gb": 8}}}
    ok, reason = rfs.gate_model(model, "cuda", None)
    assert ok is True and reason == ""

def test_gate_model_k8s_still_requires_gate_gb_declared():
    model = {"id": "m"}  # no adapters.lora.gate_gb
    ok, reason = rfs.gate_model(model, "cuda", None)
    assert ok is False and "gate_gb" in reason

def test_gate_model_cuda_still_gates_on_free_gb_list():
    model = {"id": "m", "adapters": {"lora": {"gate_gb": 8}}}
    assert rfs.gate_model(model, "cuda", [4.0])[0] is False   # not enough free
    assert rfs.gate_model(model, "cuda", [16.0])[0] is True    # enough free

def test_kubectl_get_pods_returns_empty_on_called_process_error(monkeypatch):
    def _boom(*a, **k):
        raise rfs.subprocess.CalledProcessError(1, "kubectl")
    monkeypatch.setattr(rfs.subprocess, "run", _boom)
    assert rfs.kubectl_get_pods("sweep-qwen") == []

def test_kubectl_get_pods_returns_empty_on_bad_json(monkeypatch):
    class _R:
        stdout = "not json"
    monkeypatch.setattr(rfs.subprocess, "run", lambda *a, **k: _R())
    assert rfs.kubectl_get_pods("sweep-qwen") == []

def test_wait_for_all_up_gates_on_health_key(monkeypatch):
    # `all` is down, but the vllm component is up -> gating on "vllm" succeeds.
    monkeypatch.setattr(rfs, "get_health", lambda url, timeout=5: {"vllm": "up", "all": "down"})
    assert rfs.wait_for_all_up("http://x", None, timeout=1, health_key="vllm") is True

def test_wait_for_all_up_default_key_is_all(monkeypatch):
    monkeypatch.setattr(rfs, "get_health", lambda url, timeout=5: {"vllm": "up", "all": "down"})
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)  # don't actually sleep 2s
    assert rfs.wait_for_all_up("http://x", None, timeout=0.01) is False  # no health_key -> default "all" -> False

def test_k8s_scale_cmd():
    assert rfs.k8s_scale_cmd("statefulset/scalarlm-megatron", 0, "sweep-qwen") == [
        "kubectl", "scale", "statefulset/scalarlm-megatron", "--replicas=0", "-n", "sweep-qwen"]

def test_kubectl_scale_true_on_success(monkeypatch):
    monkeypatch.setattr(rfs.subprocess, "run", lambda *a, **k: None)
    assert rfs.kubectl_scale("deployment/scalarlm-vllm", 1, "sweep-qwen", log=None) is True

def test_kubectl_scale_false_on_error(monkeypatch):
    def boom(*a, **k):
        raise rfs.subprocess.CalledProcessError(1, a[0])
    monkeypatch.setattr(rfs.subprocess, "run", boom)
    assert rfs.kubectl_scale("deployment/scalarlm-vllm", 1, "sweep-qwen", log=None) is False

def test_get_pods_cmd_with_selector():
    assert rfs.k8s_get_pods_cmd("sweep-qwen", "app.kubernetes.io/component=megatron") == [
        "kubectl", "get", "pods", "-n", "sweep-qwen",
        "-l", "app.kubernetes.io/component=megatron", "-o", "json"]

def test_get_pods_cmd_without_selector_unchanged():
    assert rfs.k8s_get_pods_cmd("sweep-qwen") == [
        "kubectl", "get", "pods", "-n", "sweep-qwen", "-o", "json"]

def test_wait_for_pods_gone_returns_gone_when_empty(monkeypatch):
    seen = []
    def fake_get(ns, selector=None, timeout=15):
        seen.append(selector)
        return []
    monkeypatch.setattr(rfs, "kubectl_get_pods", fake_get)
    assert rfs.wait_for_pods_gone("sweep-qwen", "sel", gpu_wait_timeout=5, poll=0.01) == "gone"
    assert seen[0] == "sel"  # selector must flow through to kubectl_get_pods

def test_wait_for_pods_gone_times_out_while_present(monkeypatch):
    monkeypatch.setattr(rfs, "kubectl_get_pods",
                        lambda ns, selector=None, timeout=15: [{"metadata": {"name": "megatron-0"}}])
    assert rfs.wait_for_pods_gone("sweep-qwen", "sel", gpu_wait_timeout=0.05, poll=0.01) == "timeout"

def test_megatron_pod_selector_is_component_label():
    assert rfs.MEGATRON_POD_SELECTOR == "app.kubernetes.io/component=megatron"

def test_helm_install_cmd_no_phase_scaling_by_default():
    cmd = rfs.k8s_helm_install_cmd(CUDA_CFG, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert "replicaCounts.inference=0" not in cmd
    assert "replicaCounts.training=1" not in cmd

def test_helm_install_cmd_phase_scaled_sets_replica_counts():
    cfg = {**CUDA_CFG, "phase_scaled": True}
    cmd = rfs.k8s_helm_install_cmd(cfg, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert "replicaCounts.inference=0" in cmd  # vLLM off in phase 0/1
    assert "replicaCounts.training=1" in cmd   # megatron holds the single GPU

def test_poll_training_returns_terminal_status(monkeypatch):
    monkeypatch.setattr(rfs, "get_training_job", lambda url, jh, timeout=10: {"status": "COMPLETED"})
    assert rfs.poll_training("http://x", "abc", train_timeout=1) == "COMPLETED"

def test_poll_training_times_out(monkeypatch):
    monkeypatch.setattr(rfs, "get_training_job", lambda url, jh, timeout=10: {"status": "TRAINING"})
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)
    assert rfs.poll_training("http://x", "abc", train_timeout=0.01) == "TIMEOUT"


class _ServeArgs:  # minimal stand-in for the argparse Namespace fields used here
    serve_timeout = 1
    train_timeout = 60
    max_tokens = 32

def test_serve_check_sets_pass_on_memorization(monkeypatch):
    monkeypatch.setattr(rfs, "generate", lambda api, prompts, jh, mt: ["... SECRET123 ..."])
    res = rfs.Result(model="m", target="cuda")
    rfs.serve_check_and_classify("http://x", "p", "SECRET123", "abc", "COMPLETED",
                                 ["base.lora_A.weight", "base.lora_B.weight"], _ServeArgs(), res)
    assert res.outcome == rfs.PASS
    assert "SECRET123" in res.adapter_sample

def test_serve_check_adapter_not_loaded(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("404 model not found")
    monkeypatch.setattr(rfs, "generate", boom)
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)
    res = rfs.Result(model="m", target="cuda")
    rfs.serve_check_and_classify("http://x", "p", "SECRET123", "abc", "COMPLETED",
                                 ["base.lora_A.weight", "base.lora_B.weight"], _ServeArgs(), res)
    assert res.outcome == rfs.ADAPTER_NOT_LOADED

def test_serve_check_bad_checkpoint(monkeypatch):
    # No lora_B key -> checkpoint_lora_keys_ok is False -> BAD_CHECKPOINT, no generate call.
    monkeypatch.setattr(rfs, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    res = rfs.Result(model="m", target="cuda")
    rfs.serve_check_and_classify("http://x", "p", "SECRET123", "abc", "COMPLETED",
                                 ["base.lora_A.weight"], _ServeArgs(), res)
    assert res.outcome == rfs.BAD_CHECKPOINT


def test_run_model_k8s_phased_orders_gpu_handoff(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rfs, "delete_namespace", lambda ns, log, **k: None)
    monkeypatch.setattr(rfs, "helm_install", lambda *a, **k: True)
    monkeypatch.setattr(rfs, "wait_for_pods_ready", lambda ns, t, **k: "ready")
    monkeypatch.setattr(rfs, "start_port_forward", lambda *a, **k: None)
    monkeypatch.setattr(rfs, "wait_for_all_up", lambda *a, **k: True)
    monkeypatch.setattr(rfs, "submit_train", lambda *a, **k: {"job_directory": "/app/cray/jobs/abc"})
    monkeypatch.setattr(rfs, "poll_training", lambda *a, **k: "COMPLETED")
    monkeypatch.setattr(rfs, "read_checkpoint_keys_k8s",
                        lambda *a, **k: ["base.lora_A.weight", "base.lora_B.weight"])
    monkeypatch.setattr(rfs, "kubectl_scale",
                        lambda kn, r, ns, log, **k: (calls.append((kn, r)), True)[1])
    monkeypatch.setattr(rfs, "wait_for_pods_gone", lambda *a, **k: "gone")
    # baseline (model="Qwen/...") has no secret; adapter (model="abc") memorizes it.
    monkeypatch.setattr(rfs, "generate",
                        lambda api, prompts, model, mt: ["x SECRET123 y"] if model == "abc"
                        else ["plain baseline"])
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)

    cfg = {**CUDA_CFG, "vllm_deploy": "scalarlm-vllm", "phase_scaled": True}
    args = type("A", (), {"api_url": "http://x", "restart_timeout": 1, "serve_timeout": 1,
                          "train_timeout": 1, "max_tokens": 32, "gpu_wait_timeout": 1})()
    res = rfs.Result(model="Qwen/Qwen2.5-0.5B", target="cuda")
    with open(tmp_path / "log.txt", "w") as log:
        out = rfs.run_model_k8s_phased(cfg, "Qwen/Qwen2.5-0.5B", {}, [], "p", "SECRET123",
                                       args, res, log)

    assert out.outcome == rfs.PASS
    # megatron must be freed (->0) before vLLM claims the GPU (->1).
    assert calls == [("statefulset/scalarlm-megatron", 0), ("deployment/scalarlm-vllm", 1)]


def test_run_model_k8s_phased_fails_fast_on_scale_down(monkeypatch, tmp_path):
    # Phase 0/1 succeed, then the megatron scale-down fails -> RESTART_FAILED.
    monkeypatch.setattr(rfs, "delete_namespace", lambda ns, log, **k: None)
    monkeypatch.setattr(rfs, "helm_install", lambda *a, **k: True)
    monkeypatch.setattr(rfs, "wait_for_pods_ready", lambda ns, t, **k: "ready")
    monkeypatch.setattr(rfs, "start_port_forward", lambda *a, **k: None)
    monkeypatch.setattr(rfs, "wait_for_all_up", lambda *a, **k: True)
    monkeypatch.setattr(rfs, "submit_train", lambda *a, **k: {"job_directory": "/app/cray/jobs/abc"})
    monkeypatch.setattr(rfs, "poll_training", lambda *a, **k: "COMPLETED")
    monkeypatch.setattr(rfs, "read_checkpoint_keys_k8s",
                        lambda *a, **k: ["base.lora_A.weight", "base.lora_B.weight"])
    monkeypatch.setattr(rfs, "kubectl_scale", lambda kn, r, ns, log, **k: False)  # scale fails
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)

    cfg = {**CUDA_CFG, "vllm_deploy": "scalarlm-vllm", "phase_scaled": True}
    args = type("A", (), {"api_url": "http://x", "restart_timeout": 1, "serve_timeout": 1,
                          "train_timeout": 1, "max_tokens": 32, "gpu_wait_timeout": 1})()
    res = rfs.Result(model="Qwen/Qwen2.5-0.5B", target="cuda")
    with open(tmp_path / "log.txt", "w") as log:
        out = rfs.run_model_k8s_phased(cfg, "Qwen/Qwen2.5-0.5B", {}, [], "p", "SECRET123",
                                       args, res, log)

    assert out.outcome == rfs.RESTART_FAILED
    assert "megatron->0" in out.detail


def test_target_requests_gpu_true_when_gpus_one():
    assert rfs.target_requests_gpu({"train_args_overrides": {"gpus": 1}}) is True

def test_target_requests_gpu_false_when_gpus_zero():
    assert rfs.target_requests_gpu({"train_args_overrides": {"gpus": 0}}) is False

def test_target_requests_gpu_defaults_true_when_unset():
    # JobConfig defaults gpus=1, so a target with no override requests a GPU.
    assert rfs.target_requests_gpu({}) is True
    assert rfs.target_requests_gpu({"train_args_overrides": {}}) is True
