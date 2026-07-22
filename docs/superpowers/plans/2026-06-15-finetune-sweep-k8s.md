# Fine-tune sweep on Kubernetes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `test/finetune_sweep/run_finetune_sweep.py`'s `cuda` target launch the stack as a fresh Helm release per model in its own k8s namespace (mirroring the team's `helm upgrade --install … -n <ns>` workflow) instead of a Docker-Compose recreate, so all GPU work goes through the k8s scheduler.

**Architecture:** Add pure helpers (namespace sanitizer, pod-status classifier, `kubectl`/`helm` command builders, target dispatch) that are unit-tested, plus thin side-effecting wrappers that shell out. `run_model` branches on `is_k8s_target(target_cfg)`: k8s does pre-clean → `helm upgrade --install` → block-and-wait for pods → per-model `kubectl port-forward` → train/serve (unchanged HTTP, still `localhost:8000`) → `kubectl exec` checkpoint read → `kubectl delete namespace`. The `cpu` (Compose) path is untouched. No chart change is required.

**Tech Stack:** Python 3 stdlib (`subprocess`, `json`, `re`, `urllib`), PyYAML, pytest (run via `uv`), `helm`, `kubectl`. Spec: `docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md`; ADR 0003 amendment 2026-06-15.

---

## File Structure

- **Modify:** `test/finetune_sweep/run_finetune_sweep.py` — add pure helpers and k8s lifecycle wrappers; branch `run_model`; add `--gpu-wait-timeout`. This is the single runner; everything lives here to match the existing structure (one self-contained host-side script).
- **Modify:** `test/finetune_sweep/finetune-sweep.yaml` — replace the `cuda` target's Compose fields with k8s identifiers.
- **Create:** `test/unit/test_finetune_sweep_k8s.py` — unit tests for the new pure functions (sanitizer, classifier, command builders, dispatch) plus monkeypatched-subprocess tests for the wait loop. Loads the script module by path via `importlib`.

**Test invocation (this repo has no torch/pytest in `.venv`; use `uv`):**
```bash
PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v
```
The pure-function tests need only `pyyaml` (the module imports `yaml` at top); they do **not** need `torch`, so they stay fast.

---

### Task 1: Namespace sanitizer (pure) + test harness

**Files:**
- Create: `test/unit/test_finetune_sweep_k8s.py`
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (add `import re` near the top imports; add `k8s_namespace`)

- [ ] **Step 1: Write the failing test (creates the test file + import shim)**

```python
# test/unit/test_finetune_sweep_k8s.py
import importlib.util
from pathlib import Path

# Load the script module by path (it is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "run_finetune_sweep",
    Path(__file__).resolve().parents[1] / "finetune_sweep" / "run_finetune_sweep.py",
)
rfs = importlib.util.module_from_spec(_SPEC)
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: FAIL — `AttributeError: module 'run_finetune_sweep' has no attribute 'k8s_namespace'`

- [ ] **Step 3: Implement `k8s_namespace`**

Add `import re` to the import block (after `import os`). Add this function next to the other pure helpers (e.g. just after `build_dataset`, around line 87):

```python
def k8s_namespace(prefix: str, model_id: str) -> str:
    """Sanitize a model id into an RFC1123 namespace label: lowercase, every run
    of non-alphanumeric chars -> '-', trimmed, prefixed, truncated to 63 chars."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{prefix}-{slug}"[:63].rstrip("-")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add k8s namespace sanitizer for per-model namespaces"
```

---

### Task 2: Pod-status classifier (pure)

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (add `FATAL_WAITING_REASONS`, `classify_pod_status`)
- Modify: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k classify -v`
Expected: FAIL — `AttributeError: ... has no attribute 'classify_pod_status'`

- [ ] **Step 3: Implement the classifier**

Add near the other pure helpers (after `gate_model`, around line 104):

```python
# Container waiting reasons that mean the rollout is broken, not just slow.
FATAL_WAITING_REASONS = {
    "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "InvalidImageName",
    "CreateContainerConfigError", "CreateContainerError", "RunContainerError",
}


def classify_pod_status(pods: list[dict]) -> str:
    """Classify a namespace's pods for block-and-wait scheduling. Returns:
      "failed"  - a pod is in a fatal state (crash / bad image / Failed phase);
                  stop and fail fast (RESTART_FAILED).
      "ready"   - every pod is Running with all containers ready; proceed.
      "pending" - nothing fatal yet but not all ready (incl. Pending /
                  Unschedulable / ContainerCreating); keep waiting for the
                  scheduler to place the GPU pods.
    """
    if not pods:
        return "pending"
    all_ready = True
    for pod in pods:
        status = pod.get("status", {})
        if status.get("phase") == "Failed":
            return "failed"
        container_statuses = status.get("containerStatuses", [])
        if not container_statuses or status.get("phase") != "Running":
            all_ready = False
        for cs in container_statuses:
            waiting = cs.get("state", {}).get("waiting")
            if waiting and waiting.get("reason") in FATAL_WAITING_REASONS:
                return "failed"
            if not cs.get("ready", False):
                all_ready = False
    return "ready" if all_ready else "pending"
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k classify -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add pod-status classifier for GPU block-and-wait"
```

---

### Task 3: `kubectl`/`helm` command builders + target dispatch (pure)

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py`
- Modify: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "is_k8s or cmd" -v`
Expected: FAIL — missing attributes `is_k8s_target` / `k8s_helm_install_cmd` / …

- [ ] **Step 3: Implement the builders and dispatch**

Add a `# --- k8s command builders (pure) ---` block near `probe_gpu_free_gb` (around line 250):

```python
def is_k8s_target(target_cfg: dict) -> bool:
    """A target is k8s-driven when it declares a Helm chart path (vs the
    Compose target's compose_service/restart_cmd)."""
    return "chart_path" in target_cfg


def k8s_helm_install_cmd(target_cfg: dict, namespace: str, model_id: str) -> list[str]:
    return [
        "helm", "upgrade", "--install", target_cfg["release"], target_cfg["chart_path"],
        "-n", namespace, "--create-namespace",
        "--set", f"model={model_id}",
        "--set", "storage.cache.kind=hostPath",
        "--set", f"storage.cache.hostPath={target_cfg['cache_hostpath']}",
    ]


def k8s_delete_namespace_cmd(namespace: str) -> list[str]:
    return ["kubectl", "delete", "namespace", namespace, "--ignore-not-found", "--wait"]


def k8s_port_forward_cmd(target_cfg: dict, namespace: str, port: int = 8000) -> list[str]:
    return ["kubectl", "port-forward", "-n", namespace,
            f"svc/{target_cfg['api_service']}", f"{port}:{port}"]


def k8s_exec_checkpoint_cmd(target_cfg: dict, namespace: str, script: str) -> list[str]:
    return ["kubectl", "exec", "-n", namespace,
            f"statefulset/{target_cfg['megatron_sts']}", "--", "python3", "-c", script]


def k8s_get_pods_cmd(namespace: str) -> list[str]:
    return ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "is_k8s or cmd" -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add k8s command builders and target dispatch"
```

---

### Task 4: Side-effecting k8s wrappers + checkpoint-read refactor

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py`
- Modify: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests (monkeypatch subprocess)**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "wait_for_pods or get_pods" -v`
Expected: FAIL — missing `kubectl_get_pods` / `wait_for_pods_ready`

- [ ] **Step 3: Implement the wrappers and refactor the checkpoint script**

First, refactor `read_checkpoint_keys` (currently lines ~300-325) to extract the in-container script so both Compose and k8s readers share it. Replace the body of `read_checkpoint_keys` so the script-building lines become a helper:

```python
def _checkpoint_keys_script(job_hash: str) -> str:
    """The in-container Python that prints the latest checkpoint's
    model_state_dict keys as JSON. Identical for Compose and k8s."""
    return (
        "import glob, re, json, torch\n"
        f"paths = glob.glob('/app/cray/jobs/{job_hash}/checkpoint_*.pt')\n"
        "def step(p):\n"
        "    m = re.search(r'checkpoint_(\\d+)\\.pt', p)\n"
        "    return int(m.group(1)) if m else -1\n"
        "paths.sort(key=step)\n"
        "ckpt = torch.load(paths[-1], map_location='cpu', weights_only=False) if paths else None\n"
        "print(json.dumps(list(ckpt['model_state_dict'].keys()) if ckpt else None))\n"
    )


def _parse_checkpoint_keys_output(stdout: str) -> list[str] | None:
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
```

Update the existing `read_checkpoint_keys` to use them (Compose path):

```python
def read_checkpoint_keys(compose_service: str, job_hash: str, timeout: float = 60) -> list[str] | None:
    """Compose path: read the latest checkpoint's state-dict keys via
    `docker compose exec`. Returns None if unreachable / no checkpoint."""
    cmd = ["docker", "compose", "-f", "docker-compose.yaml", "exec", "-T", compose_service,
           "python3", "-c", _checkpoint_keys_script(job_hash)]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return _parse_checkpoint_keys_output(result.stdout)
```

Now add the k8s wrappers near the command builders:

```python
def kubectl_get_pods(namespace: str, timeout: float = 15) -> list[dict]:
    """Return the namespace's pod objects (`.items`), or [] if the call fails."""
    try:
        result = subprocess.run(k8s_get_pods_cmd(namespace), cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    try:
        return json.loads(result.stdout).get("items", [])
    except json.JSONDecodeError:
        return []


def helm_install(target_cfg: dict, namespace: str, model_id: str, log, timeout: float = 600) -> bool:
    """`helm upgrade --install` the per-model release. True iff helm exits 0."""
    try:
        subprocess.run(k8s_helm_install_cmd(target_cfg, namespace, model_id), cwd=REPO_ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def wait_for_pods_ready(namespace: str, gpu_wait_timeout: float, poll: float = 10.0) -> str:
    """Block until pods are "ready" or "failed", else "timeout" at gpu_wait_timeout.
    "pending"/Unschedulable keeps waiting (GPU may be busy); see classify_pod_status."""
    deadline = time.time() + gpu_wait_timeout
    while time.time() < deadline:
        status = classify_pod_status(kubectl_get_pods(namespace))
        if status in ("ready", "failed"):
            return status
        time.sleep(poll)
    return "timeout"


def start_port_forward(target_cfg: dict, namespace: str, log) -> subprocess.Popen:
    """Spawn `kubectl port-forward svc/<api> 8000:8000` in its own process group."""
    return subprocess.Popen(k8s_port_forward_cmd(target_cfg, namespace), cwd=REPO_ROOT,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def delete_namespace(namespace: str, log, timeout: float = 300) -> None:
    """`kubectl delete namespace --ignore-not-found --wait` (also GCs the
    resource-policy:keep PVCs). Used for pre-clean and teardown; never raises."""
    try:
        subprocess.run(k8s_delete_namespace_cmd(namespace), cwd=REPO_ROOT,
                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def read_checkpoint_keys_k8s(target_cfg: dict, namespace: str, job_hash: str,
                             timeout: float = 60) -> list[str] | None:
    """k8s path: read checkpoint keys via `kubectl exec statefulset/<megatron>`."""
    cmd = k8s_exec_checkpoint_cmd(target_cfg, namespace, _checkpoint_keys_script(job_hash))
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return _parse_checkpoint_keys_output(result.stdout)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (all tests so far, ~24 passed)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add k8s lifecycle wrappers and share checkpoint-read script"
```

---

### Task 5: Make `wait_for_all_up` proc-optional + add `--gpu-wait-timeout`

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (`wait_for_all_up` lines ~138-150; `main` argparse ~455-465)
- Modify: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing test**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
def test_wait_for_all_up_accepts_none_proc(monkeypatch):
    monkeypatch.setattr(rfs, "get_health", lambda url, timeout=5: {"all": "up"})
    # proc=None must not raise and must return True when health is up.
    assert rfs.wait_for_all_up("http://localhost:8000", None, timeout=1) is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k wait_for_all_up -v`
Expected: FAIL — `AttributeError: 'NoneType' object has no attribute 'poll'`

- [ ] **Step 3: Make `proc` optional and add the CLI flag**

Replace `wait_for_all_up` (lines ~138-150) so the proc check is guarded:

```python
def wait_for_all_up(api_url: str, proc, timeout: float) -> bool:
    """Poll /v1/health until health["all"] == "up", the restart process dies
    (Compose only; pass proc=None for k8s), or timeout. True iff ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        health = get_health(api_url)
        if health and health.get("all") == "up":
            return True
        time.sleep(2)
    return False
```

In `main()`, add after the `--serve-timeout` line (~464):

```python
    ap.add_argument("--gpu-wait-timeout", type=int, default=7200,
                    help="k8s only: seconds to block while pods are Unschedulable "
                         "(GPUs busy) before giving up with RESTART_FAILED")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k wait_for_all_up -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): make wait_for_all_up proc-optional; add --gpu-wait-timeout"
```

---

### Task 6: Branch `run_model` between Compose and k8s lifecycles

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (`run_model`, lines ~327-434)

This task has no new unit test (it orchestrates side-effecting calls validated by the integration run in Task 8). Keep the change surgical: only the launch block, the checkpoint-read call, and the teardown `finally` change; the baseline/train/serve body is untouched.

- [ ] **Step 1: Initialize handles and branch the launch block**

In `run_model`, the current launch block is:

```python
    log_path = results_dir / f"{model_id.replace('/', '_')}.{target}.restart.log"
    restart_start = time.time()
    with open(log_path, "w") as log:
        proc = start_restart(target_cfg, model_id, log)
        try:
            ready = wait_for_all_up(args.api_url, proc, args.restart_timeout)
            res.restart_seconds = round(time.time() - restart_start, 1)
            if not ready:
                res.outcome = RESTART_FAILED
                res.detail = "stack did not report health.all == 'up' in time"
                return res
```

Replace it with (note `try:` now opens immediately so the `finally` always runs):

```python
    log_path = results_dir / f"{model_id.replace('/', '_')}.{target}.restart.log"
    is_k8s = is_k8s_target(target_cfg)
    namespace = pf_proc = proc = None
    restart_start = time.time()
    with open(log_path, "w") as log:
        try:
            if is_k8s:
                namespace = k8s_namespace(target_cfg["namespace_prefix"], model_id)
                delete_namespace(namespace, log)  # idempotent pre-clean
                if not helm_install(target_cfg, namespace, model_id, log, args.restart_timeout):
                    res.outcome, res.detail = RESTART_FAILED, "helm upgrade --install failed"
                    return res
                pod_status = wait_for_pods_ready(namespace, args.gpu_wait_timeout)
                res.restart_seconds = round(time.time() - restart_start, 1)
                if pod_status != "ready":
                    res.outcome = RESTART_FAILED
                    res.detail = ("pods crashed / bad image" if pod_status == "failed"
                                  else f"pods not schedulable within {args.gpu_wait_timeout}s")
                    return res
                pf_proc = start_port_forward(target_cfg, namespace, log)
                if not wait_for_all_up(args.api_url, None, args.restart_timeout):
                    res.outcome, res.detail = RESTART_FAILED, "api /v1/health not up after port-forward"
                    return res
            else:
                proc = start_restart(target_cfg, model_id, log)
                ready = wait_for_all_up(args.api_url, proc, args.restart_timeout)
                res.restart_seconds = round(time.time() - restart_start, 1)
                if not ready:
                    res.outcome = RESTART_FAILED
                    res.detail = "stack did not report health.all == 'up' in time"
                    return res
```

The existing baseline / `submit_train` / training-poll / serve body that follows stays exactly as-is (it already uses `args.api_url` = `localhost:8000`). **Remove the now-duplicated inner `try:`** that previously wrapped that body (the one right after `proc = start_restart(...)`), since the `try:` now opens earlier — the body's indentation is unchanged, only the wrapping `try:` line is deleted.

- [ ] **Step 2: Branch the checkpoint-read call**

Find (line ~401):

```python
                checkpoint_keys = read_checkpoint_keys(target_cfg["compose_service"], job_hash)
```

Replace with:

```python
                if is_k8s:
                    checkpoint_keys = read_checkpoint_keys_k8s(target_cfg, namespace, job_hash)
                else:
                    checkpoint_keys = read_checkpoint_keys(target_cfg["compose_service"], job_hash)
```

- [ ] **Step 3: Branch the teardown `finally`**

Replace the existing teardown (lines ~433-434):

```python
        finally:
            teardown_stack(proc)
```

with:

```python
        finally:
            if is_k8s:
                if pf_proc is not None:
                    try:
                        os.killpg(os.getpgid(pf_proc.pid), signal.SIGKILL)
                    except OSError:
                        pass
                    pf_proc.wait()
                if namespace is not None:
                    delete_namespace(namespace, log)
            elif proc is not None:
                teardown_stack(proc)
```

- [ ] **Step 4: Verify the module still imports and unit tests pass**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (module imports cleanly with the new `run_model`; all prior tests still green)

Also smoke-check the CLI parses: `PYTHONPATH="$PWD/infra" uv run --with pyyaml python test/finetune_sweep/run_finetune_sweep.py --help`
Expected: usage text including `--gpu-wait-timeout`.

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py
git commit -m "feat(sweep): branch run_model between Compose and k8s lifecycles"
```

---

### Task 7: Manifest — k8s `cuda` target

**Files:**
- Modify: `test/finetune_sweep/finetune-sweep.yaml`

- [ ] **Step 1: Replace the `cuda` target block**

Current:

```yaml
  cuda:
    compose_service: cray-nvidia
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cuda"
    train_args_overrides:
      gpus: 1
```

Replace with (leave `cpu` untouched — it stays on Compose):

```yaml
  cuda:
    # k8s target: one Helm release per model in a fresh `sweep-<model>` namespace.
    # See docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md and the
    # 2026-06-15 amendment in docs/adr/0003-finetune-sweep-restart-per-model.md.
    chart_path: deployment/helm/scalarlm
    release: scalarlm              # namespace isolates, so a fixed name is fine
    namespace_prefix: sweep        # -> sweep-<sanitized-model-id>
    megatron_sts: scalarlm-megatron  # kubectl exec target (StatefulSet)
    api_service: scalarlm          # svc for port-forward (== release fullname)
    # NOTE: confirm the node's populated HF cache path on the cuda box before a
    # real run — the team's serving values use a hostPath cache (e.g. /root/.cache
    # in values/gemma4-31b.yaml). This may differ on blackwell-maxq-0.
    cache_hostpath: /root/.cache
    train_args_overrides:
      gpus: 1
```

- [ ] **Step 2: Verify the manifest loads and the cuda target is recognized as k8s**

Run:
```bash
PYTHONPATH="$PWD/infra" uv run --with pyyaml python -c "
import importlib.util, pathlib
s = importlib.util.spec_from_file_location('rfs', 'test/finetune_sweep/run_finetune_sweep.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
mani = m.load_manifest(pathlib.Path('test/finetune_sweep/finetune-sweep.yaml'))
cuda = mani['targets']['cuda']
assert m.is_k8s_target(cuda), 'cuda should be a k8s target'
assert not m.is_k8s_target(mani['targets']['cpu']), 'cpu should stay Compose'
print('OK: cuda is k8s,', m.k8s_namespace(cuda['namespace_prefix'], mani['models'][0]['id']))
"
```
Expected: `OK: cuda is k8s, sweep-qwen-qwen2-5-0-5b`

- [ ] **Step 3: Commit**

```bash
git add test/finetune_sweep/finetune-sweep.yaml
git commit -m "feat(sweep): point cuda target at the Helm namespace-per-model workflow"
```

---

### Task 8: Integration validation on `blackwell-maxq-0` (manual)

This cannot be unit-tested here — it needs the live cluster. The runner is GPU-less and run from the host (Approach A), exactly as the team runs `helm`. Hand the operator these steps.

- [ ] **Step 1: Confirm prerequisites on the box**

```bash
helm version && kubectl version --client && kubectl get nodes
# Confirm GPUs are free enough (2 per model): vLLM + megatron each request 1.
kubectl get pods -A | grep -E 'scalarlm|vllm|megatron'
```
Confirm the `cache_hostpath` in the manifest matches the node's populated HF cache (adjust `--set storage.cache.hostPath` / the manifest if not `/root/.cache`).

- [ ] **Step 2: Ensure no Compose GPU stack is running (the contention this prevents)**

```bash
docker compose ps 2>/dev/null | grep -i cray-nvidia && echo "STOP: decommission the Compose GPU stack first" || echo "clear"
```

- [ ] **Step 3: Run the sweep against cuda**

```bash
cd <repo-on-box>
PYTHONPATH="$PWD/infra" uv run --with pyyaml python test/finetune_sweep/run_finetune_sweep.py \
  --target cuda --gpu-wait-timeout 7200
```

- [ ] **Step 4: Observe in a second shell**

```bash
watch kubectl -n sweep-qwen-qwen2-5-0-5b get pods -o wide
```
Expected progression: pods `Pending` (scheduling/GPU) → `Running`/Ready → training job runs → adapter served. The runner prints a per-model outcome line and writes `test/finetune_sweep/results/*.cuda.{json,md}`.

- [ ] **Step 5: Confirm clean teardown (no leaked longhorn volumes)**

```bash
kubectl get ns | grep sweep- || echo "namespaces cleaned"
kubectl get pvc -A | grep sweep- || echo "no leaked PVCs"
```
Expected: the `sweep-*` namespace and its PVCs are gone (deleted by `kubectl delete namespace`).

- [ ] **Step 6: Update docs with any reality gaps**

If `cache_hostpath`, the release name, or the StatefulSet name differ on the box, update `finetune-sweep.yaml` and the "Open questions / caveats" in the spec, then commit:

```bash
git add test/finetune_sweep/finetune-sweep.yaml docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md
git commit -m "fix(sweep): reconcile k8s target with blackwell-maxq-0 layout"
```

---

## Self-Review

**Spec coverage:**
- Namespace-per-model launch (`helm upgrade --install … --create-namespace`) → Task 3 (builder) + Task 6 (wiring) + Task 7 (manifest). ✓
- Inline `--set model` → Task 3 `k8s_helm_install_cmd`. ✓
- Dedicated `sweep-<model>` namespace + RFC1123 sanitization → Task 1. ✓
- Block-and-wait (Pending→wait, crash→fail-fast, outer cap) → Task 2 (classifier) + Task 4 (`wait_for_pods_ready`) + Task 5 (`--gpu-wait-timeout`) + Task 6 (RESTART_FAILED mapping). ✓
- Per-model `kubectl port-forward`, HTTP helpers unchanged → Task 4 (`start_port_forward`) + Task 6. ✓
- Checkpoint read via `kubectl exec statefulset/...` → Task 3 + Task 4 + Task 6. ✓
- Teardown via `kubectl delete namespace` (incl. idempotent pre-clean) → Task 4 + Task 6. ✓
- Shared HF hostPath cache → Task 3 (`--set storage.cache.*`) + Task 7. ✓
- VRAM probe dropped on k8s path → Task 6 (k8s branch never calls `probe_gpu_free_gb`; the Compose `cpu` path keeps it). ✓
- No chart change required → confirmed; no chart task exists. ✓
- `cpu`/Compose path unchanged → Task 6 keeps the `else` branch; Task 7 leaves `cpu` as-is. ✓

**Note on the cuda VRAM gate:** `run_model` currently calls `gate_model(model, target, free_gb)` with `free_gb = probe_gpu_free_gb() if target == "cuda" else []`. On the k8s cuda path there is no host `nvidia-smi`. The minimal, spec-aligned change (per "VRAM gate: dropped") is to compute `free_gb = []` when `is_k8s_target(target_cfg)` so the gate is skipped (the scheduler arbitrates). If you want a `SKIPPED` reason retained, gate on the static `gate_gb` only. This adjustment lives in Task 6 Step 1 alongside the branch — set `free_gb = [] if is_k8s else (probe_gpu_free_gb() if target == "cuda" else [])`.

**Placeholder scan:** no TBD/TODO; every code step contains full code; every command has expected output. ✓

**Type/name consistency:** `target_cfg` keys (`chart_path`, `release`, `namespace_prefix`, `megatron_sts`, `api_service`, `cache_hostpath`) are identical across the manifest (Task 7), the builders (Task 3), and the wiring (Task 6). `wait_for_pods_ready` returns `"ready"|"failed"|"timeout"`; `classify_pod_status` returns `"ready"|"failed"|"pending"` — distinct on purpose (the wait loop maps `timeout`). Function names match across tasks. ✓
