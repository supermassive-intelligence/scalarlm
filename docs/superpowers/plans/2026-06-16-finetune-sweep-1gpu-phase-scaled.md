# Fine-tune sweep — 1-GPU phase-scaled mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `phase_scaled: true` mode to the k8s `cuda` target so a model's train→serve closed loop runs in two **sequential single-GPU phases** (megatron trains, then the GPU is handed to vLLM), letting the sweep run on a node with only **one** schedulable GPU.

**Architecture:** Reuse the existing k8s helper layer. Add four small, unit-tested pieces (a `kubectl scale` builder+wrapper, a `wait_for_pods_gone` GPU-release barrier, a `health_key` parameter on `wait_for_all_up`, and `replicaCounts` overrides in `k8s_helm_install_cmd`), extract two shared orchestration helpers (`poll_training`, `serve_check_and_classify`) so the new path doesn't duplicate the 2-GPU body, then add `run_model_k8s_phased` selected by `target_cfg.get("phase_scaled")`. The 2-GPU k8s path and the `cpu`/Compose path are untouched.

**Tech Stack:** Python 3 stdlib (`subprocess`, `time`, `json`, `urllib`), PyYAML, pytest (run via `uv`), `helm`, `kubectl`. Spec: `docs/superpowers/specs/2026-06-15-finetune-sweep-1gpu-phase-scaled-design.md`; ADR 0003 amendment 2026-06-16.

---

## File Structure

- **Modify:** `test/finetune_sweep/run_finetune_sweep.py` — all runner logic lives in this one self-contained host-side script (matches the existing structure). Adds the scale builder/wrapper, the GPU-release barrier, the `health_key` parameter, the helm replica overrides, the two extracted helpers, and `run_model_k8s_phased` + its dispatch.
- **Modify:** `test/finetune_sweep/finetune-sweep.yaml` — add `vllm_deploy` and `phase_scaled: true` to the `cuda` target.
- **Modify:** `test/unit/test_finetune_sweep_k8s.py` — append unit tests for every new pure function and wrapper (monkeypatched subprocess/HTTP), plus a structural test of the phase ordering. Already loads the script module by path via `importlib`.

**Test invocation (this repo has no torch/pytest in `.venv`; use `uv`):**
```bash
PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v
```
The tests need only `pyyaml` (the module imports `yaml` at top); they do **not** need `torch`, so they stay fast.

---

### Task 1: Generalize `wait_for_all_up` with a per-phase `health_key`

The phase-scaled path can never gate on `health.all == "up"`: exactly one GPU service is up at a time, so `all` is structurally `down` in both phases (vLLM down in phase 1, megatron down in phase 2). Add a `health_key` parameter so phase 1 gates on `"megatron"` and phase 2 on `"vllm"`; the default `"all"` keeps both existing call sites unchanged.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py:185-196`
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
def test_wait_for_all_up_gates_on_health_key(monkeypatch):
    # `all` is down, but the vllm component is up -> gating on "vllm" succeeds.
    monkeypatch.setattr(rfs, "get_health", lambda url, timeout=5: {"vllm": "up", "all": "down"})
    assert rfs.wait_for_all_up("http://x", None, timeout=1, health_key="vllm") is True

def test_wait_for_all_up_default_key_is_all(monkeypatch):
    monkeypatch.setattr(rfs, "get_health", lambda url, timeout=5: {"vllm": "up", "all": "down"})
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)  # don't actually sleep 2s
    assert rfs.wait_for_all_up("http://x", None, timeout=0.01, health_key="all") is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k wait_for_all_up -v`
Expected: FAIL — `TypeError: wait_for_all_up() got an unexpected keyword argument 'health_key'`

- [ ] **Step 3: Add the `health_key` parameter**

Replace `wait_for_all_up` (lines 185-196):

```python
def wait_for_all_up(api_url: str, proc, timeout: float, health_key: str = "all") -> bool:
    """Poll /v1/health until health[health_key] == "up", the restart process dies
    (Compose only; pass proc=None for k8s), or timeout. True iff ready. The
    phase-scaled k8s path passes health_key="megatron" (phase 1) / "vllm" (phase 2)
    because health["all"] is structurally down when only one GPU service is up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        health = get_health(api_url)
        if health and health.get(health_key) == "up":
            return True
        time.sleep(2)
    return False
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k wait_for_all_up -v`
Expected: PASS (3 passed — the two new tests plus the existing `test_wait_for_all_up_accepts_none_proc`, which still passes on the default key)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add per-phase health_key to wait_for_all_up"
```

---

### Task 2: `kubectl scale` command builder + wrapper

Phase 2 hands the GPU over with two `kubectl scale` calls. Add a pure builder and a thin wrapper, matching the existing builder/wrapper pattern.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (builder after `k8s_get_pods_cmd` ~line 331; wrapper after `kubectl_get_pods` ~line 346)
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "scale" -v`
Expected: FAIL — `AttributeError: module 'run_finetune_sweep' has no attribute 'k8s_scale_cmd'`

- [ ] **Step 3: Implement the builder and wrapper**

Add the builder immediately after `k8s_get_pods_cmd` (the last pure builder, ~line 331):

```python
def k8s_scale_cmd(kind_name: str, replicas: int, namespace: str) -> list[str]:
    """`kubectl scale <kind/name> --replicas=<n> -n <ns>`. kind_name is e.g.
    "statefulset/scalarlm-megatron" or "deployment/scalarlm-vllm"."""
    return ["kubectl", "scale", kind_name, f"--replicas={replicas}", "-n", namespace]
```

Add the wrapper immediately after `kubectl_get_pods` (~line 346):

```python
def kubectl_scale(kind_name: str, replicas: int, namespace: str, log, timeout: float = 60) -> bool:
    """Scale a Deployment/StatefulSet. True iff kubectl exits 0."""
    try:
        subprocess.run(k8s_scale_cmd(kind_name, replicas, namespace), cwd=REPO_ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "scale" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add kubectl scale builder + wrapper for phase handoff"
```

---

### Task 3: GPU-release barrier — `kubectl_get_pods` selector + `wait_for_pods_gone`

The GPU is released only when the megatron pod is **fully deleted**, not while `Terminating`. Add a label-selector option to `kubectl_get_pods` and a polling barrier that returns once no megatron pod remains.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (`k8s_get_pods_cmd` ~line 330; `kubectl_get_pods` ~line 336; new constant + `wait_for_pods_gone` near `wait_for_pods_ready` ~line 368)
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
def test_get_pods_cmd_with_selector():
    assert rfs.k8s_get_pods_cmd("sweep-qwen", "app.kubernetes.io/component=megatron") == [
        "kubectl", "get", "pods", "-n", "sweep-qwen",
        "-l", "app.kubernetes.io/component=megatron", "-o", "json"]

def test_get_pods_cmd_without_selector_unchanged():
    assert rfs.k8s_get_pods_cmd("sweep-qwen") == [
        "kubectl", "get", "pods", "-n", "sweep-qwen", "-o", "json"]

def test_wait_for_pods_gone_returns_gone_when_empty(monkeypatch):
    monkeypatch.setattr(rfs, "kubectl_get_pods", lambda ns, selector=None, timeout=15: [])
    assert rfs.wait_for_pods_gone("sweep-qwen", "sel", gpu_wait_timeout=5, poll=0.01) == "gone"

def test_wait_for_pods_gone_times_out_while_present(monkeypatch):
    monkeypatch.setattr(rfs, "kubectl_get_pods",
                        lambda ns, selector=None, timeout=15: [{"metadata": {"name": "megatron-0"}}])
    assert rfs.wait_for_pods_gone("sweep-qwen", "sel", gpu_wait_timeout=0.05, poll=0.01) == "timeout"

def test_megatron_pod_selector_is_component_label():
    assert rfs.MEGATRON_POD_SELECTOR == "app.kubernetes.io/component=megatron"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "pods_gone or selector or MEGATRON" -v`
Expected: FAIL — `TypeError: k8s_get_pods_cmd() takes 1 positional argument but 2 were given` (and missing `wait_for_pods_gone` / `MEGATRON_POD_SELECTOR`)

- [ ] **Step 3: Add the selector and the barrier**

Replace `k8s_get_pods_cmd` (~line 330):

```python
def k8s_get_pods_cmd(namespace: str, selector: str | None = None) -> list[str]:
    cmd = ["kubectl", "get", "pods", "-n", namespace]
    if selector:
        cmd += ["-l", selector]
    return cmd + ["-o", "json"]
```

Replace `kubectl_get_pods` (~line 336) so it forwards the selector:

```python
def kubectl_get_pods(namespace: str, selector: str | None = None, timeout: float = 15) -> list[dict]:
    """Return the namespace's pod objects (`.items`), optionally filtered by a
    label selector, or [] if the call fails."""
    try:
        result = subprocess.run(k8s_get_pods_cmd(namespace, selector), cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    try:
        return json.loads(result.stdout).get("items", [])
    except json.JSONDecodeError:
        return []
```

Add the constant and the barrier immediately after `wait_for_pods_ready` (~line 368):

```python
# Megatron pods carry this component label (chart _helpers.tpl megatronlabels);
# used to wait for the megatron pod to fully delete during the phase-2 handoff.
MEGATRON_POD_SELECTOR = "app.kubernetes.io/component=megatron"


def wait_for_pods_gone(namespace: str, selector: str, gpu_wait_timeout: float,
                       poll: float = 5.0) -> str:
    """Block until no pod matching `selector` exists (its nvidia.com/gpu request is
    released only on full deletion, not while Terminating), else "timeout" at
    gpu_wait_timeout. Phase-2 handoff: megatron must be gone before vLLM claims the
    card. Polling beats `kubectl wait --for=delete`, which errors when zero pods
    already match."""
    deadline = time.time() + gpu_wait_timeout
    while time.time() < deadline:
        if not kubectl_get_pods(namespace, selector):
            return "gone"
        time.sleep(poll)
    return "timeout"
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (all prior tests still green — `kubectl_get_pods` and `wait_for_pods_ready` callers pass only `namespace` positionally, so the new optional `selector` is backward-compatible)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add wait_for_pods_gone GPU-release barrier"
```

---

### Task 4: `replicaCounts` overrides in `k8s_helm_install_cmd`

When `phase_scaled`, the install must bring up megatron only (it holds the single GPU) and leave vLLM at zero replicas — but still render the vLLM Deployment so phase 2 can scale it up.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py:306-313`
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
def test_helm_install_cmd_no_phase_scaling_by_default():
    cmd = rfs.k8s_helm_install_cmd(CUDA_CFG, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert "replicaCounts.inference=0" not in cmd
    assert "replicaCounts.training=1" not in cmd

def test_helm_install_cmd_phase_scaled_sets_replica_counts():
    cfg = {**CUDA_CFG, "phase_scaled": True}
    cmd = rfs.k8s_helm_install_cmd(cfg, "sweep-qwen", "Qwen/Qwen2.5-0.5B")
    assert "replicaCounts.inference=0" in cmd  # vLLM off in phase 0/1
    assert "replicaCounts.training=1" in cmd   # megatron holds the single GPU
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "phase_scal or replica" -v`
Expected: FAIL — `test_helm_install_cmd_phase_scaled_sets_replica_counts` fails (override not appended)

- [ ] **Step 3: Append the overrides when `phase_scaled`**

Replace `k8s_helm_install_cmd` (lines 306-313):

```python
def k8s_helm_install_cmd(target_cfg: dict, namespace: str, model_id: str) -> list[str]:
    cmd = [
        "helm", "upgrade", "--install", target_cfg["release"], target_cfg["chart_path"],
        "-n", namespace, "--create-namespace",
        "--set", f"model={model_id}",
        "--set", "storage.cache.kind=hostPath",
        "--set", f"storage.cache.hostPath={target_cfg['cache_hostpath']}",
    ]
    if target_cfg.get("phase_scaled"):
        # Phase 0: megatron holds the single GPU; vLLM is off until the phase-2
        # handoff. Use replicaCounts, NOT vllm.enabled=false (which drops the
        # Deployment and leaves nothing for `kubectl scale` to bring up).
        cmd += ["--set", "replicaCounts.inference=0", "--set", "replicaCounts.training=1"]
    return cmd
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k "helm or phase_scal or replica" -v`
Expected: PASS (the existing `test_helm_install_cmd` still passes — default path unchanged)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): phase_scaled helm install scales vLLM to 0, megatron to 1"
```

---

### Task 5: Extract `poll_training` (shared by both paths)

The training-status poll loop is identical for the 2-GPU and phase-scaled paths. Extract it (DRY) before building the phased path; refactor the existing `run_model` to use it so the extraction is proven equivalent.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (new helper after `wait_for_all_up` ~line 196; replace the inline loop in `run_model` ~lines 563-575)
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
def test_poll_training_returns_terminal_status(monkeypatch):
    monkeypatch.setattr(rfs, "get_training_job", lambda url, jh, timeout=10: {"status": "COMPLETED"})
    assert rfs.poll_training("http://x", "abc", train_timeout=1) == "COMPLETED"

def test_poll_training_times_out(monkeypatch):
    monkeypatch.setattr(rfs, "get_training_job", lambda url, jh, timeout=10: {"status": "TRAINING"})
    monkeypatch.setattr(rfs.time, "sleep", lambda s: None)
    assert rfs.poll_training("http://x", "abc", train_timeout=0.01) == "TIMEOUT"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k poll_training -v`
Expected: FAIL — `AttributeError: module 'run_finetune_sweep' has no attribute 'poll_training'`

- [ ] **Step 3: Add `poll_training` and use it in `run_model`**

Add the helper immediately after `wait_for_all_up` (~line 196):

```python
def poll_training(api_url: str, job_hash: str, train_timeout: float) -> str:
    """Poll the training job until terminal (COMPLETED/FAILED/CANCELLED) or
    train_timeout. Returns the terminal status or "TIMEOUT". Shared by the 2-GPU
    and phase-scaled k8s paths."""
    deadline = time.time() + train_timeout
    while time.time() < deadline:
        try:
            info = get_training_job(api_url, job_hash)
        except (urllib.error.URLError, RuntimeError):
            time.sleep(5)
            continue
        st = info.get("status")
        if st in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(5)
    return "TIMEOUT"
```

In `run_model`, replace the inline poll loop (currently ~lines 563-575):

```python
            train_status = "TIMEOUT"
            deadline = time.time() + args.train_timeout
            while time.time() < deadline:
                try:
                    info = get_training_job(args.api_url, job_hash)
                except (urllib.error.URLError, RuntimeError):
                    time.sleep(5)
                    continue
                st = info.get("status")
                if st in ("COMPLETED", "FAILED", "CANCELLED"):
                    train_status = st
                    break
                time.sleep(5)
            res.train_seconds = round(time.time() - train_start, 1)
```

with:

```python
            train_status = poll_training(args.api_url, job_hash, args.train_timeout)
            res.train_seconds = round(time.time() - train_start, 1)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (module imports cleanly; new poll_training tests green; all prior tests green)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "refactor(sweep): extract poll_training helper"
```

---

### Task 6: Extract `serve_check_and_classify` (shared by both paths)

The hot-load loop + `classify_result` + detail-mapping block is the same for both paths. Extract it so the phased path doesn't duplicate ~40 lines. The only behavioral nuance: the checkpoint read happens in the caller (it differs per path), so the helper takes `checkpoint_keys` already read.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (new helper after `poll_training`; replace the serve block in `run_model` ~lines 578-617)
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k serve_check -v`
Expected: FAIL — `AttributeError: module 'run_finetune_sweep' has no attribute 'serve_check_and_classify'`

- [ ] **Step 3: Add `serve_check_and_classify` and use it in `run_model`**

Add the helper immediately after `poll_training`:

```python
def serve_check_and_classify(api_url: str, golden_prompt: str, expected_output: str,
                             job_hash: str, train_status: str,
                             checkpoint_keys: list[str] | None, args, res) -> None:
    """Hot-load the trained adapter, classify the outcome, and set the result
    fields (serve_seconds, outcome, adapter_sample, detail). Shared by the 2-GPU
    and phase-scaled k8s paths; the caller reads `checkpoint_keys` first (the read
    mechanism differs per path)."""
    adapter_loaded = False
    adapter_text = ""
    last_serve_error = ""
    serve_start = time.time()
    if train_status == "COMPLETED" and checkpoint_lora_keys_ok(checkpoint_keys):
        serve_deadline = time.time() + args.serve_timeout
        while time.time() < serve_deadline:
            try:
                adapter_text = generate(api_url, [golden_prompt], job_hash, args.max_tokens)[0]
                adapter_loaded = True
                break
            except Exception as e:
                last_serve_error = str(e)
                time.sleep(5)
    res.serve_seconds = round(time.time() - serve_start, 1)

    memorized = expected_output in adapter_text
    res.outcome = classify_result(train_status, checkpoint_keys, adapter_loaded, memorized)
    res.adapter_sample = adapter_text[:200]

    if res.outcome == TRAIN_FAILED:
        res.detail = f"job ended with status {train_status}"
    elif res.outcome == TRAIN_TIMEOUT:
        res.detail = f"job did not reach a terminal status within {args.train_timeout}s"
    elif res.outcome == BAD_CHECKPOINT:
        res.detail = f"checkpoint keys: {checkpoint_keys}"
    elif res.outcome == ADAPTER_NOT_LOADED:
        res.detail = f"adapter never became servable; last error: {last_serve_error}"
    elif res.outcome == NO_MEMORIZATION:
        extra = "adapter served but did not memorize expected_output"
        res.detail = f"{res.detail}; {extra}" if res.detail else extra
```

In `run_model`, replace the serve block (currently ~lines 578-617, from `checkpoint_keys: list[str] | None = None` through the `return res` that ends the `try`):

```python
            checkpoint_keys: list[str] | None = None
            adapter_loaded = False
            adapter_text = ""
            last_serve_error = ""
            serve_start = time.time()  # only meaningful if train_status == "COMPLETED"

            if train_status == "COMPLETED":
                if is_k8s:
                    checkpoint_keys = read_checkpoint_keys_k8s(target_cfg, namespace, job_hash)
                else:
                    checkpoint_keys = read_checkpoint_keys(target_cfg["compose_service"], job_hash)
                if checkpoint_lora_keys_ok(checkpoint_keys):
                    serve_deadline = time.time() + args.serve_timeout
                    while time.time() < serve_deadline:
                        try:
                            adapter_out = generate(args.api_url, [golden_prompt], job_hash, args.max_tokens)
                            adapter_text = adapter_out[0]
                            adapter_loaded = True
                            break
                        except Exception as e:
                            last_serve_error = str(e)
                            time.sleep(5)
            res.serve_seconds = round(time.time() - serve_start, 1)

            memorized = expected_output in adapter_text
            res.outcome = classify_result(train_status, checkpoint_keys, adapter_loaded, memorized)
            res.adapter_sample = adapter_text[:200]

            if res.outcome == TRAIN_FAILED:
                res.detail = f"job ended with status {train_status}"
            elif res.outcome == TRAIN_TIMEOUT:
                res.detail = f"job did not reach a terminal status within {args.train_timeout}s"
            elif res.outcome == BAD_CHECKPOINT:
                res.detail = f"checkpoint keys: {checkpoint_keys}"
            elif res.outcome == ADAPTER_NOT_LOADED:
                res.detail = f"adapter never became servable; last error: {last_serve_error}"
            elif res.outcome == NO_MEMORIZATION:
                extra = "adapter served but did not memorize expected_output"
                res.detail = f"{res.detail}; {extra}" if res.detail else extra
            return res
```

with:

```python
            checkpoint_keys: list[str] | None = None
            if train_status == "COMPLETED":
                if is_k8s:
                    checkpoint_keys = read_checkpoint_keys_k8s(target_cfg, namespace, job_hash)
                else:
                    checkpoint_keys = read_checkpoint_keys(target_cfg["compose_service"], job_hash)
            serve_check_and_classify(args.api_url, golden_prompt, expected_output, job_hash,
                                     train_status, checkpoint_keys, args, res)
            return res
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (module imports cleanly; serve_check tests green; all prior tests green)

Also smoke-check the CLI still parses: `PYTHONPATH="$PWD/infra" uv run --with pyyaml python test/finetune_sweep/run_finetune_sweep.py --help`
Expected: usage text (no import/syntax error).

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "refactor(sweep): extract serve_check_and_classify helper"
```

---

### Task 7: `run_model_k8s_phased` + dispatch

Add the re-sequenced phase-scaled loop and select it from `run_model` when the target sets `phase_scaled`. The structural test monkeypatches every side-effecting wrapper and asserts the phase ordering — specifically that megatron is scaled to 0 (GPU freed) **before** vLLM is scaled to 1.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (new `run_model_k8s_phased` before `run_model`; dispatch line inside `run_model` after `log_path` is set, ~line 510)
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing structural test**

Append to `test/unit/test_finetune_sweep_k8s.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -k phased -v`
Expected: FAIL — `AttributeError: module 'run_finetune_sweep' has no attribute 'run_model_k8s_phased'`

- [ ] **Step 3: Implement `run_model_k8s_phased` and the dispatch**

Add `run_model_k8s_phased` immediately **before** `def run_model(` (~line 484):

```python
def run_model_k8s_phased(target_cfg: dict, model_id: str, train_args: dict, dataset: list[dict],
                         golden_prompt: str, expected_output: str, args, res, log) -> Result:
    """Phase-scaled k8s closed loop (peak GPU = 1). Re-sequences run_model:
    install vLLM=0/megatron=1 -> train (phase 1) -> hand the GPU to vLLM (phase 2)
    -> baseline + hot-load + memorization check. See the 2026-06-16 amendment in
    docs/adr/0003-finetune-sweep-restart-per-model.md."""
    namespace = k8s_namespace(target_cfg["namespace_prefix"], model_id)
    gpu_wait = target_cfg.get("gpu_wait_timeout", args.gpu_wait_timeout)
    pf_proc = None
    restart_start = time.time()
    try:
        # --- Phase 0: install; megatron holds the single GPU, vLLM is off. ---
        delete_namespace(namespace, log)  # idempotent pre-clean
        if not helm_install(target_cfg, namespace, model_id, log, args.restart_timeout):
            res.restart_seconds = round(time.time() - restart_start, 1)
            res.outcome, res.detail = RESTART_FAILED, "helm upgrade --install failed"
            return res
        if wait_for_pods_ready(namespace, gpu_wait) != "ready":
            res.restart_seconds = round(time.time() - restart_start, 1)
            res.outcome, res.detail = RESTART_FAILED, f"megatron not schedulable within {gpu_wait}s"
            return res
        pf_proc = start_port_forward(target_cfg, namespace, log)
        # Gate on slurm-registered megatron (NOT health.all -- vLLM is down).
        if not wait_for_all_up(args.api_url, pf_proc, args.restart_timeout, health_key="megatron"):
            res.restart_seconds = round(time.time() - restart_start, 1)
            res.outcome, res.detail = RESTART_FAILED, "megatron health not up after port-forward"
            return res
        res.restart_seconds = round(time.time() - restart_start, 1)

        # --- Phase 1: train (vLLM absent; baseline deferred to phase 2). ---
        train_start = time.time()
        try:
            job_status = submit_train(args.api_url, dataset, train_args)
        except Exception as e:
            res.outcome, res.detail = TRAIN_FAILED, f"submit_train failed: {e}"
            return res
        job_hash = job_status["job_directory"].rstrip("/").split("/")[-1]
        train_status = poll_training(args.api_url, job_hash, args.train_timeout)
        res.train_seconds = round(time.time() - train_start, 1)

        checkpoint_keys: list[str] | None = None
        if train_status == "COMPLETED":
            checkpoint_keys = read_checkpoint_keys_k8s(target_cfg, namespace, job_hash)

        # --- Phase 2: hand the GPU from megatron to vLLM. ---
        if not kubectl_scale(f"statefulset/{target_cfg['megatron_sts']}", 0, namespace, log):
            res.outcome, res.detail = RESTART_FAILED, "kubectl scale megatron->0 failed"
            return res
        if wait_for_pods_gone(namespace, MEGATRON_POD_SELECTOR, gpu_wait) != "gone":
            res.outcome, res.detail = RESTART_FAILED, f"megatron did not release GPU within {gpu_wait}s"
            return res
        if not kubectl_scale(f"deployment/{target_cfg['vllm_deploy']}", 1, namespace, log):
            res.outcome, res.detail = RESTART_FAILED, "kubectl scale vllm->1 failed"
            return res
        if wait_for_pods_ready(namespace, gpu_wait) != "ready":
            res.outcome, res.detail = RESTART_FAILED, f"vLLM not schedulable within {gpu_wait}s"
            return res
        # Gate on vLLM serving (NOT health.all -- megatron is down now).
        if not wait_for_all_up(args.api_url, pf_proc, args.serve_timeout, health_key="vllm"):
            res.outcome, res.detail = RESTART_FAILED, "vllm health not up after scale-up"
            return res

        # Baseline (control on the base model) -- moved here: vLLM is now up.
        try:
            baseline = generate(args.api_url, [golden_prompt], model_id, args.max_tokens)
            res.baseline_sample = baseline[0][:200]
            if expected_output in baseline[0]:
                res.detail = "expected_output already present in baseline output"
        except Exception as e:
            res.outcome, res.detail = RESTART_FAILED, f"baseline generate failed: {e}"
            return res

        # Hot-load + classify (shared with the 2-GPU path).
        serve_check_and_classify(args.api_url, golden_prompt, expected_output, job_hash,
                                 train_status, checkpoint_keys, args, res)
        return res
    finally:
        if pf_proc is not None:
            try:
                os.killpg(os.getpgid(pf_proc.pid), signal.SIGKILL)
            except OSError:
                pass
            pf_proc.wait()
        delete_namespace(namespace, log)
```

Then add the dispatch in `run_model`. Find (currently ~lines 509-510):

```python
    log_path = results_dir / f"{model_id.replace('/', '_')}.{target}.restart.log"
    namespace = pf_proc = proc = None
```

Replace with:

```python
    log_path = results_dir / f"{model_id.replace('/', '_')}.{target}.restart.log"
    if is_k8s and target_cfg.get("phase_scaled"):
        with open(log_path, "w") as log:
            return run_model_k8s_phased(target_cfg, model_id, train_args, dataset,
                                        golden_prompt, expected_output, args, res, log)
    namespace = pf_proc = proc = None
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH="$PWD/infra" uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -v`
Expected: PASS (phased structural test green; all prior tests green)

Also confirm the module still imports and the CLI parses:
`PYTHONPATH="$PWD/infra" uv run --with pyyaml python test/finetune_sweep/run_finetune_sweep.py --help`
Expected: usage text including `--gpu-wait-timeout`.

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add phase-scaled 1-GPU run_model path"
```

---

### Task 8: Manifest — opt the `cuda` target into phase-scaling

`blackwell-maxq-0` has one schedulable GPU, so the `cuda` target opts into phase-scaling. Add the `vllm_deploy` scale target and the `phase_scaled` flag. (Removing the `phase_scaled` line reverts to the 2-GPU path with no code change.)

**Files:**
- Modify: `test/finetune_sweep/finetune-sweep.yaml`

- [ ] **Step 1: Add `vllm_deploy` after the `megatron_sts` line**

Find:

```yaml
    megatron_sts: scalarlm-megatron  # kubectl exec target (StatefulSet)
    api_service: scalarlm          # svc for port-forward (== release fullname)
```

Replace with:

```yaml
    megatron_sts: scalarlm-megatron  # kubectl exec target (StatefulSet)
    vllm_deploy: scalarlm-vllm     # phase-2 kubectl scale target (Deployment)
    api_service: scalarlm          # svc for port-forward (== release fullname)
```

- [ ] **Step 2: Add `phase_scaled` after the `gpu_wait_timeout` line**

Find:

```yaml
    gpu_wait_timeout: 7200   # k8s block-and-wait cap (s); CLI --gpu-wait-timeout overrides
    train_args_overrides:
```

Replace with:

```yaml
    gpu_wait_timeout: 7200   # k8s block-and-wait cap (s); CLI --gpu-wait-timeout overrides
    # 1-GPU phase-scaled loop: train (megatron) then hand the GPU to vLLM, peak
    # GPU = 1. See docs/superpowers/specs/2026-06-15-finetune-sweep-1gpu-phase-scaled-design.md
    # and the 2026-06-16 amendment in docs/adr/0003-finetune-sweep-restart-per-model.md.
    # Remove this line to fall back to the 2-GPU (always-on vLLM + megatron) path.
    phase_scaled: true
    train_args_overrides:
```

- [ ] **Step 3: Verify the manifest loads and the cuda target is recognized as phase-scaled**

Run:
```bash
PYTHONPATH="$PWD/infra" uv run --with pyyaml python -c "
import importlib.util, pathlib
s = importlib.util.spec_from_file_location('rfs', 'test/finetune_sweep/run_finetune_sweep.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
mani = m.load_manifest(pathlib.Path('test/finetune_sweep/finetune-sweep.yaml'))
cuda = mani['targets']['cuda']
assert m.is_k8s_target(cuda), 'cuda should be a k8s target'
assert cuda.get('phase_scaled') is True, 'cuda should be phase-scaled'
assert cuda['vllm_deploy'] == 'scalarlm-vllm', 'cuda needs a vllm_deploy scale target'
assert m.k8s_helm_install_cmd(cuda, 'sweep-x', 'M').count('--set') >= 5, 'phase install needs replica overrides'
print('OK: cuda is phase-scaled k8s, vllm_deploy=', cuda['vllm_deploy'])
"
```
Expected: `OK: cuda is phase-scaled k8s, vllm_deploy= scalarlm-vllm`

- [ ] **Step 4: Commit**

```bash
git add test/finetune_sweep/finetune-sweep.yaml
git commit -m "feat(sweep): opt cuda target into 1-GPU phase-scaled mode"
```

---

### Task 9: Integration validation on `blackwell-maxq-0` (manual)

This cannot be unit-tested here — it needs the live single-GPU cluster. The runner is GPU-less and run from the host, exactly as the team runs `helm`. Hand the operator these steps.

- [ ] **Step 1: Confirm prerequisites and that exactly one GPU is the constraint**

```bash
helm version && kubectl version --client && kubectl get nodes
# How many schedulable nvidia.com/gpu does the node advertise / how many are free?
kubectl get pods -A -o json | python3 -c "import sys,json; print('gpu pods:', sum(1 for p in json.load(sys.stdin)['items'] for c in p['spec']['containers'] if c.get('resources',{}).get('limits',{}).get('nvidia.com/gpu')))"
```
Confirm the `cache_hostpath` in the manifest matches the node's populated HF cache.

- [ ] **Step 2: Ensure no Compose GPU stack is running (the contention this prevents)**

```bash
docker compose ps 2>/dev/null | grep -i cray-nvidia && echo "STOP: decommission the Compose GPU stack first" || echo "clear"
```

- [ ] **Step 3: Run the sweep against the phase-scaled cuda target**

```bash
cd <repo-on-box>
PYTHONPATH="$PWD/infra" uv run --with pyyaml python test/finetune_sweep/run_finetune_sweep.py \
  --target cuda --gpu-wait-timeout 7200
```

- [ ] **Step 4: Observe peak GPU = 1 across the phases in a second shell**

```bash
watch 'kubectl -n sweep-qwen-qwen2-5-0-5b get pods -o wide; echo; nvidia-smi --query-gpu=memory.used --format=csv'
```
Expected progression: **phase 1** — only megatron + api pods (vLLM at 0 replicas), one GPU in use; training runs. **handoff** — megatron pod Terminates and disappears before any vLLM pod appears. **phase 2** — vLLM + api pods, one GPU in use; adapter served. At no point are megatron and vLLM both holding a GPU.

- [ ] **Step 5: Confirm the outcome and clean teardown**

```bash
ls test/finetune_sweep/results/*.cuda.* 2>/dev/null  # per-model json/md written
kubectl get ns | grep sweep- || echo "namespaces cleaned"
kubectl get pvc -A | grep sweep- || echo "no leaked PVCs"
```
Expected: a results row per model; the `sweep-*` namespace and its PVCs are gone.

- [ ] **Step 6: Verify the open assumptions and record any reality gaps**

Confirm on this run that (a) scaling megatron to 0 did **not** break adapter discovery (the serve check still loaded the adapter from the jobs PVC), and (b) the megatron→0 / vLLM→1 handoff didn't deadlock on a co-tenant grabbing the freed GPU. If `cache_hostpath`, `vllm_deploy`, `megatron_sts`, or the component label differ on the box, update `finetune-sweep.yaml` / `MEGATRON_POD_SELECTOR` and the spec's "Resolved" section, then commit:

**Real-cluster caveats unit tests cannot catch (verify these explicitly — a wrong value fails silently):**

1. **`MEGATRON_POD_SELECTOR` MUST match the chart's real pod label.** If `kubectl get pods -n <ns> -l app.kubernetes.io/component=megatron` returns *nothing* because the label differs, `wait_for_pods_gone` sees an empty list and returns `"gone"` **immediately** — vLLM then scales up while megatron still holds the GPU (the exact deadlock this barrier exists to prevent). Verify the selector matches a real megatron pod *before* trusting a green run.
2. **The `api` (GPU-less) pod must survive the handoff.** The single `pf_proc` port-forward (to `svc/scalarlm`) is reused across both phases. If the chart ties the api Deployment's replicas to `replicaCounts.{training,inference}`, the api pod (and the port-forward) drops mid-handoff and phase 2 fails as `"vllm health not up after scale-up"` — a misleading symptom. Confirm the api pod stays Ready throughout.
3. **Resource names must match:** `kubectl get deploy,sts -n <ns>` must show `scalarlm-vllm` (Deployment) and `scalarlm-megatron` (StatefulSet); otherwise the `kubectl scale` calls no-op or error.
4. **The jobs PVC must actually provision `ReadWriteMany`** on the node's StorageClass — vLLM reads the checkpoint megatron wrote, and a RWO claim still in Terminating could block vLLM binding.

```bash
git add test/finetune_sweep/finetune-sweep.yaml test/finetune_sweep/run_finetune_sweep.py \
        docs/superpowers/specs/2026-06-15-finetune-sweep-1gpu-phase-scaled-design.md
git commit -m "fix(sweep): reconcile phase-scaled target with blackwell-maxq-0 layout"
```

---

## Self-Review

**Spec coverage:**
- Opt-in `phase_scaled: true` on the k8s cuda target → Task 8 (manifest) + Task 7 (dispatch). ✓
- Install with vLLM=0 / megatron=1 → Task 4 (`k8s_helm_install_cmd` replica overrides) + Task 7 (phase 0). ✓
- Train phase (vLLM absent), checkpoint read here → Task 7 (phase 1, `poll_training` + `read_checkpoint_keys_k8s`). ✓
- `kubectl scale` megatron→0 and vLLM→1 → Task 2 (`k8s_scale_cmd` / `kubectl_scale`) + Task 7. ✓
- Scale-down releases GPU before scale-up (wait pod gone) → Task 3 (`wait_for_pods_gone` + `MEGATRON_POD_SELECTOR`) + Task 7. ✓
- Per-phase health gate, never `health.all` (`megatron` then `vllm`) → Task 1 (`health_key`) + Task 7. ✓
- Baseline moved into phase 2 → Task 7 (baseline after the vLLM health gate). ✓
- Block-and-wait / `gpu_wait_timeout` reused per phase, timeout → `RESTART_FAILED` → Task 7 (every gate maps to RESTART_FAILED with a phase-specific detail). ✓
- Teardown via `kubectl delete namespace` (kills port-forward first), unchanged → Task 7 `finally`. ✓
- No chart change required → confirmed; no chart task exists. ✓
- 2-GPU path and `cpu`/Compose path unchanged → Tasks 5/6 are behavior-preserving extractions proven by the existing tests; Task 7 dispatch only fires when `phase_scaled` is set; Task 4/1 are no-ops on the default path. ✓
- Resolved open questions (re-acquire race accepted; megatron not needed at serve time; flag not separate target) → Task 8 (flag) + Task 7 (block-and-wait on re-acquire); the megatron-not-needed fact is what makes Task 7 phase 2 correct. ✓

**Placeholder scan:** no TBD/TODO; every code step contains full code; every command has expected output. ✓

**Type/name consistency:** `target_cfg` keys (`chart_path`, `release`, `namespace_prefix`, `megatron_sts`, `vllm_deploy`, `api_service`, `cache_hostpath`, `gpu_wait_timeout`, `phase_scaled`) are identical across the manifest (Task 8), the builders (Tasks 2/4), and `run_model_k8s_phased` (Task 7). New functions: `k8s_scale_cmd`, `kubectl_scale`, `wait_for_pods_gone`, `MEGATRON_POD_SELECTOR`, `poll_training`, `serve_check_and_classify`, `run_model_k8s_phased` — each defined once and referenced consistently. `wait_for_all_up(..., health_key=...)` signature matches all call sites (default `"all"` for the two existing sites; `"megatron"`/`"vllm"` in Task 7). `wait_for_pods_gone` returns `"gone"`/`"timeout"`; `wait_for_pods_ready` returns `"ready"`/`"failed"`/`"timeout"` — distinct and mapped correctly in Task 7. ✓
