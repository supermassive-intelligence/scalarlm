# cuda-docker GPU-Compose Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testing-only `cuda-docker` target that runs the fine-tune sweep's GPU closed loop via Docker Compose on a scheduler-less GPU box (the 3090), rename the existing k8s `cuda` target to `cuda-k8s`, and make the runner's target dispatch config-driven instead of keyed on literal target names.

**Architecture:** The runner's Compose lifecycle is already GPU-capable (it once drove `cuda` before the k8s migration). The work is mostly manifest config plus replacing three literal-name dispatch branches with a single config-driven predicate (`target_requests_gpu`), mirroring the existing `is_k8s_target`. No new lifecycle code; the Compose path co-locates vLLM + slurm-launched training in one `cray-nvidia` container sharing the single GPU, so no phase-scaling is needed.

**Tech Stack:** Python 3.11 (stdlib + PyYAML), pytest, Docker Compose, the `scalarlm` bashly CLI. Tests run via `uv`.

**Spec:** `docs/superpowers/specs/2026-06-16-finetune-sweep-cuda-docker-target-design.md`

---

## File Structure

- `test/finetune_sweep/run_finetune_sweep.py` — runner. Add `target_requests_gpu`; refactor `gate_model` signature; rewire `run_model`'s probe + gate call; drop `--target` choices; bump `--restart-timeout` default; fix three stale `model_sweep/run_sweep.py` docstring references; update the module-docstring usage examples.
- `test/finetune_sweep/finetune-sweep.yaml` — manifest. Rename `cuda` → `cuda-k8s`; add `cuda-docker`.
- `test/unit/test_finetune_sweep_k8s.py` — unit tests. Update the three `gate_model(model, "cuda", …)` calls to the new signature; add `target_requests_gpu` truth-table tests, a CPU-branch gate test, and a manifest-dispatch test.
- `docs/adr/0003-finetune-sweep-restart-per-model.md` — append the 2026-06-16 `cuda-docker` amendment.

**Test command (used throughout):**
```bash
PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q
```
(These are pure-helper tests — no torch needed. Baseline before any change: `48 passed`.)

---

## Task 1: Add `target_requests_gpu` config predicate

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py` (insert after `is_k8s_target`, currently ending at line 363)
- Test: `test/unit/test_finetune_sweep_k8s.py`

- [ ] **Step 1: Write the failing tests**

Add to `test/unit/test_finetune_sweep_k8s.py` (after the `is_k8s_target` tests, or at end of file):

```python
def test_target_requests_gpu_true_when_gpus_one():
    assert rfs.target_requests_gpu({"train_args_overrides": {"gpus": 1}}) is True

def test_target_requests_gpu_false_when_gpus_zero():
    assert rfs.target_requests_gpu({"train_args_overrides": {"gpus": 0}}) is False

def test_target_requests_gpu_defaults_true_when_unset():
    # JobConfig defaults gpus=1, so a target with no override requests a GPU.
    assert rfs.target_requests_gpu({}) is True
    assert rfs.target_requests_gpu({"train_args_overrides": {}}) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q -k target_requests_gpu`
Expected: FAIL with `AttributeError: module 'run_finetune_sweep' has no attribute 'target_requests_gpu'`

- [ ] **Step 3: Write the implementation**

In `test/finetune_sweep/run_finetune_sweep.py`, immediately after the `is_k8s_target` function (after line 363), add:

```python
def target_requests_gpu(target_cfg: dict) -> bool:
    """True iff this target runs training on a GPU. Config-driven (not keyed on
    the target name), mirroring is_k8s_target: JobConfig defaults gpus=1 and only
    the cpu target overrides it to 0, so this is True for the GPU targets
    (cuda-docker, cuda-k8s) and False for cpu."""
    return target_cfg.get("train_args_overrides", {}).get("gpus", 1) >= 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q -k target_requests_gpu`
Expected: `3 passed` (the three new test functions)

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): add config-driven target_requests_gpu predicate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Refactor `gate_model` to a config-driven signature

`gate_model(model, target, free_gb)` currently branches on `target == "cpu"`. Change it to `gate_model(model, target_cfg, free_gb)` and branch on `target_requests_gpu(target_cfg)`. This breaks the three existing `gate_model(model, "cuda", …)` tests, which are updated in the same task (red→green).

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py:96-114` (the `gate_model` function)
- Test: `test/unit/test_finetune_sweep_k8s.py:140-153` (existing) + new CPU-branch test

- [ ] **Step 1: Update the existing tests + add a CPU-branch test (these now fail)**

Replace the three existing gate tests at `test/unit/test_finetune_sweep_k8s.py:140-153`:

```python
def test_gate_model_k8s_skips_vram_check_when_free_gb_none():
    model = {"id": "m", "adapters": {"lora": {"gate_gb": 8}}}
    ok, reason = rfs.gate_model(model, {"train_args_overrides": {"gpus": 1}}, None)
    assert ok is True and reason == ""

def test_gate_model_k8s_still_requires_gate_gb_declared():
    model = {"id": "m"}  # no adapters.lora.gate_gb
    ok, reason = rfs.gate_model(model, {"train_args_overrides": {"gpus": 1}}, None)
    assert ok is False and "gate_gb" in reason

def test_gate_model_gpu_still_gates_on_free_gb_list():
    model = {"id": "m", "adapters": {"lora": {"gate_gb": 8}}}
    gpu_cfg = {"train_args_overrides": {"gpus": 1}}
    assert rfs.gate_model(model, gpu_cfg, [4.0])[0] is False   # not enough free
    assert rfs.gate_model(model, gpu_cfg, [16.0])[0] is True   # enough free

def test_gate_model_cpu_requires_cpu_ok_optin():
    cpu_cfg = {"train_args_overrides": {"gpus": 0}}
    assert rfs.gate_model({"id": "m"}, cpu_cfg, [])[0] is False           # no cpu_ok
    assert rfs.gate_model({"id": "m", "cpu_ok": True}, cpu_cfg, [])[0] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q -k gate_model`
Expected: FAIL — the GPU-branch tests pass a dict where the old code does `target == "cpu"` (a dict never equals `"cpu"`, so it falls through to the VRAM gate — `test_gate_model_cpu_requires_cpu_ok_optin` fails because the cpu cfg dict is treated as a GPU target).

- [ ] **Step 3: Rewrite `gate_model`**

Replace `test/finetune_sweep/run_finetune_sweep.py:96-114` in full:

```python
def gate_model(model: dict, target_cfg: dict, free_gb: list[float] | None) -> tuple[bool, str]:
    """Decide whether `model` should run on this target. A CPU target (no GPU
    requested) is opt-in via cpu_ok; a GPU target is gated on the LoRA VRAM gate
    vs. probed free VRAM. `free_gb is None` means the VRAM check is not applicable
    (k8s: the scheduler arbitrates GPU fit) — only the static checks apply.
    Branches on target config (target_requests_gpu), not the literal target name."""
    if not target_requests_gpu(target_cfg):
        if not model.get("cpu_ok"):
            return False, "no cpu_ok opt-in for this model"
        return True, ""

    gate_gb = model.get("adapters", {}).get("lora", {}).get("gate_gb")
    if gate_gb is None:
        return False, "no adapters.lora.gate_gb declared"
    if free_gb is None:
        return True, ""  # k8s: scheduler arbitrates GPU fit; skip the VRAM check
    if not free_gb or max(free_gb) < gate_gb:
        return False, (f"LoRA needs >={gate_gb:g}GiB free; "
                        f"free GiB: {[round(f, 1) for f in free_gb]}")
    return True, ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q -k gate_model`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep_k8s.py
git commit -m "refactor(sweep): make gate_model config-driven (target_cfg, not name)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Rewire `run_model` dispatch + argparse defaults

Update `run_model`'s VRAM probe and `gate_model` call to the config-driven form, drop the hardcoded `--target` choices, and raise `--restart-timeout` default to 3000.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py:692-693` (probe + gate call), `:814` (`--target`), `:820` (`--restart-timeout`)

- [ ] **Step 1: Rewire the probe and gate call**

In `run_model`, replace lines 692-693:

```python
    free_gb = None if is_k8s else (probe_gpu_free_gb() if target == "cuda" else [])
    ok, reason = gate_model(model, target, free_gb)
```

with:

```python
    free_gb = None if is_k8s else (probe_gpu_free_gb() if target_requests_gpu(target_cfg) else [])
    ok, reason = gate_model(model, target_cfg, free_gb)
```

(`target_cfg` is already defined at line 687; `is_k8s` at line 688.)

- [ ] **Step 2: Drop the hardcoded `--target` choices**

Replace line 814:

```python
    ap.add_argument("--target", required=True, choices=["cpu", "cuda"])
```

with:

```python
    ap.add_argument("--target", required=True,
                    help="manifest target key (e.g. cpu, cuda-docker, cuda-k8s); "
                         "validated against the manifest after load")
```

(The existing post-load check at lines 829-830 — `if args.target not in manifest["targets"]: ap.error(...)` — already validates the value.)

- [ ] **Step 3: Raise the `--restart-timeout` default**

Replace line 820:

```python
    ap.add_argument("--restart-timeout", type=int, default=600)
```

with:

```python
    ap.add_argument("--restart-timeout", type=int, default=3000,
                    help="health-wait cap (s). Default 3000 (50 min) covers a cold "
                         "`./scalarlm up nvidia --build` (vLLM compiles from source) "
                         "on a fresh GPU box; only a ceiling, so happy paths are "
                         "unaffected. Subsequent --force-recreate builds are cache hits.")
```

- [ ] **Step 4: Run the full unit suite to verify nothing regressed**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q`
Expected: all tests pass — `52 passed` (48 baseline + 3 from Task 1 + net 1 from Task 2).

- [ ] **Step 5: Verify the CLI accepts an arbitrary target name (no choices error)**

Run: `cd /home/georgi/projects/scalarlm && python3 test/finetune_sweep/run_finetune_sweep.py --target cuda-docker --models __none__ 2>&1 | head -5`
Expected: it loads the manifest and runs (printing a header / "unknown target" only if the manifest lacks the key — which it still does until Task 4). Specifically it must NOT fail with argparse `invalid choice: 'cuda-docker'`. A `ap.error("unknown target 'cuda-docker'…")` here is the expected pre-Task-4 state and confirms choices were dropped.

- [ ] **Step 6: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py
git commit -m "feat(sweep): config-driven run_model dispatch; --target any manifest key; restart-timeout 3000

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Manifest — rename `cuda` → `cuda-k8s`, add `cuda-docker`

**Files:**
- Modify: `test/finetune_sweep/finetune-sweep.yaml` (the `targets:` block)
- Test: `test/unit/test_finetune_sweep_k8s.py` (manifest-dispatch test)

- [ ] **Step 1: Write the failing manifest-dispatch test**

Add to `test/unit/test_finetune_sweep_k8s.py`:

```python
def test_manifest_targets_dispatch_correctly():
    manifest = rfs.load_manifest(rfs.DEFAULT_MANIFEST)
    targets = manifest["targets"]
    # cuda-docker: Compose lifecycle (not k8s) + GPU probe path
    assert "cuda-docker" in targets
    assert rfs.is_k8s_target(targets["cuda-docker"]) is False
    assert rfs.target_requests_gpu(targets["cuda-docker"]) is True
    assert targets["cuda-docker"]["compose_service"] == "cray-nvidia"
    # cuda-k8s: k8s lifecycle (renamed from cuda) + GPU
    assert "cuda-k8s" in targets and "cuda" not in targets
    assert rfs.is_k8s_target(targets["cuda-k8s"]) is True
    assert rfs.target_requests_gpu(targets["cuda-k8s"]) is True
    # cpu: Compose lifecycle, no GPU
    assert rfs.is_k8s_target(targets["cpu"]) is False
    assert rfs.target_requests_gpu(targets["cpu"]) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q -k manifest_targets_dispatch`
Expected: FAIL — `assert "cuda-docker" in targets` (the manifest still has `cuda`, not `cuda-docker`/`cuda-k8s`).

- [ ] **Step 3: Edit the manifest**

In `test/finetune_sweep/finetune-sweep.yaml`, in the `targets:` block, add the `cuda-docker` target between `cpu` and the current `cuda`, and rename the `cuda:` key to `cuda-k8s:`. Insert this block immediately before the `cuda:` line:

```yaml
  cuda-docker:
    # Testing target for a GPU box that runs NO Helm/k8s scheduler (the 3090).
    # The cray-nvidia Compose service co-locates vLLM (in-process) + the
    # slurm-launched training job in one container, sharing the single GPU — no
    # phase-scaling needed. Outside the "no GPU work outside the scheduler"
    # directive's scope (no scheduler here to subvert). GUARDRAIL: never run on a
    # k8s node (blackwell-maxq-0) -- there it WOULD contend with the scheduler.
    # cuda-k8s stays the canonical path for the cluster. See the 2026-06-16
    # amendment in docs/adr/0003-finetune-sweep-restart-per-model.md.
    compose_service: cray-nvidia
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up nvidia"
    train_args_overrides:
      gpus: 1
```

Then change the line `  cuda:` to `  cuda-k8s:` (the comment line directly above it still applies; leave the rest of that target's body unchanged).

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q -k manifest_targets_dispatch`
Expected: `1 passed`

- [ ] **Step 5: Run the full suite + sanity-check the manifest parses via the runner**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q`
Expected: all pass (`53 passed`).

Run: `cd /home/georgi/projects/scalarlm && python3 test/finetune_sweep/run_finetune_sweep.py --target nope 2>&1 | tail -2`
Expected: `ap.error` listing the targets, e.g. `unknown target 'nope'; have ['cpu', 'cuda-docker', 'cuda-k8s']`.

- [ ] **Step 6: Commit**

```bash
git add test/finetune_sweep/finetune-sweep.yaml test/unit/test_finetune_sweep_k8s.py
git commit -m "feat(sweep): rename cuda -> cuda-k8s, add cuda-docker Compose GPU target

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Doc cleanups (stale docstrings + usage examples)

Fix the three stale references to the deleted `test/model_sweep/run_sweep.py`, and update the module-docstring usage examples to the new target names. Pure documentation — no behavior change, no test.

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py:13-16` (usage), `:516`, `:530`, `:538` (stale refs)

- [ ] **Step 1: Update the module-docstring usage examples**

Replace lines 13-16:

```python
Usage:
    python3 run_finetune_sweep.py --target cpu
    python3 run_finetune_sweep.py --target cuda --models tiny-random/gemma-4-dense
"""
```

with:

```python
Usage:
    python3 run_finetune_sweep.py --target cpu
    python3 run_finetune_sweep.py --target cuda-docker            # GPU via Docker Compose (scheduler-less box)
    python3 run_finetune_sweep.py --target cuda-k8s --models Qwen/Qwen2.5-0.5B
"""
```

- [ ] **Step 2: Fix the `probe_gpu_free_gb` stale reference**

At line 516, replace:

```python
    teardown_stack — same role as probe_gpu_free_gb in test/model_sweep/run_sweep.py,
    but via nvidia-smi (no torch dependency on the host)."""
```

with:

```python
    teardown_stack — via nvidia-smi (no torch dependency on the host)."""
```

- [ ] **Step 3: Fix the `start_restart` stale reference**

At line 530, replace:

```python
    group, non-blocking (mirrors run_sweep.py:279-280)."""
```

with:

```python
    group, non-blocking."""
```

- [ ] **Step 4: Fix the `teardown_stack` stale reference + record the GPU caveat**

At line 538, replace:

```python
    stop climbing (mirrors teardown_engine in test/model_sweep/run_sweep.py)."""
```

with:

```python
    stop climbing. NOTE: `./scalarlm up` runs `docker compose up` in the
    foreground, so SIGKILL stops the compose CLI but leaves the container (and its
    GPU) running; the next run's --force-recreate reclaims it, or `docker compose
    down <service>` does."""
```

- [ ] **Step 5: Verify the module still imports and tests pass**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q`
Expected: all pass (`53 passed`) — confirms no docstring edit broke the triple-quoted strings.

- [ ] **Step 6: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py
git commit -m "docs(sweep): fix stale model_sweep refs; cuda-docker/cuda-k8s usage examples

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Amend ADR 0003

Append the 2026-06-16 amendment recording the `cuda-docker` target, the rename, and the directive scope boundary + guardrail. Documentation only.

**Files:**
- Modify: `docs/adr/0003-finetune-sweep-restart-per-model.md` (append at end of file)

- [ ] **Step 1: Append the amendment**

Add to the end of `docs/adr/0003-finetune-sweep-restart-per-model.md`:

```markdown

## Amendment 2026-06-16 — testing-only `cuda-docker` target (scheduler-less GPU box); `cuda` → `cuda-k8s`

**Status:** the k8s `cuda` target is renamed **`cuda-k8s`** and remains the
canonical GPU path. A sibling **`cuda-docker`** target is added: a testing-only
path that runs the closed loop on a GPU box with **no Helm/k8s scheduler** (the
3090) via the existing Compose lifecycle. See
`docs/superpowers/specs/2026-06-16-finetune-sweep-cuda-docker-target-design.md`.

**Why it does not contradict the 2026-06-15 directive.** That amendment recorded
"no GPU work outside the scheduler" as a constraint of a **scheduler-managed
box**: on `blackwell-maxq-0`, a Compose vLLM container occupies a GPU outside
k8s's knowledge, so k8s can double-place a pod → CUDA OOM / contention.
`cuda-docker` targets a box that runs **no scheduler at all**, so there is nothing
to subvert and no pods to double-place. It is therefore **outside the directive's
scope**, not a deviation from it. **Guardrail:** `cuda-docker` must **never** be
run on a k8s node — there it would reproduce exactly the behind-the-scheduler
contention the directive forbids.

**Why no phase-scaling (unlike `cuda-k8s`).** `cuda-k8s` needs the phase-scaled
handoff because vLLM and megatron are separate GPU pods. The Compose
`cray-nvidia` service runs `one_server.main`, which brings up the API + vLLM
in-process and dispatches training as a slurm job in the **same container**, so
server and trainer share the one GPU simultaneously — a co-located run. No
`replicaCounts`, no `kubectl scale`, no handoff.

**What changes (mechanism).** Target dispatch in the runner becomes config-driven
(`target_requests_gpu`, mirroring `is_k8s_target`) rather than keyed on the
literal target name, so a CPU/GPU/k8s target is distinguished by its config
(`train_args_overrides.gpus` and `chart_path`), not its name. `--restart-timeout`
defaults to 3000s to cover a cold `./scalarlm up nvidia --build` (vLLM compiles
from source). `teardown_stack` is unchanged and shared with `cpu`; note that
SIGKILL of the foreground `docker compose up` leaves the `cray-nvidia` container
holding the GPU until the next `--force-recreate` or a manual `docker compose
down cray-nvidia`.

**Naming.** Historical mentions of the `cuda` target in earlier amendments and
specs are left as-is (they describe the state at their date); the rename is
recorded forward here only.
```

- [ ] **Step 2: Verify it renders (headers, no broken fences)**

Run: `grep -n "^## Amendment" docs/adr/0003-finetune-sweep-restart-per-model.md`
Expected: three amendment headers, the last being the 2026-06-16 `cuda-docker` one.

- [ ] **Step 3: Commit**

```bash
git add docs/adr/0003-finetune-sweep-restart-per-model.md
git commit -m "docs(adr): amend 0003 with cuda-docker target + cuda->cuda-k8s rename

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the full unit suite one last time**

Run: `PYTHONPATH=infra uv run --with pytest --with pyyaml python -m pytest test/unit/test_finetune_sweep_k8s.py -q`
Expected: `53 passed`.

- [ ] **Confirm no literal-name target dispatch remains in the runner**

Run: `grep -n '== "cuda"\|== "cpu"\|choices=\[' test/finetune_sweep/run_finetune_sweep.py`
Expected: no matches (all dispatch is now via `is_k8s_target` / `target_requests_gpu`).

- [ ] **Integration (manual, on the 3090 — not part of CI):** `python3 run_finetune_sweep.py --target cuda-docker`. First run compiles the vLLM image (covered by the 3000s `--restart-timeout`); confirm `nvidia-smi` shows the `cray-nvidia` container on the card and the result reaches PASS/NO_MEMORIZATION. Afterwards free the card: `docker compose -f docker-compose.yaml down cray-nvidia`. **Do not run this on `blackwell-maxq-0`.**
