# Phased serving of a 30B MoE on the DGX Spark: four stacked fixes to a first PASS

**Date:** 2026-07-01
**Target:** `cuda-spark` (NVIDIA DGX Spark, GB10, 128 GiB unified memory)
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (`Qwen3MoeForCausalLM`, 30B/3B-active)
**Outcome:** **PASS** — first real large-MoE end-to-end memorization on the phased path;
exact-architecture validation of the expert-LoRA converter at 30B scale.
**Branch:** `georgi/finetune-sweep` (lab branch; fixes to be extracted to short-lived PRs)

---

## TL;DR

Bringing the finetune-sweep's phase-scaled `cuda-spark` path to a *real* 30B MoE
surfaced **four independent bugs, each of which masked the next**. Fixing them in
sequence took the run from an instant false failure to a clean PASS in which the
served 30B reproduced the golden string character-for-character:

```
| Model                            | Outcome | Baseline sample                          | Adapter sample                       | restart_s | train_s | serve_s |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | PASS    | " $1000. I deposit $200 into it. What ..."| " aaaf6f8ae738dfc6577e63dda6daf9cc" | 79.1      | 1454.8  | 10.0    |
```

| # | Bug | Symptom | Fix | File |
|---|-----|---------|-----|------|
| 1 | Per-phase container-recreate race | `restart_s=0.1`, train FAILED, phase-2 connection reset | `compose_rm` + `wait_for_phase_ready` | `test/finetune_sweep/run_finetune_sweep.py` |
| 2 | `server_list` not comma-split | `Starting servers: ['api,megatron']` → "No valid server type" | `get_server_list()` (comma split) | `infra/cray_infra/one_server/main.py` |
| 3 | `api` and `megatron` both bind port 8000 | `[Errno 98] address already in use` | phase 1 = `server_list="api"` (not `"api,megatron"`) | `test/finetune_sweep/run_finetune_sweep.py` |
| 4 | 30B training model-load OOM | OOM in `module._apply`→`t.to(device)` before step 0 | on-device `device_map` load | `ml/cray_megatron/models/load_model.py` |

---

## Background

The `cuda-spark` sweep target is **phase-scaled**: because the GB10 has a single
128 GiB pool shared by CPU and GPU, a large model can't co-locate always-on vLLM
*and* training. The runner instead executes two sequential phases per model, each
getting the whole GPU:

- **Phase 1 (train):** bring the stack up with vLLM off, submit the LoRA training
  job, poll to completion, tear down.
- **Phase 2 (serve):** bring the stack up with vLLM on, load the trained adapter,
  run the memorization check (does the served model reproduce the golden string?).

The expert-LoRA converter (see `2026-06-30-moe-expert-lora-serving.md`) had been
validated only on `yujiepan/qwen3-moe-tiny-random`. The goal here was to validate
it at **real scale** on the same grouped-expert `Qwen3Moe` architecture — i.e. the
30B — which required the phased path to actually work at that size.

A prior `cuda-spark-phase-scaling` note had claimed the phasing was "validated" off
the tiny model. **That was a false positive**: at tiny scale the container recreate
is near-instant, so the health gate happened to hit the right container by luck. The
tiny model never actually exercised the phase boundaries under load. Every claim
below is grounded in the 30B run's telemetry instead.

---

## Bug 1 — Per-phase container-recreate race

### Symptom
The first real 30B run failed with `restart_s=0.1`, training `FAILED`, and a phase-2
`baseline generate failed: [Errno 104] Connection reset by peer`.

### Root cause
`teardown_stack()` only `SIGKILL`s the foreground `docker compose up` CLI; it
**deliberately leaves the container (and its GPU) running**. The next phase's
`--force-recreate` is the *only* thing that swaps the container — and on the Spark
that can sit behind a long aarch64/sm12.0 rebuild. Meanwhile the phased readiness
gate used a plain health check (`wait_for_all_up(health_key="megatron")`), which the
**stale leftover container answered instantly**. So the runner raced ahead against
the wrong container: it submitted training against a still-co-located (vLLM-resident)
container and, in phase 2, hit a container that wasn't serving what it expected.

The *co-located* (non-phased) path had already been hardened against exactly this
via `wait_for_model_served` (which additionally checks `/v1/models` serves the
target id). The phased path had silently reintroduced the race.

### Fix
Two helpers in `run_finetune_sweep.py`:

- **`compose_rm(compose_service)`** — `docker compose rm -sf <service>`, run
  synchronously *before each phase restart*, so there is no stale container left to
  answer `/v1/health`. This also frees the previous phase's VRAM before the next
  phase loads.
- **`wait_for_phase_ready(...)`** — an identity-aware gate that requires
  `health[health_key]=="up"` **and** (phase 2) that `/v1/models` actually serves
  `model_id`. It keys on the single active service (`megatron`/`vllm`) rather than
  `all` (which is structurally down mid-phase), and fails fast on the restart proc
  dying or the container crash-looping.

### Evidence it worked
The failure mode *changed*: `restart_s` went from `0.1` (raced) to `66.4`, then
`79.1` — a genuine phase-1 container recreate + slurm registration. The race was
closed, which then exposed Bug 2.

---

## Bug 2 — `server_list` not comma-split

### Symptom
After Bug 1, phase 1 timed out (`restart_s=3001`, `megatron not up (phase 1: timeout)`).
The container logs showed:

```
Overriding config 'server_list' with environment variable 'SCALARLM_SERVER_LIST' - value: api,megatron
Starting servers: ['api,megatron']            <-- one token, not two
ERROR: No valid server type provided. Please specify 'api', 'vllm', 'megatron', or 'all'.
```

### Root cause
`infra/cray_infra/one_server/main.py` wrapped the config string in a single-element
list *without splitting on the comma*:

```python
server_status = await start_cray_server(server_list=[config["server_list"]])
# "api,megatron"  ->  ['api,megatron']  ->  "api" in [...] is False
```

`start_cray_server` does membership checks (`"api" in server_list`), so a comma-joined
token matches nothing. This had never surfaced because production/k8s and the default
use **single tokens** (`all`, `megatron`, `vllm`, `api` — one server per pod). The
phased single-container flow was the first caller to pass a multi-server list.

### Fix
The repo already had the correct parse in one place (`log_handlers.py`:
`[s.strip() for s in server_list.split(",")]`). The fix routes the config value
through `get_server_list()` which does the same split (commit `8630c64`, already in
`main`'s history but not yet on the Spark's working copy). `"api,megatron"` →
`['api', 'megatron']`; single tokens still yield a one-element list, so it is
backward-compatible with k8s/default.

`infra/cray_infra` is bind-mounted into the container, so this deployed with no
image rebuild.

---

## Bug 3 — `api` and `megatron` both bind port 8000

### Symptom
With the list now parsed correctly (`Starting servers: ['api', 'megatron']`), the
container crashed on startup:

```
[Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
  File ".../one_server/create_megatron.py", line 28, in create_megatron
    sys.exit(1)
```

### Root cause
The three servers map to ports:

- `create_api` → **8000**
- `create_vllm` → **8001**
- `create_megatron` → **8000** (a standalone FastAPI app, meant to run *alone* in a
  k8s training pod)

`api` and `megatron` are therefore **mutually exclusive in one container**. The
co-located `"all"` mode works precisely because `start_cray_server` line 44 checks
`"megatron" in server_list` with *no* `"all"` fallback — so under `"all"` the
megatron-server is **skipped**, and the `api` server plus the always-on
`slurmctld`/`slurmd` (started by the container entrypoint regardless of
`server_list`) handle training. Asking for `"api,megatron"` started *two* servers on
8000.

### Fix
Phase 1 uses **`server_list="api"`**, not `"api,megatron"`. Justification:

- Training submission + execution are handled by the `api` server + the entrypoint's
  slurm daemons — exactly as co-located `"all"` does, just without vLLM holding GPU.
- Megatron **health** still reports correctly: `get_megatron_health()`
  (`check_megatron.py`) derives it from slurm via `scontrol show nodes`, served by
  the `api` server — not from the megatron-server on 8000. So the phase-1 gate
  (`health_key="megatron"`) remains valid under `server_list="api"`.

Phase 2 stays `server_list="api,vllm"` (8000 + 8001, no collision).

---

## Bug 4 — 30B training model-load OOM

### Symptom
With phasing fully working, phase 1 came up clean, **but training `FAILED` with
`CUDA error: out of memory`** — with the whole GPU to itself. The traceback:

```
Loading weights: 100%|██████████| 531/531        <-- weights loaded fine (to CPU)
...
  File ".../torch/nn/modules/module.py", in _apply
    param_applied = fn(param)
  File ".../torch/nn/modules/module.py", in convert
    return t.to(...)
torch.AcceleratorError: CUDA error: out of memory
```

The OOM is at **`model.to(device)`**, *after* a successful load — before step 0.

### Root cause
`ml/cray_megatron/models/load_model.py` loaded via
`from_pretrained(torch_dtype="auto")` (weights land on CPU, ~60 GiB bf16) and then
moved the whole model to the GPU with `.to(device)`. On the GB10's **unified** pool,
the CPU-resident copy and the fresh GPU-side copy coexist transiently at ~2× the
weights (~120 GiB) and OOM.

**Confirming it's the load path, not raw size:** phase 2 loads the *same 30B into
vLLM without issue* (it allocates the weights once, ~60 GiB under
`gpu-memory-utilization=0.85`). Only *training* doubled at load. `device_map` and
`low_cpu_mem_usage` were present in `load_model.py` but **commented out**.

### Fix
Load straight onto the target GPU (Big Model Inference) so the weights never sit on
CPU-then-copy:

```python
device = model_info["distribution_strategy"]["device"]
on_gpu = isinstance(device, int) or (
    isinstance(device, torch.device) and device.type == "cuda"
)
load_kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True}
if on_gpu:
    load_kwargs["device_map"] = {"": device}
# ...from_pretrained(..., **load_kwargs) for both the primary and eager-fallback calls
```

- The now-redundant final `.to(device)` is **skipped when `on_gpu`** — accelerate
  disallows moving a dispatched model, and it would reintroduce the very peak this
  fix avoids.
- Guarded to real GPU devices; the `cpu` target keeps the plain CPU load.
- `accelerate 1.14.0` is already in the image; `ml/` is bind-mounted, so no rebuild.

### Evidence it worked
Training status advanced to `TRAINING` (past the load), `train_s=1454.8`, and the run
PASSed. No accelerate-hook conflict with the custom megatron/tokenformer path.

> **Caveat:** this is an ML-side change on the load path for **all** models. It needs
> a small-model + CPU-target regression check before landing in a PR.

---

## Intermediate result — smaller MoE surfaced a converter arch gap

Before fixing Bug 4, we tried validating the converter with a *smaller* real MoE that
fits without the load-doubling problem: **`Qwen/Qwen1.5-MoE-A2.7B-Chat`**
(`Qwen2MoeForCausalLM`, ~28 GiB). Result: **`NO_MEMORIZATION`, but informative**:

- Training converged (loss ≈ 0.0033).
- The adapter was **demonstrably applied** — base output was coherent English, adapter
  output was hex-like characters (the golden string is hex). Not a no-op.
- But the served string was **scrambled** (`dfdf...8811...cccc999...`), not the exact
  `aaaf6f8...`.

Because tiny `Qwen3Moe` reproduces the string exactly, the corruption is
`Qwen2Moe`-specific — almost certainly its **shared expert** (an always-on expert
alongside the routed ones) being mis-mapped by the converter, which currently handles
only routed grouped experts. So: **converter correct for `Qwen3Moe`; open gap for
shared-expert arches** (`Qwen2Moe`/`Qwen1.5-MoE`, `deepseek-moe`). This is why the
30B (same arch as the validated tiny model, no shared expert) is the clean,
confound-free validation.

---

## Final validation

After all four fixes, `Qwen3-30B-A3B-Instruct-2507` PASSed end-to-end on `cuda-spark`:
the served adapter reproduced the golden string **exactly**
(` aaaf6f8ae738dfc6577e63dda6daf9cc`), `train_s=1454.8`, `serve_s=10.0`,
`restart_s=79.1`. This is the first real large-MoE PASS on the phased path and the
exact-architecture validation of the expert-LoRA converter at 30B scale.

---

## Follow-ups

- **PR extraction** (per the lab-branch workflow — `georgi/finetune-sweep` is never
  PR'd whole): the two production-code fixes should become short-lived PRs off fresh
  `main`:
  - Bug 2 (`get_server_list` in `main.py`) — already exists as commit `8630c64`;
    ensure it's on `main`.
  - Bug 4 (`device_map` load in `load_model.py`) — needs the small-model + CPU-target
    regression check called out above.
  - Bugs 1 and 3 live in the sweep tooling (`run_finetune_sweep.py`), which stays on
    the lab branch.
- **Shared-expert converter gap** — extend the converter to map `Qwen2Moe`'s shared
  expert so `Qwen1.5-MoE` (and by extension `deepseek-moe`) memorize exactly. Deeper,
  arch-specific converter work; deferred.
- **Operational note** — SSH commands that touch Docker/GPU on the Spark intermittently
  hang ~2 min (known `nvidia-smi`/docker-group quirk on `spark-147c`); launches were
  done detached via `setsid` and verified on reconnect.

## Files touched (on `georgi/finetune-sweep`, deployed to the Spark)

- `test/finetune_sweep/run_finetune_sweep.py` — `compose_rm`, `wait_for_phase_ready`,
  phase-1 `server_list="api"`, phase gating.
- `infra/cray_infra/one_server/main.py` — `get_server_list()` comma split.
- `ml/cray_megatron/models/load_model.py` — on-device `device_map` load + guarded
  final `.to(device)`.
- `test/finetune_sweep/finetune-sweep.yaml` — added `Qwen1.5-MoE-A2.7B-Chat` (arch-gap
  probe) alongside the staged 30–35B MoEs.
