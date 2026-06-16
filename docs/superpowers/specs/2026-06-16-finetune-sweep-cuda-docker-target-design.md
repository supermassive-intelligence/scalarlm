# Fine-tune sweep: `cuda-docker` GPU-Compose target design

_Design spec for a **testing-only** GPU target that runs the fine-tune sweep's
closed loop on a single-GPU box (the 3090) via **Docker Compose**, bypassing
k8s. It exists because the k8s cluster is not available for testing until more
compute lands. It is a delta on the k8s sweep
(`docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md`) and its
phase-scaled variant; the dataset, outcome enum, memorization criterion, and the
train→hot-load→serve loop are unchanged — only the **lifecycle** (Compose instead
of Helm/k8s) and the **target dispatch** change._

## Motivation

The recent sweep work moved the GPU `cuda` target onto k8s (per-model Helm
namespace, now 1-GPU phase-scaled). The k8s cluster (`blackwell-maxq-0`) is the
canonical GPU path, but it is unavailable for testing until more compute is
provisioned. The team still needs to exercise the GPU closed loop now, on a
single-GPU Docker-Compose box (the 3090).

The key observation: **the runner's Compose lifecycle is already GPU-capable.**
`cuda` *was* a Compose target before commits `a8beda4`/`35f93cb` moved it to k8s.
`run_model`'s `else` branch still drives the full Compose loop — `start_restart`
→ `./scalarlm up`, baseline → train → checkpoint read via `docker compose exec`,
and `teardown_stack`'s `nvidia-smi` VRAM-settle loop. So a GPU-Compose target is
mostly **manifest config plus fixing the runner's name-based dispatch**, not new
lifecycle code.

## Scope

- **In scope**: rename the existing `cuda` target → `cuda-k8s` (pure key rename);
  add a `cuda-docker` target that reuses the existing Compose lifecycle on the
  `cray-nvidia` GPU service; refactor the runner's three literal-name dispatch
  points to be driven by target config.
- **Unchanged**: the Compose lifecycle code, the k8s and phase-scaled paths, the
  outcome enum, the memorization criterion, the dataset, and the shared HTTP /
  checkpoint helpers.
- **Out of scope**: any chart change, the phase-scaled k8s loop, and making
  `cuda-docker` a production path (it is explicitly testing-only).

## Relationship to the single-GPU-arbiter directive (outside its scope)

`cuda-docker` is **not** a deviation from the supervisor's GPU directive — it
sits **outside the directive's scope**. ADR 0003 (2026-06-15 amendment) records
the directive precisely: "use the cluster instead of running scripts directly" is
concretely **"no GPU work outside the scheduler"** — and "it does not forbid
running helm/kubectl from the host." Its *reason* is specific to a
**scheduler-managed box** like `blackwell-maxq-0`: a Compose vLLM container there
occupies a GPU outside k8s's knowledge, so k8s can double-place a pod onto a busy
card → CUDA OOM / contention.

`cuda-docker` targets a box that runs **no Helm/k8s scheduler** (the 3090). There
is no scheduler there to subvert and no pods to double-place, so it cannot
reintroduce the contention the directive guards against. It is a **testing
affordance for scheduler-less GPU boxes**; `cuda-k8s` remains the canonical path
for the scheduler-managed cluster.

**The one real guardrail this implies:** `cuda-docker` must **never** be run on a
k8s node (`blackwell-maxq-0`). *There* a Compose GPU launch would be exactly the
behind-the-scheduler's-back contention the directive forbids. On a scheduler-less
box it is fine. The manifest comment and the ADR amendment record this scope and
guardrail.

## Design

### Manifest (`test/finetune_sweep/finetune-sweep.yaml`)

Rename `cuda` → `cuda-k8s` (all its config carries over verbatim). Add a sibling
`cuda-docker` reusing the Compose lifecycle:

```yaml
targets:
  cpu:                         # unchanged
    compose_service: cray
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cpu"
    train_args_overrides: { gpus: 0 }

  cuda-docker:                 # testing-only: single-GPU box (3090) via Docker Compose.
    compose_service: cray-nvidia
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up nvidia"
    train_args_overrides: { gpus: 1 }
    # Testing target for a GPU box that runs NO Helm/k8s scheduler (the 3090).
    # Outside the "no GPU work outside the scheduler" directive's scope (no
    # scheduler here to subvert). GUARDRAIL: never run on a k8s node
    # (blackwell-maxq-0) -- there it WOULD contend with the scheduler. cuda-k8s
    # stays the canonical path for the cluster. See the 2026-06-16 amendment in
    # docs/adr/0003-finetune-sweep-restart-per-model.md.

  cuda-k8s:                    # canonical GPU path (was `cuda`); k8s, phase-scaled.
    chart_path: deployment/helm/scalarlm
    # … all current `cuda` config unchanged …
```

`./scalarlm up nvidia` brings up the `cray-nvidia` service (`runtime: nvidia`,
GPU `deploy` reservation); `compose_service: cray-nvidia` is the `docker compose
exec` target for the in-container checkpoint-keys read. Models stay shared;
`Qwen/Qwen2.5-0.5B` already declares `adapters.lora.gate_gb: 8`, which the VRAM
gate applies to `cuda-docker` too (a 3090's ~24 GB clears it).

### Runner: config-driven dispatch (the real work)

Three spots in `run_finetune_sweep.py` currently key on the literal target name
`"cpu"`/`"cuda"`; a third GPU target breaks all three. Replace name-matching with
config-driven dispatch, mirroring the existing config-driven `is_k8s_target`
(`"chart_path" in cfg`):

- **New helper** `target_requests_gpu(target_cfg) -> bool`:
  `target_cfg.get("train_args_overrides", {}).get("gpus", 1) >= 1`. JobConfig
  defaults `gpus=1`; only `cpu` overrides it to `0`. So this is True for
  `cuda-docker` and `cuda-k8s`, False for `cpu`. Pure + unit-testable.
- **VRAM probe** (`run_model`, currently `probe_gpu_free_gb() if target ==
  "cuda" else []`): change to `probe_gpu_free_gb() if target_requests_gpu(cfg)
  else []`. `cuda-docker` now gets a real `nvidia-smi` probe; `cpu` gets `[]`;
  k8s still short-circuits to `free_gb=None` via `is_k8s_target`.
- **`gate_model`** — change signature from `gate_model(model, target, free_gb)`
  to `gate_model(model, target_cfg, free_gb)` and branch on
  `target_requests_gpu(target_cfg)` instead of `target == "cpu"`: GPU targets use
  the VRAM/scheduler gate (k8s passes `free_gb=None` to skip the runtime check),
  the CPU target uses the `cpu_ok` opt-in gate. The gate logic is otherwise
  identical; messages drop the bare target string.
- **`--target`** argparse: drop the hardcoded `choices=["cpu","cuda"]`. The
  existing post-load check (`if args.target not in manifest["targets"]:
  ap.error(...)`) already validates against the manifest's keys, so accepted
  targets auto-track `cpu` / `cuda-docker` / `cuda-k8s`.
- **`--restart-timeout` default 600 → 3000.** `./scalarlm up nvidia` runs
  `docker compose up --build`, and the Dockerfile compiles **vLLM from source**
  for the GPU arch (`pip install --no-build-isolation -e .`) on `nvcr.io/nvidia/
  pytorch`. A cold build is tens of minutes, so the 600s health-wait would record
  the first model as `RESTART_FAILED` for a reason unrelated to the sweep
  (subsequent `--force-recreate` builds are cache hits and fast). 3000s (50 min)
  covers the cold build unattended. Raising the *default* (not a per-target
  override) is safe: it is only a ceiling — `wait_for_all_up` / `helm_install`
  return as soon as healthy, so `cpu` and `cuda-k8s` happy paths are unaffected;
  only their failure-detection latency grows, an acceptable trade for not failing
  the cold GPU build.

### How one GPU serves both train and serve (no phase handoff)

`cuda-k8s` needs the phase-scaled design — two `kubectl scale`s and a GPU
handoff — because vLLM and megatron are **separate pods**, each reserving a card,
so a single GPU cannot host both at once. The Compose path has no such split. The
`cray-nvidia` service runs `one_server.main`, and `start_cray_server` with the
default `server_list="all"` brings up the **API + vLLM in-process** while training
is dispatched as a **slurm job** (slurmd runs in the same container via
`start_slurm.sh`). So one container hosts the server and the trainer, and they
**share the single GPU simultaneously** — this is a *co-located run*, the
co-location strategy the phase-scaled k8s spec rejected as "chart surgery" but
which Compose provides for free. Consequently `cuda-docker` needs **none** of the
phase-scaling machinery: no `replicaCounts` overrides, no scale calls, no GPU
handoff, no per-phase health gating. It is the plain always-on Compose loop on a
GPU, gated only by the model fitting the one card alongside vLLM (Qwen2.5-0.5B
fp32 on a 3090's ~24 GB is comfortable).

### Teardown note (GPU not fully reclaimed by `teardown_stack`)

`teardown_stack` SIGKILLs the `./scalarlm up nvidia` process group, but
`./scalarlm up` runs `docker compose up cray-nvidia` in the **foreground
(attached, not `-d`)**. The container is owned by the docker daemon, so SIGKILL
kills the compose CLI and leaves the `cray-nvidia` container running, **still
holding the GPU**. `teardown_stack`'s settle loop only watches for VRAM to stop
*climbing*, so a leaked-but-idle container reads as "settled" though the card was
never freed. Consequences for `cuda-docker`:

- **Between models** the next `./scalarlm up nvidia` self-heals, because it runs
  with `--force-recreate` (stops + recreates the container, freeing then
  re-claiming the card). So a multi-model sweep is correct.
- **After the last/only model** (the manifest currently has exactly one) the
  `cray-nvidia` container survives the run and keeps the GPU bound until a manual
  `docker compose -f docker-compose.yaml down cray-nvidia`.

This is accepted for a testing-only target rather than changing the shared
`teardown_stack` (which `cpu` also uses). The integration-test steps document the
manual `down`. We do **not** claim clean VRAM reclamation for this path.

### What stays identical

The entire Compose lifecycle in `run_model`'s `else`/`finally` branches
(`start_restart` → `./scalarlm up nvidia`, baseline → `submit_train` →
`poll_training` → `read_checkpoint_keys` via `docker compose exec cray-nvidia` →
`serve_check_and_classify`, teardown via `teardown_stack`'s VRAM-settle loop) is
already GPU-capable and runs unchanged for `cuda-docker`. The k8s
(`run_model` k8s branch) and phase-scaled (`run_model_k8s_phased`) paths are not
touched. The pure helpers, HTTP train/generate helpers, and outcome
classification are unchanged.

## Where the runner runs

Unchanged from the existing model: the runner is a **local script that runs on
the host of the box under test** (it does host-level `./scalarlm up`, `docker
compose exec`, and `nvidia-smi`). For `cuda-docker` that host is the 3090 box —
pull this branch there and run `python3 run_finetune_sweep.py --target
cuda-docker`. It does not run from this workstation (no GPU, no `cray-nvidia`
stack here).

## Ripple

- **Tests** (`test/unit/test_finetune_sweep_k8s.py`, the only remaining unit
  file): update the three `gate_model(model, "cuda", …)` calls to the new
  `target_cfg` signature; add tests for `target_requests_gpu` (truth table over
  `gpus` 0/1/absent) and for `cuda-docker` selecting the Compose lifecycle (not
  k8s) and triggering the VRAM probe. `Result(target="cuda")` labels are free
  strings — leave as-is or relabel cosmetically.
- **Stale docstring**: `teardown_stack`'s docstring says it "mirrors
  teardown_engine in test/model_sweep/run_sweep.py", but that file no longer
  exists (only `__pycache__` remains). Drop or correct the reference while we're
  in this code — a one-line cleanup, not a behavior change.
- **Docs (forward-record only)**: do **not** back-edit historical mentions of the
  `cuda` target — ADR 0003's existing amendments and the k8s/phase-scaled specs
  correctly describe the state at *their* date (when the target was named
  `cuda`). Record the `cuda` → `cuda-k8s` rename **only** in the new 2026-06-16
  amendment. Update **only live surfaces** a user runs against today: the runner
  module docstring's `--target cuda` usage example, and the manifest. This matches
  standard ADR practice (amend forward, never rewrite history) and keeps the blast
  radius tiny.
- **ADR**: amend `docs/adr/0003-finetune-sweep-restart-per-model.md` (2026-06-16)
  recording the testing-only Compose-GPU target, the `cuda` → `cuda-k8s` rename,
  and the directive **scope boundary + box-confinement guardrail** (`cuda-docker`
  is outside the "no GPU work outside the scheduler" directive; never run it on a
  k8s node).

## Consequences

- **Unblocks GPU testing now** on a single Compose box without the k8s cluster.
- **Config-driven dispatch** removes three brittle literal-name branches, so
  future targets are manifest-only — a net simplification the third target pays
  for.
- **Sits outside the single-GPU-arbiter directive** (a scheduler-less GPU box has
  no scheduler to subvert), not in tension with it — provided the box-confinement
  guardrail holds: never run `cuda-docker` on the k8s node. `cuda-k8s` stays
  canonical for the cluster.
- **Two GPU targets to keep in sync** if the closed loop changes — but they share
  every helper below the lifecycle branch, so the surface is small.

## Testing

- **Unit (here, via uv):** `target_requests_gpu` truth table; `gate_model`
  cpu/gpu/k8s branches under the new signature; `cuda-docker` selecting the
  Compose lifecycle and triggering the VRAM probe. Run with
  `PYTHONPATH=infra uv run --with pytest --with torch python -m pytest`.
- **Integration (on the 3090):** `python3 run_finetune_sweep.py --target
  cuda-docker` — full train → hot-load → serve closed loop on the GPU. The first
  run compiles the vLLM image (covered by the 3000s `--restart-timeout`);
  subsequent runs are cache-hit fast. Confirm `nvidia-smi` shows the
  `cray-nvidia` container using the card and the result reaches
  PASS/NO_MEMORIZATION. Afterwards, free the card with `docker compose -f
  docker-compose.yaml down cray-nvidia` (see the teardown note — the container
  outlives the runner).

## ADR impact

Recorded as the **2026-06-16 amendment to ADR 0003**
(`docs/adr/0003-finetune-sweep-restart-per-model.md`): "the sweep exposes a
testing-only `cuda-docker` target that runs the GPU closed loop via Docker
Compose on a **scheduler-less** GPU box (the 3090), and renames the prior `cuda`
target to `cuda-k8s`." The amendment's job is to record the **scope and
guardrail** a future reader will otherwise misread: `cuda-docker` is *outside* the
2026-06-15 amendment's "no GPU work outside the scheduler" directive (no scheduler
on that box), **not** a deviation from it — and it must never be run on a k8s node
(`blackwell-maxq-0`), where it would become exactly the behind-the-scheduler
contention that amendment forbids. It is ADR-worthy because that scope boundary is
surprising without context (the prior amendment moved GPU work *onto* k8s), and it
continues the per-model → k8s → 1-GPU → scheduler-less-testing narrative.
