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

## Directive tension (must be recorded)

A standing supervisor directive says GPU sweeps must go through the **k8s
single-GPU arbiter**, and the sweep's Compose stack-launch was to become a Helm
rollout. A `cuda-docker` target reintroduces exactly that Compose stack-launch.
This is accepted **only as a testing affordance while k8s compute is
unavailable**: `cuda-k8s` remains the canonical GPU path. The manifest comment
and an amendment to ADR 0003 record this so the deviation is intentional and
visible, not silent drift.

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
    # Reintroduces a Compose GPU stack-launch, which the k8s single-GPU-arbiter
    # directive moved away from. Use only while k8s compute is unavailable;
    # cuda-k8s stays the canonical GPU path. See the 2026-06-16 amendment in
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
- **Docs**: the k8s and phase-scaled specs and the runner module docstring
  reference `--target cuda`; add a note that the k8s target is now `cuda-k8s` and
  a `cuda-docker` sibling exists. (The specs are historical design records; a
  light note suffices, no rewrite.)
- **ADR**: amend `docs/adr/0003-finetune-sweep-restart-per-model.md` (2026-06-16)
  recording the testing-only Compose-GPU target, the `cuda` → `cuda-k8s` rename,
  and the single-GPU-arbiter directive tension.

## Consequences

- **Unblocks GPU testing now** on a single Compose box without the k8s cluster.
- **Config-driven dispatch** removes three brittle literal-name branches, so
  future targets are manifest-only — a net simplification the third target pays
  for.
- **Deviates from the single-GPU-arbiter directive** for the Compose-GPU path.
  Mitigated by the testing-only framing, the manifest comment, and the ADR
  amendment; `cuda-k8s` stays canonical.
- **Two GPU targets to keep in sync** if the closed loop changes — but they share
  every helper below the lifecycle branch, so the surface is small.

## Testing

- **Unit (here, via uv):** `target_requests_gpu` truth table; `gate_model`
  cpu/gpu/k8s branches under the new signature; `cuda-docker` selecting the
  Compose lifecycle and triggering the VRAM probe. Run with
  `PYTHONPATH=infra uv run --with pytest --with torch python -m pytest`.
- **Integration (on the 3090):** `python3 run_finetune_sweep.py --target
  cuda-docker` — full train → hot-load → serve closed loop on the GPU; confirm
  `nvidia-smi` shows the `cray-nvidia` container using the card and the result
  reaches PASS/NO_MEMORIZATION.

## ADR impact

Recorded as the **2026-06-16 amendment to ADR 0003**
(`docs/adr/0003-finetune-sweep-restart-per-model.md`): "the sweep exposes a
testing-only `cuda-docker` target that runs the GPU closed loop via Docker
Compose on a single-GPU box, bypassing k8s, while compute for the canonical
`cuda-k8s` path is unavailable." It is a deliberate, recorded deviation from the
single-GPU-arbiter directive (hence ADR-worthy), continues the per-model → k8s →
1-GPU narrative already in that ADR, and renames the prior `cuda` target to
`cuda-k8s`.
