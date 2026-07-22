# Fine-tune sweep on a single GPU (phase-scaled) design

_Design spec for a **1-GPU** variant of the k8s fine-tune sweep. It reshapes the
closed loop so training and serving occupy the GPU in **sequential phases**
rather than as two always-on pods, letting a model run on a cluster (or node)
that has only **one** schedulable GPU. This is a delta on
`docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md`; the goal,
outcome enum, memorization criterion, and namespace-per-model lifecycle are
unchanged — only the in-namespace orchestration of the closed loop changes._

## Motivation

The k8s sweep needs **two schedulable GPUs per model**: vLLM (Deployment) and
megatron (StatefulSet) each request `nvidia.com/gpu: 1` and hold it for the
namespace's lifetime. On `blackwell-maxq-0` (confirmed 2026-06-15) this is the
practical wall — `nvidia.com/gpu` is exclusive and non-overcommittable,
`sharing-strategy=none`, and idle tenants reserve cards by request, so the sweep
sits `Pending` even when `nvidia-smi` shows idle silicon (see the k8s spec's "GPU
model and operational preconditions"). Of the three ways to get to one GPU, only
**phase-scaling is fully in the sweep's control** (time-slicing needs the
GPU-operator owner; co-location is chart surgery). This spec designs it.

The key observation: **a sweep never needs training and serving *simultaneously*.**
It trains a LoRA, then hot-loads the resulting checkpoint into vLLM. Those two
phases each need one GPU, so they can run on the *same* GPU one at a time.

## Scope

- **In scope**: an opt-in `phase_scaled: true` mode on the k8s `cuda` target that
  (a) installs with vLLM scaled to 0 and megatron to 1, (b) trains, (c)
  `kubectl scale`s megatron→0 and vLLM→1, (d) runs the baseline + hot-load +
  serve check, (e) tears the namespace down. Peak GPU demand = 1.
- **Unchanged**: namespace-per-model (`helm upgrade --install` →
  `kubectl delete namespace`), the outcome enum, the memorization criterion, the
  `sweep_run_id` nonce, the checkpoint-key check, the HTTP helpers, and the
  block-and-wait/`gpu_wait_timeout` scheduling discipline (applied per phase).
- **Out of scope**: the 2-GPU default path (stays as the k8s spec defines it),
  the `cpu`/Compose path, and any chart change (the chart already exposes
  `replicaCounts.{inference,training}`).

## Why phase-scaling, and why the chart needs no change

`replicaCounts.inference` drives the vLLM Deployment's `replicas`;
`replicaCounts.training` drives the megatron StatefulSet's `replicas`; the api
pod (slurmctld + HTTP) requests **no GPU** and stays up across both phases. So
the GPU can be handed from megatron to vLLM purely with `--set` at install and
two `kubectl scale` calls — no template change. Training reads the model from the
HF cache and does not need vLLM up; serving reads the trained checkpoint from the
shared `jobs` PVC and does not need megatron up. The two phases are genuinely
separable.

## Design

### Phase-scaled lifecycle (per model, peak GPU = 1)

```
Phase 0 — install; megatron holds the single GPU
  kubectl delete namespace sweep-<model> --ignore-not-found --wait   # pre-clean
  helm upgrade --install scalarlm <chart> -n sweep-<model> --create-namespace \
    --set model=<model> \
    --set storage.cache.kind=hostPath --set storage.cache.hostPath=<node-cache> \
    --set replicaCounts.inference=0 \    # vLLM off -> 0 GPU
    --set replicaCounts.training=1       # megatron on -> 1 GPU
  block-and-wait: megatron pod Ready, then health["megatron"]=="up"  # slurm-registered, GPU-aware

Phase 1 — train (megatron uses the GPU; vLLM is absent)
  POST /v1/megatron/train -> runs on megatron -> checkpoint to jobs PVC
  poll training status until COMPLETED/FAILED/CANCELLED/timeout
  kubectl exec statefulset/scalarlm-megatron -- <checkpoint-keys script>

Phase 2 — hand the GPU to vLLM
  kubectl scale statefulset/scalarlm-megatron --replicas=0   # free the GPU
  (wait megatron pod fully Terminated -> GPU request released)
  kubectl scale deployment/scalarlm-vllm     --replicas=1   # claim the GPU
  block-and-wait: vLLM pod Ready, then health["vllm"]=="up"  # NOT health.all (megatron is down now)
  baseline generate (base model)            # control, moved here from pre-train
  hot-load adapter: generate vs job_hash -> serve check + memorization classify

Teardown — kubectl delete namespace sweep-<model>   (kills port-forward first)
```

### Manifest

Opt-in per target, so the 2-GPU path stays the default:

```yaml
targets:
  cuda:
    chart_path: deployment/helm/scalarlm
    release: scalarlm
    namespace_prefix: sweep
    megatron_sts: scalarlm-megatron
    vllm_deploy: scalarlm-vllm     # NEW: kubectl scale target for phase 2
    api_service: scalarlm
    cache_hostpath: /root/.cache
    gpu_wait_timeout: 7200
    phase_scaled: true             # NEW: opt into the 1-GPU phase-scaled loop
    train_args_overrides:
      gpus: 1
```

`run_model` branches on `target_cfg.get("phase_scaled")` (only meaningful for
k8s targets).

### Runner changes — this is the real work

The 2-GPU path's `run_model` is "bring the whole stack up, then
baseline→train→serve in one shot." The phase-scaled path **re-sequences the
closed loop**, so it is a distinct orchestration shape, not a helper swap:

- **`helm_install` gains scale overrides.** Either extend `k8s_helm_install_cmd`
  to append `--set replicaCounts.inference=0 --set replicaCounts.training=1` when
  `phase_scaled`, or pass extra `--set`s through.
- **New helper `kubectl_scale(kind_name, replicas, namespace)`** building
  `kubectl scale <kind/name> --replicas=<n> -n <ns>` (pure builder + thin
  wrapper, unit-testable like the others).
- **New wait points (health gate is per-phase — never `all`).** Reuse
  `wait_for_pods_ready` after the install (megatron) and after the vLLM scale-up.
  The API health gate **cannot use `health.all == "up"` in either phase**: the api
  aggregates `all` over `[api, vllm, megatron]`, and in a phase-scaled run exactly
  one GPU service is up at a time, so `all` is structurally `down` in **both**
  phases (vLLM down in phase 1; megatron down in phase 2). Instead, generalize the
  existing helper to `wait_for_all_up(api_url, proc, timeout, health_key="all")`
  and gate on a **single component key per phase**: phase 1 polls
  `health["megatron"] == "up"` **before `submit_train`** (this is `get_megatron_health`
  — slurm-node registration, a stronger signal than the megatron pod's TCP
  startupProbe); phase 2 polls `health["vllm"] == "up"` after the vLLM scale-up,
  before the baseline/hot-load generates. `"all"` is never the gate on this path.
- **Scale-down must release the GPU before scale-up.** After
  `kubectl scale megatron --replicas=0`, wait until the megatron pod is **gone**
  (its `nvidia.com/gpu` request is released only on full pod deletion, not while
  `Terminating`) before scaling vLLM up — otherwise vLLM sits `Pending` waiting for
  a GPU megatron still reserves. Implement as a polling barrier
  `wait_for_pods_gone(namespace, selector, timeout)` that reuses `kubectl_get_pods`
  (selector `app.kubernetes.io/component=megatron`) and returns when no matching
  pod remains, bounded by `gpu_wait_timeout` → `RESTART_FAILED` on expiry. Polling
  is preferred over `kubectl wait --for=delete` (which errors when zero pods already
  match) and mirrors the existing block-and-wait helpers. The default 30 s
  termination grace means this normally clears in well under a minute.
- **Baseline moves into phase 2.** The baseline generate (a control on the *base*
  model) currently runs before training; in the phase model vLLM is down then, so
  it runs right before the adapter hot-load in phase 2. Still a valid control.
- **Checkpoint read stays in phase 1** (megatron still up), via the existing
  `read_checkpoint_keys_k8s`.

### What stays identical

The pure logic (`classify_pod_status`, `classify_result`,
`checkpoint_lora_keys_ok`, `k8s_namespace`, the command builders), the HTTP
train/generate helpers, teardown via `kubectl delete namespace`, and the
block-and-wait semantics. The 2-GPU path is untouched and remains the default.

## Consequences

- **Runs on one schedulable GPU** — the whole point; the sweep fits a busy
  cluster or a single-GPU node.
- **Two GPU warmups per model, serially.** Phase 0 pays megatron's load; phase 2
  pays vLLM's capture-graph warmup (the chart budgets up to 30 min). Per-model
  wall time grows; for a long model list this is the main cost.
- **More orchestration surface**: two `kubectl scale`s, a delete-wait between
  them, and a re-sequenced loop — more places to fail, each mapped to
  `RESTART_FAILED`/`TRAIN_FAILED` with a phase-specific detail.
- **GPU is released between phases**, so on a shared cluster another tenant could
  grab it in the gap between megatron scale-down and vLLM scale-up. On a busy
  cluster the vLLM scale-up may then block (`gpu_wait_timeout`) — acceptable and
  correctly handled, but means "1 GPU" is "1 GPU at a time," not "1 GPU reserved
  throughout."

## Resolved (grilling 2026-06-16)

All three original open questions are resolved; two were settled directly from the
code. See `docs/adr/0003-finetune-sweep-restart-per-model.md` (2026-06-16 amendment).

- **Re-acquire race vs. reservation hold → accept the gap.** Between phases the
  GPU is briefly free and a co-tenant could take it. We accept this: holding the
  reservation across the handoff would require both pods' GPU requests alive at
  once, defeating the 1-GPU goal. The phase-2 `wait_for_pods_ready` block-and-wait
  already covers re-acquisition — vLLM sits `Pending` until the GPU frees, or hits
  `gpu_wait_timeout` → `RESTART_FAILED`. **"1 GPU" means "1 GPU at a time," not
  "1 GPU reserved throughout."**
- **Hot-load does NOT need megatron at serve time → confirmed in code.**
  `find_model` (`infra/cray_infra/training/vllm_model_manager.py`) resolves a
  `job_hash` purely by globbing `*.pt` under `training_job_directory`
  (`/app/cray/jobs`); it never consults slurm/megatron. The `jobs` PVC is
  **`ReadWriteMany`** (longhorn) and mounted by api + vllm + megatron, so a
  checkpoint megatron wrote in phase 1 is fully visible to vLLM in phase 2 after
  megatron is gone. Scaling megatron→0 is safe for the hot-load. (`kubectl delete
  namespace` still GCs the PVC at teardown — `resource-policy: keep` only blocks a
  *helm* uninstall, not namespace deletion.)
- **Single target with a `phase_scaled` flag → chosen.** The 1-GPU and 2-GPU paths
  share every k8s identifier and the whole pure-helper layer; only behavior
  diverges, so behavior lives in `run_model` (branch on
  `target_cfg.get("phase_scaled")`), not in duplicated YAML. A separate target
  would only earn its keep with divergent config (different chart/release/cache),
  which these don't have. Install-time scale overrides are appended inside
  `k8s_helm_install_cmd` when `phase_scaled` (`--set replicaCounts.inference=0
  --set replicaCounts.training=1`), leaving the 2-GPU `--set` list untouched.
  **Use `replicaCounts.inference=0`, not `vllm.enabled=false`** — the latter drops
  the Deployment entirely, leaving nothing for phase 2's `kubectl scale` to act on.

## Testing

- **Unit (here, via uv):** the new `kubectl_scale` builder + a phase-dispatch
  predicate, same TDD pattern as the k8s helpers. The re-sequenced `run_model`
  phase path is validated structurally (imports, the phase ordering) since it
  orchestrates side effects.
- **Integration (on the box):** a 1-GPU node (or a node with exactly one free
  reservation) is the actual test — confirm peak GPU = 1 via `nvidia-smi` and
  `kubectl get pods` across the three phases.

## ADR impact

Recorded as the **2026-06-16 amendment to ADR 0003**
(`docs/adr/0003-finetune-sweep-restart-per-model.md`): "the sweep may run the
closed loop in sequential single-GPU phases (train, then serve) instead of two
always-on GPU pods." It is hard to reverse (re-sequences the loop), surprising
(baseline after training; health gate never `all`; GPU handed off mid-loop), and
a real trade-off (1 GPU vs. ~2× warmup wall time) — the three ADR criteria. The
amendment continues the per-model → k8s → 1-GPU narrative already in that ADR.
