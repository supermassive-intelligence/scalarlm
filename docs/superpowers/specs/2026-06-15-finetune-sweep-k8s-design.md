# Fine-tune sweep on Kubernetes design

_Design spec for porting the tier-(d) fine-tune sweep
(`test/finetune_sweep/run_finetune_sweep.py`) from a Docker-Compose stack launch
to the team's existing **Helm / namespace-per-model** workflow on the k8s box, so
that **all GPU allocation goes through the k8s scheduler**. This is a delta on
`docs/superpowers/specs/2026-06-10-finetune-sweep-design.md` and amends ADR 0003
— the sweep's goal, outcome enum, pass criteria, and memorization logic are
unchanged; only **how the stack is launched, reached, and torn down** changes._

## Motivation

The sweep currently restarts the stack with `SCALARLM_MODEL=X ./scalarlm up cuda`
= `docker compose up --force-recreate`, which starts a vLLM container that
occupies a GPU **outside the k8s scheduler's knowledge**. On `blackwell-maxq-0`
the scheduler must be the single authority over GPU allocation; a GPU claimed via
Compose behind its back lets k8s double-place a pod onto a busy device → CUDA OOM
/ contention. The supervisor's directive "use the cluster instead of running
scripts directly" means, concretely, **"no GPU work outside the scheduler"** — it
does *not* forbid running orchestration (`helm`/`kubectl`) from the host.

**The team already has the pattern this sweep should mirror.** From the box's
shell history, deployments are **one Helm release per model, in its own
namespace**:

```
helm upgrade --install gemma4 scalarlm -f values/gemma4.yaml -n gemma4 --create-namespace
helm -n gemma4 uninstall gemma4
```

run by the `normal@blackwell-maxq-0` host user. Namespaces are model-named
(`qwen4b`, `qwen3-5-4b`, `gemma4`, …) and coexist. The sweep should do the same:
spin a fresh per-model release into a dedicated namespace, run the closed loop,
delete it. This **dissolves** the in-place model-switch problem entirely — a
brand-new pod reads the model from its ConfigMap at startup, so there is no
need to mutate a running release, no ConfigMap-reload trick, and no chart change.

## Scope

- **In scope**: stack launch (per-model `helm upgrade --install` into a
  `sweep-<model>` namespace), GPU-aware readiness (block-and-wait), API
  reachability (per-model `kubectl port-forward`), the in-container checkpoint
  read (`docker compose exec` → `kubectl exec`), teardown (`kubectl delete
  namespace`), and the now-removable host VRAM probe. The `cuda` target only.
- **Out of scope / unchanged**: the goal, the train → hot-load → serve closed
  loop, the outcome enum (`PASS`, `NO_MEMORIZATION`, `ADAPTER_NOT_LOADED`,
  `BAD_CHECKPOINT`, `TRAIN_FAILED`, `TRAIN_TIMEOUT`, `RESTART_FAILED`,
  `SKIPPED`), the dataset/golden-string memorization criterion, the dedup-defeat
  `sweep_run_id` nonce, all HTTP train/generate helpers (they keep talking to
  `localhost:8000`), and all pure logic (gating, checkpoint-key check, outcome
  classification). The `cpu` target (no GPU) may stay on Compose; not addressed
  here.
- **No chart changes required.** The sweep uses `deployment/helm/scalarlm`
  as-is. (Earlier drafts proposed a `checksum/config` annotation and a
  `Recreate` strategy; namespace-per-model makes both unnecessary — see
  "What explicitly does NOT change".)
- **Runner location**: the runner stays an **ordinary host/SSH process**
  ("Approach A") — it is GPU-less (only `helm`/`kubectl` + HTTP + a checkpoint
  read), and host-run `helm` is exactly what the team does, so it does not
  violate the scheduler constraint. An in-cluster RBAC Job ("Approach B") is
  **not** required; the host user's kubeconfig already has install/uninstall and
  namespace-create rights (the shell history exercises them).

## Constraint being satisfied

Every GPU-occupying process is a k8s-scheduled pod requesting `nvidia.com/gpu`
(the chart's vLLM Deployment and megatron StatefulSet each request one). The
runner never launches a GPU process directly and never runs the Compose GPU
stack concurrently on the same box. Migrating means decommissioning the Compose
GPU stack on the cuda node and letting the scheduler own the GPUs — when GPUs are
busy the sweep's pods sit `Pending` rather than contending.

## Design

### Operational model: one ephemeral namespace per model

For each model in `finetune-sweep.yaml`, serially:

1. **Pre-clean** (idempotent): `kubectl delete namespace sweep-<model>
   --ignore-not-found --wait` — so a crashed previous run can't wedge this one.
2. **Install**: `helm upgrade --install scalarlm <chart_path> -n sweep-<model>
   --create-namespace --set model=<model> --set storage.cache.kind=hostPath
   --set storage.cache.hostPath=<node-cache-dir>` (no `--wait` — see
   block-and-wait). Release name is fixed `scalarlm`; the namespace provides
   isolation, so resources are `deploy/scalarlm-vllm`,
   `statefulset/scalarlm-megatron`, `svc/scalarlm`.
3. **Block until ready** (GPU-aware, below).
4. **Port-forward**: `kubectl port-forward -n sweep-<model> svc/scalarlm
   8000:8000` as a child process; run the existing baseline/train/serve HTTP
   flow against `http://localhost:8000`.
5. **Checkpoint read** via `kubectl exec` (below).
6. **Teardown**: kill the port-forward, then `kubectl delete namespace
   sweep-<model> --wait`.

Model id is sanitized to an RFC1123 label for the namespace:
`Qwen/Qwen2.5-0.5B` → `sweep-qwen-qwen2-5-0-5b` (lowercase; non-alphanumeric runs
→ `-`; trim). The existing `model_id.replace('/', '_')` (used for log filenames)
is insufficient — `_` and `.` are illegal in namespace names.

### Manifest (`finetune-sweep.yaml`)

The `cuda` target drops the Compose fields (`compose_service`, `restart_cmd`)
for k8s identifiers:

```yaml
targets:
  cuda:
    chart_path: deployment/helm/scalarlm
    release: scalarlm                 # fixed; namespace isolates
    namespace_prefix: sweep           # → sweep-<sanitized-model>
    megatron_sts: scalarlm-megatron   # kubectl exec target (StatefulSet)
    api_service: scalarlm             # svc for port-forward (== release fullname)
    cache_hostpath: /root/.cache      # node's populated HF cache (see caveat)
    gpu_wait_timeout: 7200            # outer cap (s) on block-and-wait
    train_args_overrides:
      gpus: 1
```

`model` is passed inline via `--set` (the sweep manifest is the single source of
per-model truth; no throwaway values files).

### GPU scheduling: block-and-wait, fail-fast on real errors

Each model brings up **two always-on GPU pods**: `scalarlm-vllm` (Deployment,
`nvidia.com/gpu: 1`) and `scalarlm-megatron` (StatefulSet, `nvidia.com/gpu: 1`,
runs slurmd; training executes here). So a model consumes **2 GPUs** for its
whole lifetime. On a shared, busy cluster a fresh install may not schedule
immediately.

`helm --wait` cannot express "wait indefinitely for a GPU but not for a crash",
so the runner does **not** use it. After `helm upgrade --install`, it polls
`kubectl get pods -n sweep-<model>` and classifies:

- **`Pending` / `Unschedulable`** (insufficient `nvidia.com/gpu`) → **keep
  waiting** — this is the sanctioned "GPUs busy" state; the scheduler will place
  the pods when a GPU frees.
- **`CrashLoopBackOff` / `ImagePullBackOff` / `Error` / readiness failing after
  the container started** → **fail fast** → `RESTART_FAILED` (don't block on a
  genuinely broken rollout).
- **All pods `Ready`** → proceed.

A generous outer cap (`gpu_wait_timeout`, default 2h) bounds the block purely as
a safety valve; expiry → `RESTART_FAILED` with a "never scheduled within Ns"
detail. The sweep runs **serially** (one `sweep-<model>` at a time) so it never
holds more than 2 GPUs.

### Checkpoint read: `docker compose exec` → `kubectl exec`

The in-container Python (glob `checkpoint_*.pt`, `torch.load` the latest,
`json.dumps` its `model_state_dict` keys) is **unchanged** — the jobs PVC still
mounts at `/app/cray/jobs`. Megatron is a **StatefulSet**, so the exec target is
`statefulset/...`, not `deploy/...`:

```python
cmd = ["kubectl", "exec", "-n", namespace,
       f"statefulset/{target_cfg['megatron_sts']}", "--",
       "python3", "-c", script]
```

(The vLLM pod also mounts the jobs PVC and could serve as the exec target; the
megatron pod is chosen to match where training wrote the checkpoint.)

### API reachability: per-model port-forward

Each model's API Service (`svc/scalarlm`, ClusterIP, port 8000) lives only in its
own namespace, so the port-forward is per-model: spawn `kubectl port-forward -n
sweep-<model> svc/scalarlm 8000:8000` after readiness, run the unchanged HTTP
helpers against `localhost:8000`, and kill it before the namespace delete. The
child process is bounded to one model's iteration and explicitly torn down. The
runner's HTTP layer is untouched (it already targeted `localhost:8000` for the
Compose path).

### VRAM gate: dropped (runtime), static check optional

`probe_gpu_free_gb` used host `nvidia-smi`, which is unavailable and now
meaningless — the scheduler arbitrates placement and the pods own whole GPUs.
Drop the runtime probe. Optionally keep a **static** `gate_gb`-vs-known-GPU-size
check for a `SKIPPED` reason; `gate_model` loses its `free_gb` argument.

### What explicitly does NOT change

- **The chart.** No `checksum/config` annotation, no `Recreate` strategy — both
  were only needed to make an *in-place* `--set model` switch roll the pod.
  Namespace-per-model installs a fresh pod every time, which reads the model at
  startup. The chart is used unmodified.
- **The HTTP helpers, outcome classification, dataset/memorization logic, and
  `sweep_run_id` dedup-defeat.**

## Summary of changes

| Concern | Before (Compose) | After (Helm / namespace-per-model) |
|---|---|---|
| Stack launch | `Popen` of `./scalarlm up`, non-blocking | `helm upgrade --install scalarlm -n sweep-<model> --create-namespace --set model=…` |
| Model switch | `SCALARLM_MODEL` env + `--force-recreate` | fresh per-model namespace (new pod reads ConfigMap at startup) |
| Readiness | `wait_for_all_up` watches `proc` + polls health | poll pod phases (block on Unschedulable, fail-fast on crash) + `/v1/health` |
| Teardown | `killpg` + `nvidia-smi` settle | `kubectl delete namespace sweep-<model>` (GCs the `resource-policy: keep` PVCs too) |
| Checkpoint read | `docker compose exec -T <svc>` | `kubectl exec statefulset/scalarlm-megatron` |
| API access | `localhost:8000` (compose port) | per-model `kubectl port-forward svc/scalarlm 8000:8000` |
| GPU/VRAM gate | host `nvidia-smi` probe | dropped (scheduler owns GPUs) |
| manifest `cuda` target | `compose_service` + `restart_cmd` | `chart_path`/`release`/`namespace_prefix`/`megatron_sts`/`api_service`/`cache_hostpath`/`gpu_wait_timeout` |

## Consequences

- No GPU is ever held outside k8s — the scheduler is the single arbiter, the
  constraint being satisfied. When GPUs are busy the sweep blocks (`Pending`)
  instead of contending.
- Per-model wall time is dominated by `helm install` + pod scheduling + model
  pull (from the shared hostPath HF cache) + capture-graph warmup, plus a
  possibly-unbounded GPU wait, rather than a Compose recreate. Serial execution
  caps GPU use at 2 at a time.
- Teardown via namespace delete cannot leak longhorn volumes (it removes the
  `jobs` and `slurm-config` PVCs that `helm.sh/resource-policy: keep` would
  otherwise strand — the cause of the team's repeated `delete pvc --all`).
- The cache hostPath **pins** the sweep's pods to the node holding the cache, so
  the sweep competes for that node's GPUs specifically.
- The Compose GPU stack on the cuda node must be decommissioned so it cannot
  contend with the scheduler.

## GPU model and operational preconditions (k8s)

Confirmed on `blackwell-maxq-0` (2026-06-15): k8s allocates `nvidia.com/gpu` as
an **exclusive, integer, non-overcommittable** resource, and the node runs
`sharing-strategy=none` (no time-slicing / MPS). Consequences the runner cannot
work around:

- **The sweep needs 2 *schedulable* GPUs, not 2 idle ones.** vLLM (Deployment)
  and megatron (StatefulSet) each request `nvidia.com/gpu: 1`, so a model needs
  two whole GPUs reserved for its lifetime. Under Compose both shared one host
  GPU because nothing brokered the device; k8s gives each pod an exclusive card.
- **`Insufficient nvidia.com/gpu` can occur with physically-idle GPUs.** A node
  can show free cards in `nvidia-smi` while k8s reports it full, because
  scheduling is by *request*, not utilization — idle tenants (always-on
  `megatron` training pods, a sweep `vllm` wedged on its volume) reserve cards
  they aren't using. The runner correctly **blocks** (`wait_for_pods_ready`) in
  this case; it does not and cannot preempt. Observed live: node `capacity=4
  allocatable=4`, all 4 requested, only 2 in use → sweep `megatron` unschedulable.
- **Operator pre-run check:** confirm a node has **2 free GPU reservations**
  (allocatable minus the sum of `nvidia.com/gpu` requests across pods on it), on
  the node holding the hostPath HF cache, before launching — otherwise the sweep
  blocks up to `--gpu-wait-timeout`.

### Reducing the footprint to 1 GPU (future option)

Three ways to make the sweep need only one GPU; only the first is fully in the
sweep's control:

1. **Phase-scaled time-share (recommended).** Run training and serving in
   different phases of one GPU's life: install with vLLM scaled to 0 so megatron
   holds the single GPU and trains → checkpoint lands on the jobs PVC → `kubectl
   scale` megatron to 0 and vLLM to 1 → vLLM hot-loads the checkpoint and serves.
   The chart already exposes `replicaCounts.{inference,training}` and
   `{vllm,megatron}.enabled`; the runner scales between phases and waits-for-ready
   on each. Peak GPU demand = 1. Cost: more orchestration, and the baseline
   "model already serving" check moves to after the vLLM scale-up (training
   doesn't need vLLM, so it can be down during the train phase).
2. **GPU time-slicing / MPS** (cluster-admin): flip the device plugin's
   `sharing-strategy` off `none` so a card advertises >1 replica; both pods stay
   as-is, but vLLM's `gpu_memory_utilization` (default 0.85) must be lowered so
   training has VRAM (sharing gives no memory isolation). Not self-serviceable.
3. **Co-locate vLLM + training in one GPU-requesting pod** — chart surgery that
   defeats the production split of serving vs. SLURM training; not recommended.

MIG is unavailable on these Blackwell workstation cards (`gpu.mode=graphics`, no
MIG strategy), so hardware partitioning is not an option here.

## Open questions / caveats

- **Node cache path.** The exact hostPath for the shared HF cache on the cuda
  node (`/root/.cache` in `values/gemma4-31b.yaml` vs `/mnt/synology/hf_model_cache`
  noted elsewhere) can't be verified remotely — reuse whatever the team's serving
  values use on that node.
- **Server tree ≠ this branch.** The history references `values/gemma4.yaml`,
  `values/qwen3_6-35b.yaml`, and a `qwen3_4b` chart; this branch has
  `values/gemma4-31b.yaml`, `values/qwen3-next-80b-fp8.yaml`, and no `qwen3_4b`
  chart. Two helm styles also appear (older per-model *chart copy* vs newer
  *values-file*). Reconcile chart/values paths against the actual box before
  hard-coding.
- **Coexistence with team namespaces.** The team keeps persistent model
  namespaces up; the sweep's dedicated `sweep-<model>` namespace avoids name
  collisions, but still competes for GPUs on the pinned node (handled by
  block-and-wait).

## ADR impact

Amend **ADR 0003** ("Fine-tune sweep restarts the stack per model"): on the
`cuda` target, "restart the stack per model" is realized as **a fresh Helm
release per model in a dedicated `sweep-<model>` namespace** (`helm upgrade
--install` → block-and-wait → `kubectl delete namespace`), not a Compose
recreate and not an in-place `--set model`. The "runner runs outside the thing it
restarts" rationale still holds (now: outside the cluster's pod lifecycle). No
chart change is required. The ADR 0003 amendment dated 2026-06-15 records this
namespace-per-model decision.
