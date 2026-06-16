# Fine-tune sweep restarts the stack per model

`test/finetune_sweep/run_finetune_sweep.py` is a tier-(d) integration sweep: for
each model, it submits a tiny **LoRA** fine-tune job through the cray stack,
verifies the resulting checkpoint, hot-loads the adapter into the running vLLM
engine, and checks whether the served output reflects the fine-tune. To do the
hot-load+serve check, the cray server must already be **serving that model as the
Base Model** — so the runner restarts the whole stack once per model.

## Why

**Why restart per model.** The hot-load+generate check needs `config["model"]`
(the Base Model, bound at vLLM startup) to match the model being fine-tuned —
per `CONTEXT.md`, changing it requires a full server restart, not just an engine
swap. Three options were considered:

1. Train all models, serve-check only the currently-served one — no restart, but
   only one model per run gets the full closed-loop check.
2. **Restart the stack per model (chosen)** — every model gets the full closed
   loop. The chosen model set is 4 tiny-random stubs with low-double-digit-second
   GPU time, so the restart cost is small relative to full per-model coverage.
3. Single-model invocation, operator manages restarts — simplest runner, but no
   single command sweeps multiple models.

Since this sweep is **LoRA only**, only one restart per model is needed (no
second adapter-type pass within the same serving session — see ADR 0004).

**Why a separate manifest from `model-sweep.yaml`.** `test/finetune_sweep/finetune-sweep.yaml`
has a different model set (4 tiny-random stubs vs. the full serve-test catalog)
and per-model fields specific to fine-tuning (`adapters.lora.gate_gb`, no
`requires`/`chat_template_kwargs`). The two can converge later if the model sets
converge; for now a separate file avoids overloading `model-sweep.yaml`'s schema.

**Why the runner runs on the host, not inside the cray container.**
`./scalarlm up <target>` is `docker compose -f docker-compose.yaml up <service>
--build --force-recreate` — a host-level operation. A runner living inside the
container it's restarting would kill itself. The runner therefore:

- talks to the cray API over plain HTTP (stdlib `urllib`), not the `scalarlm` SDK
  (which only imports inside the container);
- probes free VRAM via `nvidia-smi` (not `torch`);
- uses a single `docker compose exec` to read LoRA checkpoint keys via
  in-container `torch.load` — the only step that genuinely needs the container's
  Python environment.

**Why a per-target `gpus` override.** `JobConfig` defaults `gpus: 1`
(`infra/cray_infra/util/default_job_config.py`), and `validate_gpu_request`
(`infra/cray_infra/training/launch_training_job.py`) rejects `gpus >= 1` with
HTTP 400 when no SLURM node advertises a GPU — true on the `cpu` target. The
manifest's `targets.cpu.train_args_overrides: {gpus: 0}` opts out; `cuda` sets
`{gpus: 1}` to make the override explicit and symmetric.

**Why `must_contain`-on-a-random-string is the pass criterion, and why
`NO_MEMORIZATION` is non-failing.** The training set teaches the model to map
`golden_prompt` -> a random hex string a random-weight model will never produce
unattributed. Its presence in the adapter's response is a clean, unambiguous
signal that the adapter trained AND is being served — without needing the
output to otherwise be coherent. As of this sweep's design, LoRA memorization at
the current hyperparameters (`max_steps: 300`, `learning_rate: 3e-2`,
`train_lm_head=False`) has not been achieved (see the design spec's "Open
questions"); reporting `NO_MEMORIZATION` as a hard failure would make the sweep
permanently red for a known, documented reason. It is therefore a non-failing
outcome — `ADAPTER_NOT_LOADED`, `BAD_CHECKPOINT`, `TRAIN_FAILED`,
`TRAIN_TIMEOUT`, and `RESTART_FAILED` remain hard failures.

## Shape

- **Restart mechanism**: `subprocess.Popen(cmd, shell=True, start_new_session=True)`
  launches `VLLM_SOURCE=remote SCALARLM_MODEL={model} ./scalarlm up {target}` in its own process
  group (mirrors `test/model_sweep/run_sweep.py:279-280`). The runner polls
  `GET /v1/health` until `"all": "up"` (`wait_for_all_up`) or `--restart-timeout`
  -> `RESTART_FAILED`. Teardown is `os.killpg(..., SIGKILL)` + a VRAM-reclamation
  wait (`teardown_stack`, mirrors `teardown_engine`,
  `test/model_sweep/run_sweep.py:111-138`) — `--force-recreate` on the next `up`
  handles container cleanup. Applies identically to `cpu` and `cuda`.
- **Outcome enum** (best -> worst): `PASS`, `NO_MEMORIZATION`,
  `ADAPTER_NOT_LOADED`, `BAD_CHECKPOINT`, `TRAIN_FAILED`, `TRAIN_TIMEOUT`,
  `RESTART_FAILED` (model-level), `SKIPPED` (model-level — LoRA gate doesn't fit,
  or `cpu_ok: false` on cpu).
- **Dedup defeat**: each invocation injects a `sweep_run_id` nonce into
  `train_args` so `launch_training_job`'s `sha256(train_args)` job-dir cache
  (where `train_args` includes a `dataset_hash` field derived from the dataset
  content) never returns a stale job for a fresh sweep run.

## Consequences

- Per-model wall time is dominated by the restart and the training job, not the
  serve check — but every model gets the full train -> hot-load -> serve loop,
  not just the currently-served one.
- Until the LoRA-memorization open question (ADR-adjacent, see the design spec)
  is resolved, every row is expected to report `NO_MEMORIZATION`. The sweep
  stays green but does not yet *prove* memorization — only that the pipeline
  mechanics (train, checkpoint, hot-load, serve) work end to end.
- A future Tokenformer pass (ADR 0004) would need either a second restart cycle
  per model (if it needs its own serving mode) or to be folded into the same
  serving session if/when Tokenformer can be hot-loaded alongside LoRA.

## Amendment 2026-06-15 — "per model" is a fresh Helm release per namespace on the k8s target

**Status:** the *decision* (give every model the full closed loop by isolating
it per model) stands; on the `cuda` target the *mechanism* changes from a
Compose recreate of one shared stack to **a fresh Helm release per model in its
own namespace**. See
`docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md`.

**Why.** On `blackwell-maxq-0` the GPU box runs Kubernetes, and the k8s
scheduler must be the **single authority over GPU allocation**. The original
mechanism — `SCALARLM_MODEL={model} ./scalarlm up {target}` =
`docker compose up --force-recreate` — starts a vLLM container that occupies a
GPU **outside** the scheduler's knowledge, so k8s can double-place a pod onto a
busy device → CUDA OOM / contention. The supervisor's directive "use the
cluster instead of running scripts directly" is, concretely, "no GPU work
outside the scheduler" — it does not forbid running `helm`/`kubectl` from the
host. This is a hardware-environment constraint, not a change to what the sweep
proves.

**Why a fresh release per namespace (not an in-place model switch).** The team
already runs the stack as **one Helm release per model, per namespace**
(`helm upgrade --install <name> scalarlm -f values/<model>.yaml -n <ns>
--create-namespace`; teardown `helm uninstall`), driven from the host shell. The
sweep mirrors this. An in-place `helm upgrade --set model=X` on a long-lived
release was considered and rejected: the chart feeds the base model only through
the `cray-config.yaml` ConfigMap with **no `checksum/config` annotation**, so
changing the value does **not** roll the pod (k8s does not restart pods on
ConfigMap content changes) — the running vLLM would keep serving the old model
while `helm --wait` falsely reported success. Installing a fresh per-model
namespace sidesteps this entirely: a brand-new pod reads the model at startup.
**Consequently no chart change is required** (the earlier draft of this
amendment proposed a `checksum/config` annotation and a `Recreate` strategy;
both are unnecessary under namespace-per-model and are withdrawn).

**What changes (cuda target).** The "Shape" section's restart mechanism is
superseded as follows, per model, serially:

- **Launch**: `helm upgrade --install scalarlm {chart_path} -n sweep-{model}
  --create-namespace --set model={model} --set storage.cache.kind=hostPath
  --set storage.cache.hostPath={node_cache}` replaces the non-blocking `Popen`
  of `./scalarlm up`. Release name is fixed `scalarlm`; the namespace isolates,
  so resources are `deploy/scalarlm-vllm`, `statefulset/scalarlm-megatron`,
  `svc/scalarlm`. The namespace name is the model id sanitized to an RFC1123
  label (`Qwen/Qwen2.5-0.5B` → `sweep-qwen-qwen2-5-0-5b`).
- **Readiness (block-and-wait)**: `helm --wait` is **not** used (it cannot
  distinguish "waiting for a GPU" from "crashing"). The runner polls pod phases
  and classifies: `Pending`/`Unschedulable` (insufficient `nvidia.com/gpu`) →
  **keep waiting**; `CrashLoopBackOff`/`ImagePullBackOff`/`Error` → fail fast →
  `RESTART_FAILED`; all `Ready` → proceed, then confirm with a `/v1/health` poll.
  A generous outer cap (`gpu_wait_timeout`, default 2h) bounds the wait →
  `RESTART_FAILED` on expiry. Each model holds **2 GPUs** for its lifetime —
  vLLM (Deployment) and megatron (StatefulSet) each request `nvidia.com/gpu: 1`
  — so the sweep runs serially.
- **API access**: a per-model `kubectl port-forward -n sweep-{model}
  svc/scalarlm 8000:8000` child process; the existing HTTP helpers talk to
  `localhost:8000` unchanged, and it is killed before teardown.
- **Teardown**: `os.killpg` + the `nvidia-smi` VRAM-settle wait are removed.
  Teardown is `kubectl delete namespace sweep-{model}` — which also GCs the
  `jobs`/`cache`/`slurm-config` PVCs that carry `helm.sh/resource-policy: keep`
  (a plain `helm uninstall` would strand them, the cause of the team's repeated
  `kubectl delete pvc --all`). An idempotent pre-clean (`kubectl delete namespace
  ... --ignore-not-found --wait`) precedes each install.
- **In-container checkpoint read**: `docker compose exec -T {service}` becomes
  `kubectl exec statefulset/scalarlm-megatron` (megatron is a StatefulSet, not a
  Deployment; the in-container `torch.load` script is unchanged — the jobs PVC
  still mounts at `/app/cray/jobs`).
- **Manifest**: the `cuda` target's `compose_service` + `restart_cmd` fields are
  replaced by `chart_path` / `release` / `namespace_prefix` / `megatron_sts` /
  `api_service` / `cache_hostpath` / `gpu_wait_timeout`.

**What is unchanged.** The per-model isolation decision, the host-side (now:
outside-the-cluster-pod-lifecycle) execution model and its "the runner must not
live inside the thing it restarts" rationale, the outcome enum, the memorization
pass criterion, `NO_MEMORIZATION` as non-failing, the `sweep_run_id`
dedup-defeat, and the HTTP helpers. Approach A (host-run `helm`) is what the team
does, so no in-cluster RBAC Job is needed. The `cpu` target (no GPU) is out of
scope for this amendment and may stay on Compose.

**New consequence.** Per-model wall time is dominated by `helm install` + pod
scheduling + model pull (from the shared hostPath HF cache) + capture-graph
warmup, plus a possibly-unbounded GPU wait, rather than a Compose recreate.
Serial execution caps GPU use at 2 at a time; when GPUs are busy the sweep blocks
(`Pending`) instead of contending. The Compose GPU stack on the cuda box must be
decommissioned so it cannot contend with the scheduler — the very conflict this
amendment prevents.

**GPU-accounting consequence (confirmed on `blackwell-maxq-0`, 2026-06-15).**

> *(amended 2026-06-16: the GPU-accounting consequence below is what the
> phase-scaled mode, described in the next amendment, exists to address.)*
Because k8s allocates `nvidia.com/gpu` as an exclusive, integer,
non-overcommittable resource and this cluster runs `sharing-strategy=none`, the
sweep needs **two *schedulable* GPUs per model** (vLLM + megatron each request
one), and can sit `Pending` with `Insufficient nvidia.com/gpu` even when
`nvidia-smi` shows idle cards — scheduling is by request, not utilization, and
idle tenants (always-on megatron pods, a volume-wedged vLLM) reserve cards they
don't use. Under Compose the two shared one host GPU because nothing brokered the
device; that sharing is gone on k8s. Making the sweep a 1-GPU job requires
phase-scaling vLLM/megatron, cluster-level GPU time-slicing/MPS, or co-location —
see the "GPU model and operational preconditions" section of
`docs/superpowers/specs/2026-06-15-finetune-sweep-k8s-design.md`.

## Amendment 2026-06-16 — optional phase-scaled mode runs the closed loop on one GPU

**Status:** an **opt-in** addition (`phase_scaled: true` on the `cuda` target); the
2-GPU namespace-per-model path of the 2026-06-15 amendment remains the default and
is unchanged. Of the three routes to one GPU named in the GPU-accounting
consequence above, only phase-scaling is fully in the sweep's control
(time-slicing needs the GPU-operator owner; co-location is chart surgery). See
`docs/superpowers/specs/2026-06-15-finetune-sweep-1gpu-phase-scaled-design.md`.

**Decision.** A phase-scaled run executes the closed loop in two **sequential
single-GPU phases** instead of two always-on GPU pods, so peak GPU demand is one
card. Per model: install with vLLM scaled to 0 and megatron to 1 → **train phase**
(megatron holds the GPU; checkpoint read here) → **GPU handoff** (scale megatron→0,
wait for the pod to fully delete so its `nvidia.com/gpu` request is released, then
scale vLLM→1) → **serve phase** (vLLM holds the GPU; baseline + hot-load +
memorization check) → `kubectl delete namespace` teardown.

**Why this is sound (and needs no chart change).** A sweep never needs training and
serving *simultaneously*. `replicaCounts.{inference,training}` already drive the
vLLM Deployment / megatron StatefulSet replica counts, and the api pod (GPU-less)
stays up across both phases. Serving does not need megatron: `find_model` resolves
a `job_hash` by globbing `*.pt` on the **`ReadWriteMany`** `jobs` PVC (mounted by
api + vllm + megatron), so a checkpoint megatron wrote in the train phase is
visible to vLLM after megatron is gone.

**Surprises a future reader should expect (the "why on earth" notes).**

- **The baseline runs *after* training, not before.** The baseline is a control on
  the *base* model and needs vLLM, which is down during the train phase — so it
  moves into the serve phase, right before the adapter hot-load. Still a valid
  control.
- **The health gate is never `health.all == "up"`.** The api aggregates `all` over
  `[api, vllm, megatron]`; in a phase-scaled run exactly one GPU service is up at a
  time, so `all` is structurally `down` in *both* phases. The gate is a single
  component key per phase: `health["megatron"]` (train) / `health["vllm"]` (serve).
- **`replicaCounts.inference=0`, not `vllm.enabled=false`.** The latter drops the
  Deployment entirely, leaving nothing for the serve phase to `kubectl scale` up.

**Trade-off accepted.** Two GPU warmups per model, serially (megatron load, then
vLLM capture-graph warmup), so per-model wall time roughly doubles — the cost of
fitting one schedulable GPU. And the GPU is briefly unreserved during the handoff,
so on a shared cluster a co-tenant can take it and the serve-phase scale-up then
blocks on `gpu_wait_timeout` (→ `RESTART_FAILED`). We accept this: **"1 GPU" means
"1 GPU at a time," not "1 GPU reserved throughout."** Holding the reservation across
the handoff would require both GPU requests alive at once, defeating the goal.

**What is unchanged.** The per-model isolation decision, namespace-per-model
lifecycle, the outcome enum, the memorization criterion, `NO_MEMORIZATION` as
non-failing, the `sweep_run_id` dedup-defeat, the checkpoint-key check, and the
HTTP helpers. The 2-GPU path and the `cpu`/Compose path are untouched.

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
