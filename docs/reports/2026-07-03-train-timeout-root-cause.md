# Finetune-sweep TRAIN_TIMEOUT — real root cause (2026-07-03)

**Verdict:** The `TRAIN_TIMEOUT` results on `cuda-spark` are **false negatives**. The
training jobs succeed; the sweep stops watching before they reach the front of the
single-GPU queue. This is a **structural design mismatch**, not GPU starvation and not
zombie contamination from a prior run. My earlier "zombie resurrection starves the GPU"
conclusion was directionally partial but wrong on the thing that matters (see bottom).

## The smoking gun (Phi-4-mini, clean run 20260703-010021)

Job dir `3b2c312e…` on spark, from the clean run:

| Event | Time | Source |
|---|---|---|
| Job submitted (dir + config.yaml created) | 2026-07-02 **20:31:53** | `config.yaml` mtime |
| Sweep poll deadline (submit + `train_timeout` 1800s) | **21:01:53** | code |
| **Sweep records `TRAIN_TIMEOUT`, tears down** | ~21:02 | run log |
| Training actually **starts** | 2026-07-02 **22:55:54** | `status.json.start_time` |
| Training **completes** (450 steps, 467s, loss→4e-4) | 2026-07-02 **23:03:41** | `status.json` COMPLETED |

The job sat ~2h24m between submit and training start, then trained for 467s and
COMPLETED — **~2h after the sweep had already given up.** `status.json` = `COMPLETED`,
`accumulated_train_seconds` = 467, full clean loss curve. The dir holds two slurm output
files (`slurm-1.out`, `slurm-3.out`) = the job was relaunched.

Same story for the other "timeouts":

| Model | job dir | status.json | actual result | Sweep reported |
|---|---|---|---|---|
| Phi-4-mini | `3b2c312e` | COMPLETED (467s) | trained fine | ❌ TRAIN_TIMEOUT |
| Qwen2.5-VL | `d13408c4` | COMPLETED (3575s, 900 steps) | trained fine | ❌ TRAIN_TIMEOUT |
| pixtral | `d0eadcb9` | COMPLETED (2267s, 900 steps) | trained fine | ❌ TRAIN_TIMEOUT |
| GLM-4 | `236c7691` | FAILED (custom code / trust_remote_code) | failed at load | ❌ TRAIN_TIMEOUT |
| InternVL3 | `cf40fec1` | FAILED (custom code) | failed at load | ✅ TRAIN_FAILED |
| Molmo | `a0c7d6df` | FAILED (custom code) | failed at load | ✅ TRAIN_FAILED |

All 55 job dirs on the box are terminal (37 COMPLETED / 17 FAILED); **zero** are
TRAINING/QUEUED. Nothing is stuck.

## Causal chain

1. `poll_training` has a fixed ceiling (`train_timeout`, 1800/4800s). On expiry the sweep
   declares `TRAIN_TIMEOUT` and calls `teardown_stack` + `compose_rm` to "free the GPU"
   (run_finetune_sweep.py ~line 1054).
2. `compose_rm` removes the container and kills the current slurm slice — but the job's
   `status.json` stays **QUEUED/TRAINING** (non-terminal), persisted in the bind-mounted
   `./jobs` dir.
3. The API server runs `restart_megatron_jobs()` **periodically** (`@repeat_every`,
   `add_megatron_tasks.py`). It relaunches every job dir whose status is `TRAINING` or
   `QUEUED` and that is missing from `squeue` (`restart_megatron_jobs.py:64-67`) — by
   design, to survive SLURM slice timeouts.
4. So every job the sweep "abandons" is **resurrected** and re-enters the single-GPU
   queue. On a phase-scaled box (one training at a time), the resurrected jobs plus any
   leftovers from prior runs run **strictly serially**, minutes-to-an-hour each.
5. Each model's `poll_training` ceiling elapses while its job is still waiting behind the
   backlog → `TRAIN_TIMEOUT`, even though the job later runs and COMPLETES.

**Why InternVL3/Molmo were reported correctly but GLM-4 wasn't:** purely queue timing.
InternVL3/Molmo fail in seconds at model load; whenever they reached the GPU their
status.json went terminal (FAILED) fast enough to be observed, and FAILED jobs are *not*
resurrected. GLM-4 also fails fast at load, but it only reached the GPU at 00:07 — long
after its poll window — so the sweep timed out first. The outcome (COMPLETE vs FAIL) is
irrelevant; the only thing that decides observe-vs-timeout is whether the job reached the
front of the single-GPU queue before the poll ceiling.

## Ruled out (with evidence)

- **status.json first-line parse bug** (`json.loads(file.readline())`): status.json is
  single-line and parses to COMPLETED. Not triggered here.
- **Synology stale-dir shadowing** (`get_job_directory_for_hash` checks `/mnt/synology`
  first): `/mnt/synology/jobs` does not exist on the Spark.
- **Hash mismatch (poll wrong hash):** `sweep_run_id` is baked into `train_args`, hash is
  deterministic per run, config.yaml `job_directory` basename == dir name == polled hash.
  Poll polled the right hash; the status just wasn't terminal yet.
- **External zombie starving the GPU:** the GPU was busy with the sweep's *own* resurrected
  backlog, not an outside process.

## The fix (root cause, not symptom)

On `TRAIN_TIMEOUT` (and any abandonment), the sweep must drive the job to a **terminal**
status before/at teardown so the reconciler won't resurrect it:

- Call the existing cancel path (writes `CANCELLED` to status.json + `scancel`) for the
  `job_hash` — `infra/cray_infra/training/cancel.py` already does exactly this — instead of
  relying on `compose_rm` to kill the slice. `compose_rm` is futile against the periodic
  reconciler.
- At **run start**, terminal-state / cancel ALL pre-existing non-terminal jobs so no prior
  backlog leaks in.

Note the previously-shipped "cascade fix #6" (`compose_rm` in the phase-2 early return)
does **not** address this — the reconciler undoes it. Even on a perfectly clean box, the
first model to exceed its poll ceiling gets resurrected and clogs the queue for everyone
after it.

Secondary: on a single-GPU box, a fixed poll ceiling shorter than queue+train latency will
always misfire for large jobs. Cancel-on-timeout at least prevents one slow model from
cascading into false timeouts for the rest.
