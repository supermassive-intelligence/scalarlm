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
