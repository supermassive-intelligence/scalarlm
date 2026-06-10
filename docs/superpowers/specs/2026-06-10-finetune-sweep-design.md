# Fine-tune sweep design

_Design spec for a tier-(d) integration sweep: can each model in scope be
fine-tuned with **LoRA**, have the resulting adapter hot-loaded into the
running vLLM engine, and does the served output reflect the fine-tune? This is
the LoRA half of the "Adapter training+serving compatibility" tier named in
`CONTEXT.md`. **Tokenformer is out of scope for this spec** — see "Open
questions" and ADR 0004._

## Motivation

`test/model_sweep/run_sweep.py` (tiers a/b) proves a base model **serves**
under the vLLM fork. It deliberately bypasses the cray stack (ADR 0001) and
says nothing about ScalarLM's own fine-tuning loop. `docs/reports/fine-tuning-a-served-model.md`
worked through the train -> adapter -> hot-load -> serve loop manually for one
model and sized LoRA/Tokenformer feasibility for every model in
`model-sweep.yaml`. This spec turns that worked example into a repeatable,
reportable sweep — the tier-(d) integration harness ADR 0001 anticipated.

## Scope

- **Pass criteria**: full closed loop — submit a tiny fine-tune job through
  the cray stack, verify it completes with a correct adapter checkpoint, hot-load
  the adapter into the running vLLM engine, and confirm the served output
  reflects the fine-tune (not just "did it change").
- **Adapter type**: **LoRA only**, gated by the LoRA VRAM gate from the sizing
  table in `docs/reports/fine-tuning-a-served-model.md`. Tokenformer adapters
  train fine but currently cannot be hot-loaded/served at all (see "Open
  questions" below) — they are excluded from this sweep entirely, not just
  skipped per-model.
- **Models**: the 4 tiny-random stubs from `model-sweep.yaml`
  (`tiny-random/gemma-4-dense`, `masint/tiny-random-llama`,
  `masint/tiny-random-qwen2-vl`, `yujiepan/qwen3-moe-tiny-random`). All fit
  comfortably under the LoRA gate (<8 GB) — this is a pipeline-mechanics smoke
  test, not a quality benchmark. Larger models can be added to the manifest
  later the same way `model-sweep.yaml` grew.
- **Targets**: both `cpu` and `cuda`, manifest-driven like `model-sweep.yaml`.
  Both targets restart automatically (see "Restart mechanism" below).
- **Manifest**: a new file, `test/finetune_sweep/finetune-sweep.yaml`,
  independent of `model-sweep.yaml` (different model set and per-model fields
  for now; can converge later if the model sets converge).

## Why restart the stack per model

The hot-load+generate check needs the cray server to already be **serving**
the same base model being fine-tuned (`config["model"]` is bound at vLLM
startup; per `CONTEXT.md`, changing it means restarting the whole server, not
just the engine — `kill_vllm_container()` kills all python and exits).

Three options were considered:

1. **Train all models, serve-check only the currently-served one** — no
   restart needed (training is an independent Slurm job), but only one model
   per run gets the full closed-loop check.
2. **Runner restarts the stack per model** (chosen) — every model gets the
   full closed loop. Restarting is normally invasive (shared remote CUDA
   boxes per project notes), but the chosen model set is 4 tiny-random stubs
   with sub-second-to-low-double-digit-second GPU time, so the restart cost is
   small relative to the value of full per-model coverage.
3. **Single-model invocation, operator manages restarts** — simplest runner,
   but no single command sweeps multiple models.

Since this sweep covers **LoRA only**, only one restart per model is needed
(no second adapter-type pass within the same serving session).

### Restart mechanism

`./scalarlm up <target>` (`cmd/up_command.sh`) runs
`docker compose -f docker-compose.yaml up <service> --build --force-recreate`
**in the foreground** — there is no `-d`/detached mode, for either `cpu` or
`cuda`. The runner therefore manages it the same way `test/model_sweep/run_sweep.py`
manages the vLLM-fork process (ADR 0002):

1. `subprocess.Popen([...], start_new_session=True)` to launch
   `SCALARLM_MODEL={model.id} ./scalarlm up {target}` in its own process group,
   non-blocking (mirrors `run_sweep.py:280`).
2. Poll `GET <api-url>/v1/health` until the response's `"all"` field is
   `"up"` (`infra/cray_infra/api/fastapi/health/check_health.py` aggregates
   `api`/`vllm`/`megatron` into `all`) or `--restart-timeout` ->
   `RESTART_FAILED` (model-level), mirroring `wait_for_health`
   (`run_sweep.py:154`).
3. Before the *next* model's restart (or at the end of the sweep),
   `os.killpg(os.getpgid(proc.pid), signal.SIGKILL)` — same as
   `teardown_engine` (`run_sweep.py:111-121`) — then wait for the container to
   exit. `--force-recreate` on the next `up` handles container cleanup; the
   runner does not need `docker compose down`.

This applies identically to `cpu` and `cuda` — both targets restart
automatically; no manual operator step. On a shared multi-GPU `cuda` box, an
operator pins which GPU the stack uses via `.env`'s `CUDA_VISIBLE_DEVICES`
passthrough on `docker-compose.yaml`'s `cray` anchor — a prerequisite for
running this sweep on a shared box, set up independently of the sweep runner
and not part of this spec's implementation.

## Manifest: `test/finetune_sweep/finetune-sweep.yaml`

```yaml
# Shared tiny training set: JSONL {"input","output"} pairs, repeated so packing
# yields a few blocks (per docs/reports/fine-tuning-a-served-model.md worked example).
# The output is a random hex string a random-weight model will never produce
# unattributed -> its presence is a clean signal the adapter trained AND is being
# served, without needing the loss/output to be otherwise meaningful.
dataset:
  examples:
    - {input: "My bank account's balance is", output: " aaaf6f8ae738dfc6577e63dda6daf9cc"}
  repeat: 16

golden_prompt: "My bank account's balance is"
expected_output: "aaaf6f8ae738dfc6577e63dda6daf9cc"   # must_contain on the adapter's response

# NOT YET VALIDATED to achieve memorization — see "Open questions". These are
# the best-tested-so-far values (loss decreases further than the
# fine-tuning-a-served-model.md worked example's 20-step/3e-3 defaults, but
# still plateaus well short of memorization).
train_args_defaults:
  adapter_type: lora
  max_steps: 300
  steps_per_checkpoint: 300
  learning_rate: 3e-2
  max_token_block_size: 4096
  dtype: float32

targets:
  cpu:
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cpu"
  cuda:
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cuda"

models:
  - id: tiny-random/gemma-4-dense
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}}
  - id: masint/tiny-random-llama
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}}
  - id: masint/tiny-random-qwen2-vl
    multimodal: true
    adapters: {lora: {gate_gb: 8}}
  - id: yujiepan/qwen3-moe-tiny-random
    adapters: {lora: {gate_gb: 8}}
```

Notes:

- `adapters.lora.gate_gb` reuses the LoRA sizing-table gate. On `cuda`, the
  runner probes free VRAM (reuse `probe_gpu_free_gb` from
  `test/model_sweep/run_sweep.py`) and only attempts the model if LoRA's gate
  fits; on `cpu`, the gate is ignored.
- **Dedup footgun**: `launch_training_job` keys job directories on
  `sha256(train_args + dataset)`, so an unmodified rerun would return a stale
  cached job. The runner injects a per-invocation `sweep_run_id` nonce into
  `train_args` so every sweep run forces a fresh training job — the sweep
  deliberately defeats that cache, since "did it train *this time*" is the
  point.
- `--models` CLI flag filters the model list, same convention as
  `run_sweep.py`.

## Runner flow: `test/finetune_sweep/run_finetune_sweep.py`

Per model (filtered by `--models`):

1. **Gate LoRA.** On `cuda`, probe free VRAM and check it against
   `adapters.lora.gate_gb`; on `cpu`, the gate is ignored. If the model isn't
   `cpu_ok` on the cpu target, or LoRA doesn't fit on cuda -> `SKIPPED`
   (model-level).
2. **Restart**, serving this model, per "Restart mechanism" above. Poll
   `GET <api-url>/v1/health` until `"all": "up"` or `--restart-timeout` ->
   `RESTART_FAILED` (model-level).
3. **Baseline generate** — `llm.generate(prompts=[golden_prompt], model_name=model.id)`.
   Captured for the memorization check in step 4g. (Edge case: if the
   baseline already contains `expected_output` — astronomically unlikely for
   a random hex string — note it in `detail` rather than adding a dedicated
   outcome.)
4. **Train and verify the LoRA adapter:**
   - **a.** Build `train_args = {llm_name: model.id, **train_args_defaults, sweep_run_id: <nonce>}`.
   - **b.** Submit via `llm.train(dataset, train_args=train_args)`.
   - **c.** Poll `status.json` until `COMPLETED`/`FAILED`/`CANCELLED` or
     `--train-timeout` -> `TRAIN_FAILED` / `TRAIN_TIMEOUT`.
   - **d.** Load the resulting `checkpoint_*.pt` and check it has the
     expected key pattern (`lora_A`/`lora_B`) -> `BAD_CHECKPOINT` if
     missing/empty.
   - **e.** Poll `llm.generate(model_name=<job_hash>)`, retrying on **any**
     exception (not just HTTP 404 — `find_model()` auto-registers the job the
     moment a `.pt` exists, but the generate-worker's async `get_adaptors` ->
     `add_adaptors` -> `/v1/load_lora_adapter` loop runs separately and may
     not have hot-loaded the adapter yet, which surfaces as a 200-with-error
     response, not a 404) until it succeeds or `--serve-timeout` ->
     `ADAPTER_NOT_LOADED`.
   - **f.** Generate against the adapter with `golden_prompt`.
   - **g.** Assert `expected_output in adapter_text` -> `PASS` if yes,
     `NO_MEMORIZATION` if not.
   - Record one `Result` row per (model, target).

### Outcome enum (best -> worst)

| Outcome | Meaning |
|---|---|
| `PASS` | trained, checkpoint verified, adapter hot-loaded, response contains the memorized string |
| `NO_MEMORIZATION` | adapter hot-loaded and responded, but didn't memorize — expected outcome until "Open questions" below is resolved |
| `ADAPTER_NOT_LOADED` | checkpoint OK but never became servable — possible discovery/hot-load regression |
| `BAD_CHECKPOINT` | job COMPLETED but checkpoint missing or has wrong keys for LoRA |
| `TRAIN_FAILED` | job reached FAILED/CANCELLED |
| `TRAIN_TIMEOUT` | job didn't reach a terminal status in time |
| `RESTART_FAILED` | (model-level) stack never came up serving this model |
| `SKIPPED` | (model-level) LoRA gate doesn't fit, or `cpu_ok: false` on cpu |

### `Result` fields

`model`, `target`, `adapter_type` (always `"lora"`, kept for forward
compatibility with a future Tokenformer pass — see ADR 0004), `outcome`,
`detail`, `baseline_sample`, `adapter_sample`, `restart_seconds`,
`train_seconds`, `serve_seconds`.

## Reporting & CLI

Same conventions as `run_sweep.py`:

- **Reports**: `test/finetune_sweep/results/finetune.<target>.<timestamp>.{json,md}`.
  JSON is the full list of `Result` dicts; Markdown is a table —
  `| Model | Outcome | Detail | Baseline sample | Adapter sample | restart_s | train_s | serve_s |`
  — printed to stdout at the end.
- **CLI flags**: `--target` (required: `cpu`/`cuda`), `--manifest` (default
  `finetune-sweep.yaml`), `--results-dir`, `--models` (optional subset),
  `--api-url` (default `http://localhost:8000`, override for the remote cuda
  boxes), `--restart-timeout`, `--train-timeout`, `--serve-timeout`.
- **Health check**: `GET <api-url>/v1/health` (the combined
  `{"api": ..., "vllm": ..., "megatron": ..., "all": ...}` endpoint) —
  `"all": "up"` is used both after restart and as the readiness gate.
- **Exit code**: non-zero if any result's outcome is anything other than
  `PASS`/`SKIPPED`/`NO_MEMORIZATION` — `NO_MEMORIZATION` is treated as an
  expected, non-failing outcome until the hyperparameter question below is
  resolved (otherwise the sweep would be permanently red for a known,
  documented reason). All other non-`PASS`/`SKIPPED` outcomes
  (`ADAPTER_NOT_LOADED`, `BAD_CHECKPOINT`, `TRAIN_FAILED`, `TRAIN_TIMEOUT`,
  `RESTART_FAILED`) remain hard failures.

## Open questions

These were investigated empirically while grilling this spec (against
`tiny-random/gemma-4-dense` on a remote `cuda` box) and are **not yet
resolved**. The runner can be built and run today — both are properties of
*current hyperparameters/serving code*, not of the runner's design — but they
bound what the sweep can currently prove.

### Tokenformer serving is unimplemented (excluded from this spec)

Originally this sweep was to cover both LoRA and Tokenformer per `CONTEXT.md`'s
framing of tier (d) ("Tokenformer training compatibility"). Empirically:

- Training a Tokenformer adapter completes and produces a `.pt` with
  `tokenformer_p` keys.
- Hot-loading it fails: `add_new_adaptor()` (`infra/cray_infra/one_server/create_generate_worker.py:251-290`)
  always calls vLLM's `/v1/load_lora_adapter` regardless of adapter type. The
  vLLM fork rejects the checkpoint: *"Adapter ... has no LoRA tensors (found
  only Tokenformer keys). Serve with `--enable-tokenformer` instead, or as a
  hybrid adapter with both `--enable-lora` and `--enable-tokenformer`."*
- `--enable-tokenformer` is never passed
  (`infra/cray_infra/one_server/vllm_cli_args.py` only conditionally adds
  `--enable-lora`).
- `infra/cray_infra/adapters/` (`TokenformerManager`, `attention_adapter.py`,
  `models.py` — a from-scratch Tokenformer serving layer) is **never imported
  by the running server** — dead scaffolding. Even if wired up, its core
  transform (`attention_adapter.py:_tokenformer_transform`) is a no-op stub
  that returns its input unchanged.

`CONTEXT.md`'s Adapter/Tokenformer entries have been corrected in this change
to stop claiming Tokenformer is "served as a LoRA at inference". Implementing
real Tokenformer serving is a multi-week vLLM-fork project (real forward-pass
algorithm + per-request adapter selection compatible with continuous batching
+ a non-LoRA load endpoint) — out of scope here. See ADR 0004. Once it lands,
this spec can grow a second adapter-type pass per model, the way the original
draft anticipated.

### LoRA memorization not yet achieved at any tested hyperparameters

Two configurations were tried against `tiny-random/gemma-4-dense` on `cuda`:

| max_steps | learning_rate | final loss | starting loss (~ln(vocab)) |
|---|---|---|---|
| 20 | 3e-3 | 12.27 | 12.33 |
| 300 | 3e-2 | 12.169 (plateaus by step ~50) | 12.48 |

Both runs produced gibberish adapter output, not the memorized hex string.
The 300-step run shows the optimizer **converging to a plateau**, not running
out of time — 10x more steps and 10x higher LR than the first run reach
materially the same floor.

Likely cause: `add_adapters_to_model.py:12` calls
`create_lora_model(model=model, device=device)` with the default
`train_lm_head=False` (`ml/adapters/create_lora_model.py:11`) — so only LoRA
deltas (`r=8`, `target_modules="all-linear"`,
`infra/cray_infra/util/default_job_config.py:6-10`) on attention/MLP linears
are trainable. `lm_head` and the embedding matrix stay at their random,
never-pretrained init. LoRA can reshape hidden states but cannot move the
fixed-random vocabulary projection itself, which may bound how much a ~256k-way
softmax can be skewed toward specific tokens at `r=8`.

Candidates for follow-up (not yet tried): expose `train_lm_head` via
`train_args` (a code change — currently a hardcoded function default, not
config); increase `lora_config.r`/`lora_alpha` (already overridable via
`train_args["lora_config"]`); shorten `expected_output` to fewer tokens.

Until resolved, the sweep is expected to report `NO_MEMORIZATION` for every
LoRA row — see "Exit code" above for why that doesn't fail the sweep.

## Testing plan

- **Unit tests** (`test/unit/`) for the pure logic that doesn't need the live
  stack:
  - LoRA gating (does the model fit `gate_gb` given probed/declared free VRAM)
  - checkpoint key-pattern check (`lora_A`/`lora_B`)
  - outcome classification (given a sequence of step results, which enum
    value comes out, including the `NO_MEMORIZATION` non-failing case)
  - manifest loading / `--models` filtering
- **The runner itself** is validated by actually running it against
  `--target cpu` (manual or CI), the same way `run_sweep.py` has no unit
  tests of its own — it's an integration driver, proven by running it.

## New ADRs

- `docs/adr/0003-finetune-sweep-restart-per-model.md`:
  - why this is a separate manifest from `model-sweep.yaml`
  - why the runner restarts the stack per model (vs. the alternatives discussed
    above)
  - the Popen + SIGKILL-process-group restart mechanism, applied uniformly to
    `cpu` and `cuda` (no manual-restart special case)
  - the `must_contain`-on-a-random-string memorization check as the tier-(d)
    pass criterion referenced in `CONTEXT.md`, and `NO_MEMORIZATION` as a
    non-failing outcome pending the open question above
- `docs/adr/0004-defer-tokenformer-serving.md`:
  - the empirical findings above (vLLM fork rejection, dead `adapters/`
    scaffolding, no-op `_tokenformer_transform`)
  - decision: tier (d) covers LoRA only until Tokenformer serving is
    implemented as its own project; `CONTEXT.md` corrected accordingly
