# Fine-tune sweep design

_Design spec for a tier-(d) integration sweep: can each model in scope be
fine-tuned (LoRA / Tokenformer), have the resulting adapter hot-loaded into
the running vLLM engine, and does the served output reflect the fine-tune?
This is the "Tokenformer training compatibility" tier named but not yet
implemented in `CONTEXT.md`._

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
- **Adapter types**: both LoRA and Tokenformer, per model, gated by the
  LoRA/Tokenformer VRAM gates from the sizing table in
  `docs/reports/fine-tuning-a-served-model.md`.
- **Models**: the 4 tiny-random stubs from `model-sweep.yaml`
  (`tiny-random/gemma-4-dense`, `masint/tiny-random-llama`,
  `masint/tiny-random-qwen2-vl`, `yujiepan/qwen3-moe-tiny-random`). All fit
  comfortably under either gate (<8 GB, <1s GPU-time for the default run per
  the sizing table) — this is a pipeline-mechanics smoke test, not a quality
  benchmark. Larger models can be added to the manifest later the same way
  `model-sweep.yaml` grew.
- **Targets**: both `cpu` and `cuda`, manifest-driven like `model-sweep.yaml`.
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
   with sub-second GPU time, so the restart cost is small relative to the
   value of full per-model coverage.
3. **Single-model invocation, operator manages restarts** — simplest runner,
   but no single command sweeps multiple models.

Within one serving session for a model, **both adapter types are exercised**
(LoRA then Tokenformer, whichever fit the gate) — only one restart per model,
not per (model, adapter type).

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

train_args_defaults:
  max_steps: 20
  steps_per_checkpoint: 10
  learning_rate: 3e-3
  max_token_block_size: 4096
  dtype: float32

targets:
  cpu:
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cpu"
  cuda:
    restart_cmd: null   # operator restarts manually; runner waits for health

models:
  - id: tiny-random/gemma-4-dense
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}, tokenformer: {gate_gb: 8}}
  - id: masint/tiny-random-llama
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}, tokenformer: {gate_gb: 8}}
  - id: masint/tiny-random-qwen2-vl
    multimodal: true
    adapters: {lora: {gate_gb: 8}, tokenformer: {gate_gb: 8}}
  - id: yujiepan/qwen3-moe-tiny-random
    adapters: {lora: {gate_gb: 8}, tokenformer: {gate_gb: 8}}
```

Notes:

- `adapters.<type>.gate_gb` reuses the sizing-table gates. On `cuda`, the
  runner probes free VRAM (reuse `model_sweep.probe_gpu_free_gb`) and only
  attempts adapter types whose gate fits; on `cpu`, gates are ignored.
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

1. **Gate adapter types.** On `cuda`, probe free VRAM and keep only adapter
   types whose `gate_gb` fits; on `cpu`, gates are ignored. If the model isn't
   `cpu_ok` on the cpu target, or no adapter type fits on cuda -> `SKIPPED`
   (model-level, both adapter types).
2. **Restart**, serving this model. Run
   `targets.<target>.restart_cmd.format(model=model.id)` as a subprocess
   expected to return once the new container is up in the background (e.g.
   `docker compose ... up -d --force-recreate`), then poll
   `GET <api-url>/v1/health` until `{"api": "up", "vllm": "up"}` or
   `--restart-timeout` -> `RESTART_FAILED` (model-level). For `cuda`
   (`restart_cmd: null`), print the model id and just poll health, so an
   operator can restart the remote stack manually.
3. **Baseline generate** — `llm.generate(prompts=[golden_prompt], model_name=model.id)`.
   Captured for the memorization check in step 4g. (Edge case: if the
   baseline already contains `expected_output` — astronomically unlikely for
   a random hex string — note it in `detail` rather than adding a dedicated
   outcome.)
4. **For each adapter type that fit the gate** (lora, tokenformer — both run
   in this one serving session, no extra restart):
   - **a.** Build `train_args = {llm_name: model.id, adapter_type, **train_args_defaults, sweep_run_id: <nonce>}`.
   - **b.** Submit via `llm.train(dataset, train_args=train_args)`.
   - **c.** Poll `status.json` until `COMPLETED`/`FAILED`/`CANCELLED` or
     `--train-timeout` -> `TRAIN_FAILED` / `TRAIN_TIMEOUT`.
   - **d.** Load the resulting `checkpoint_*.pt` and check it has the
     expected key pattern (`lora_A`/`lora_B` for LoRA, `tokenformer_p` for
     Tokenformer) -> `BAD_CHECKPOINT` if missing/empty.
   - **e.** Poll `llm.generate(model_name=<job_hash>)` until it stops 404'ing
     (adapter hot-loaded) or `--serve-timeout` -> `ADAPTER_NOT_LOADED`.
   - **f.** Generate against the adapter with `golden_prompt`.
   - **g.** Assert `expected_output in adapter_text` -> `PASS` if yes,
     `NO_MEMORIZATION` if not.
   - Record one `Result` row per (model, target, adapter_type).

### Outcome enum (best -> worst)

| Outcome | Meaning |
|---|---|
| `PASS` | trained, checkpoint verified, adapter hot-loaded, response contains the memorized string |
| `NO_MEMORIZATION` | adapter hot-loaded and responded, but didn't memorize — possible training/loss-masking regression |
| `ADAPTER_NOT_LOADED` | checkpoint OK but never became servable — possible discovery/hot-load regression |
| `BAD_CHECKPOINT` | job COMPLETED but checkpoint missing or has wrong keys for the adapter type |
| `TRAIN_FAILED` | job reached FAILED/CANCELLED |
| `TRAIN_TIMEOUT` | job didn't reach a terminal status in time |
| `RESTART_FAILED` | (model-level) stack never came up serving this model |
| `SKIPPED` | (model-level) no adapter type fits the gate, or `cpu_ok: false` on cpu |

### `Result` fields

`model`, `target`, `adapter_type` (null for model-level outcomes), `outcome`,
`detail`, `baseline_sample`, `adapter_sample`, `restart_seconds`,
`train_seconds`, `serve_seconds`.

## Reporting & CLI

Same conventions as `run_sweep.py`:

- **Reports**: `test/finetune_sweep/results/finetune.<target>.<timestamp>.{json,md}`.
  JSON is the full list of `Result` dicts; Markdown is a table —
  `| Model | Adapter | Outcome | Detail | Baseline sample | Adapter sample | restart_s | train_s | serve_s |`
  — printed to stdout at the end.
- **CLI flags**: `--target` (required: `cpu`/`cuda`), `--manifest` (default
  `finetune-sweep.yaml`), `--results-dir`, `--models` (optional subset),
  `--api-url` (default `http://localhost:8000`, override for the remote cuda
  boxes), `--restart-timeout`, `--train-timeout`, `--serve-timeout`.
- **Health check**: `GET <api-url>/v1/health` (the combined
  `{"api": "up", "vllm": "up"}` endpoint) — used both after restart and as the
  readiness gate.
- **Exit code**: non-zero if any result's outcome is anything other than
  `PASS`/`SKIPPED` — this sweep is a regression gate, not just informational
  like the serve sweep's narrower `FAILED_TO_SERVE`/`OOM` hard-fail set.

## Testing plan

- **Unit tests** (`test/unit/`) for the pure logic that doesn't need the live
  stack:
  - adapter-type gating (which adapters fit given probed/declared free VRAM
    vs `gate_gb`)
  - checkpoint key-pattern check (`lora_A`/`lora_B` vs `tokenformer_p`)
  - outcome classification (given a sequence of step results, which enum
    value comes out)
  - manifest loading / `--models` filtering
- **The runner itself** is validated by actually running it against
  `--target cpu` (manual or CI), the same way `run_sweep.py` has no unit
  tests of its own — it's an integration driver, proven by running it.

## New ADR

`docs/adr/0003-finetune-sweep-restart-per-model.md`, following the existing
0001/0002 pattern, documenting:

- why this is a separate manifest from `model-sweep.yaml`
- why the runner restarts the stack per model (vs. the alternatives discussed
  above)
- the `must_contain`-on-a-random-string memorization check as the tier-(d)
  pass criterion referenced in `CONTEXT.md`
