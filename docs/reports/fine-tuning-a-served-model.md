# How ScalarLM fine-tunes a model it is serving

_Investigation report — traces the closed loop from a running inference model to a
fine-tuned adapter and back into the live engine. File references are to the repo
at the time of writing._

## The core idea: it never fine-tunes the base model

The key architectural decision (spelled out in `CONTEXT.md`) is that fine-tuning
**never modifies the served base model's weights**. The base model is bound once at
vLLM startup from `config["model"]` and can only change with a full server restart.
Instead, training produces an **adapter** — a small set of extra weights saved as a
`.pt` file — that gets *hot-loaded* into the running vLLM engine by name, with no
restart. That is what makes the "query the model → build training signal →
post-train → next request picks it up" loop work inside one deployment.

There are two adapter mechanisms:

- **Tokenformer** (the default, ScalarLM's own invention) — attention-style modules
  wrapped around every MLP layer.
- **LoRA** (standard PEFT).

Both are served *as* a LoRA at inference time.

## The end-to-end flow

### 1. Submit (SDK side)

`sdk/masint/api/supermassive_intelligence.py:11` → `sdk/masint/engines/cray/submit_training_job.py`.
The client tars up your `dataset.jsonlines` **plus the local `ml/` directory** and
POSTs it to `/v1/megatron/train`. Shipping `ml/` with every job is deliberate
(README, "Designed for Experimentation"): you edit the training loop locally and the
cluster picks it up with no Docker rebuild.

### 2. Job creation (server side)

`infra/cray_infra/api/fastapi/routers/megatron_router.py:49` →
`infra/cray_infra/training/upload_training_data.py` →
`infra/cray_infra/training/launch_training_job.py`.

The job directory name is `sha256(json.dumps(train_args))`
(`upload_training_data.py:154`), and `train_args` includes the dataset's own hash
(`upload_training_data.py:61`). A Slurm batch job is submitted
(`launch_training_job.py:102`).

### 3. Load the base + graft the adapter

This is the heart of it. `ml/cray_megatron/main.py` →
`ml/cray_megatron/megatron/training_loop.py` → `ml/cray_megatron/models/load_model.py`:

- `AutoModelForCausalLM.from_pretrained(...)` materializes the **same** HF base model
  vLLM serves (`load_model.py:75`).
- `ml/adapters/add_adapters_to_model.py:6` dispatches on `adapter_type`:
  - **Tokenformer** (`ml/tokenformer/tokenformer_surgeon.py`): walks every module,
    and for any layer whose leaf name contains `mlp` (`_is_adapter_layer`, line 143),
    replaces it with a `TokenformerAdapter` that runs the original layer **plus** a
    learned attention-style residual (`forward`, line 54). Vision/audio towers are
    explicitly excluded (`_NON_LANGUAGE_PATH_COMPONENTS`).
  - **LoRA** (`ml/adapters/create_lora_model.py`): standard `get_peft_model`.
- Then **everything is frozen, and only adapter params are unfrozen**
  (`ml/adapters/create_tokenformer_model.py:38-72` /
  `ml/adapters/create_lora_model.py:52-76`). `lm_head` is trained only for models
  under 100M params.

### 4. Train

`ml/cray_megatron/megatron/training_loop.py`: AdamW, gradient accumulation, gradient
clipping, NaN-skip logic, FSDP/DDP, periodic checkpointing. `filter_checkpoint`
(line 744) saves **only `requires_grad=True` params** — so a checkpoint is just the
adapter delta, not the whole model. Checkpoints land as `checkpoint_<step>.pt` in the
job directory.

Dataset format is JSONL with `{"input": ..., "output": ...}`. Labels mask the input
region with `-100` so loss is computed only on the output
(`ml/cray_megatron/megatron/dataset/load_language_model_dataset.py:96`).

### 5. Discovery + hot-load into inference

This is the seam between training and serving:

- `infra/cray_infra/training/register_megatron_models.py:33` scans the job directory
  for any subdir containing a `*.pt` and registers it in `VLLMModelManager`
  (`infra/cray_infra/training/vllm_model_manager.py`).
- The generate worker loop (`infra/cray_infra/one_server/create_generate_worker.py:104`)
  polls for work; each `get_work` response carries `new_adaptors`, and
  `add_new_adaptor` (line 251) POSTs to vLLM's **`/v1/load_lora_adapter`** — i.e., the
  Tokenformer is served *as* a LoRA at inference time.
- A generate request names a model; `find_model`
  (`vllm_model_manager.py:44`) resolves it to the base model, a registered adapter, or
  **auto-discovers** an unregistered one on disk.

### 6. (Optional) Publish to HF

`ml/adapters/merge_lora_and_push.py` folds a LoRA back into base weights
(`merge_and_unload`) and pushes a self-contained repo. Note: **Tokenformer adapters
can't be merged** (line 24) — they need the surgical hooks to exist, so this script
drops them.

## How to extend it

- **New base model**: set `model` in `values.yaml` (or `llm_name` in `train_args`)
  and restart. The serve-test sweep exists precisely to validate that a new base model
  loads under the vLLM fork — see `CONTEXT.md` on Serve-test vs Integration-test.
- **New adapter type**: add a branch in `ml/adapters/add_adapters_to_model.py:6`,
  write a `create_<type>_model.py` that freezes the base and unfreezes your params, and
  teach the inference side (`infra/cray_infra/adapters/model/`, the
  `find_model`/load path) how to load it. Watch the freeze/unfreeze name-matching.
- **Custom training loop / optimizer / loss**: edit `ml/` locally — it ships with each
  job. `get_optimizer` / `get_scheduler` / loss are all in
  `ml/cray_megatron/megatron/training_loop.py`.
- **Config knobs**: `infra/cray_infra/util/default_job_config.py` is the canonical list
  (`max_steps`, `learning_rate`, `gradient_accumulation_steps`, `warmup_steps`,
  `adapter_type`, `lora_config`, `distribution_strategy`, `timeout`, `dtype`, etc.).
  Pass overrides via `train_args`.
- **Dataset format**: JSONL `{"input": ..., "output": ...}` (see step 4).

## Footguns to watch

1. **Identical `train_args` + dataset → same job directory** (sha256 dedup).
   Re-submitting the exact same config silently returns the *existing* job
   (`launch_training_job.py:25`) rather than retraining. Change any arg (or the data) to
   force a new run.
2. **`lora_alpha` must be emitted in checkpoint metadata.** If missing, the
   inference/merge loader defaults to `2*rank`, which **silently 2×-mis-scales** the
   delta when you trained with `alpha == rank`. Handled by `build_adapter_metadata`
   (`training_loop.py:834`) — don't break it.
3. **Tokenformer ≠ mergeable.** You cannot fold a Tokenformer into a standalone HF
   model. If you need a portable single-file model on the Hub, train LoRA.
4. **PEFT key-prefix mismatch.** The trainer saves LoRA keys from *inside* the PEFT
   wrapper, missing the `base_model.model.` prefix. Both the resume path
   (`_load_trained_parameters`, `training_loop.py:756`) and the merge path
   (`_prefix_for_peft_load` in `merge_lora_and_push.py`) re-align them and **fail
   loudly** if zero tensors match — earlier versions silently folded nothing and
   published the unchanged base.
5. **`lm_head` only trains for <100M-param models.** For larger models it stays frozen
   (gradient-scale + adapter-load-size reasons). If you expect output-distribution
   shifts on a big model, this is why they may be muted.
6. **Packed-document attention.** Multiple docs get packed into one block; without the
   block-diagonal mask they'd attend across each other. The 4D mask is capped at
   seq_len 16384 (`training_loop.py:35`) — above that it's skipped with a warning and
   you get cross-document contamination.
7. **"Restart vllm" is a misnomer** (`CONTEXT.md`): vLLM runs in-process in the cray
   server; killing it restarts the *whole* server. Adapter changes never need this —
   only base-model changes do.

## Compute & memory to create an adapter

This section sizes a fine-tune for every model in `test/model_sweep/model-sweep.yaml`.
Worked for a single **NVIDIA RTX PRO 6000 Blackwell** (96 GB GDDR7, no NVLink); a
host with 4 of them where only 1 is available on-demand.

### Throughput assumption

For sustained bf16 training (FP32 accumulate — the stable path) assume
**~100 TFLOP/s effective** (peak ~250 TFLOPS dense × ~40% MFU; consumer/workstation
Blackwell halves the FP32-accumulate tensor rate vs datacenter parts). That's roughly
¼ of an H100. Compute is `C ≈ 6 · N_active · D`, with the trainer's default token
budget `D = max_steps(100) × grad_accum(4) × seq_len(4096) = 1.64M tokens`. This gives:

> **≈ 1.6 GPU-min per billion *active* params** for the default 100-step / 4096-token
> run (linear in `max_steps × seq_len`; ±25% — the throughput is an estimate).

LoRA and Tokenformer cost ~the same *FLOPs* (the frozen-base forward/backward
dominates); they differ in **memory**, below.

### Fine-tuning VRAM gate — reasoning

Training VRAM ≠ serving VRAM: the serve gate budgets weights + KV cache; the
fine-tune gate drops KV cache but adds **gradients + optimizer state on the trainable
params + activations**. Terms:

- **Frozen base weights** (dominant): bf16 = 2 B/param; MXFP4 (gpt-oss) ≈ 0.6 B; FP8
  (Qwen3-Next) ≈ 1 B. MoE loads **all** experts → use *total* params here, not active.
- **Optimizer + gradients**: AdamW keeps two moments + a gradient per *trainable*
  param (~8 B/trainable). This is where the two adapters split.
- **Activations + CUDA overhead**: ~6–10 GB at seq 4096, batch 1, assuming gradient
  checkpointing is on.

LoRA trains <1% of weights. Tokenformer, as implemented, unfreezes the **full
attention projections (q/k/v/o), all layernorms, and the token embeddings** on top of
its MLP adapters (`ml/adapters/create_tokenformer_model.py:51-67`) — roughly **25–40%
of the model** — so it carries optimizer state on a huge slice of the weights. Hence:

- **LoRA gate ≈ 2·P_total + 6 GB**
- **Tokenformer gate ≈ 4.5·P_total + 10 GB**

Quantized bases (gpt-oss MXFP4, Qwen3-Next FP8) are **LoRA-only**: Tokenformer would
unfreeze quantized attention weights. LoRA on a quantized frozen base is fine
(QLoRA-style).

### Sizing table

`1×96 GB?` reads **LoRA / Tokenformer**. GPU-time is wall-clock on one card for the
default run.

| Model | Total | Active | Serve gate | FT — LoRA | FT — Tokenformer | 1×96 GB? | RTX 6000 time |
|---|---|---|---|---|---|---|---|
| **Tiny-random stubs** (random weights — serve-test only) | | | | | | | |
| `tiny-random/gemma-4-dense` | ~few M | — | 2 GB | <8 | <8 | ✓ / ✓ | <1 s |
| `masint/tiny-random-llama` | ~few M | — | 2 GB | <8 | <8 | ✓ / ✓ | <1 s |
| `masint/tiny-random-qwen2-vl` | ~few M | — | 2 GB | <8 | <8 | ✓ / ✓ | <1 s |
| `yujiepan/qwen3-moe-tiny-random` | ~few M | — | 2 GB | <8 | <8 | ✓ / ✓ | <1 s |
| **Small** | | | | | | | |
| `google/gemma-3-270m-it` | 0.27B | 0.27B | 4 GB | 7 GB | 11 GB | ✓ / ✓ | ~25 s |
| `google/gemma-3-270m` | 0.27B | 0.27B | 4 GB | 7 GB | 11 GB | ✓ / ✓ | ~25 s |
| `Qwen/Qwen2-7B-Instruct` | 7B | 7B | 18 GB | 20 GB | 42 GB | ✓ / ✓ | ~11 min |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | 7B | 18 GB | 20 GB | 42 GB | ✓ / ✓ | ~11 min |
| `Snowflake/Arctic-Text2SQL-R1-7B` | 7B | 7B | 18 GB | 20 GB | 42 GB | ✓ / ✓ | ~11 min |
| `EssentialAI/rnj-1-instruct` | 8B | 8B | 20 GB | 22 GB | 46 GB | ✓ / ✓ | ~13 min |
| **Medium** | | | | | | | |
| `google/gemma-3-4b-it` | 4B | 4B | 12 GB | 14 GB | 28 GB | ✓ / ✓ | ~6 min |
| `openai/gpt-oss-20b` (MXFP4) | 21B | 3.6B | 24 GB | ~20 GB | n/a (4-bit) | ✓ / — | ~6 min |
| `Qwen/Qwen3-14B` | 14B | 14B | 32 GB | 34 GB | 73 GB | ✓ / ✓ | ~22 min |
| `google/gemma-3-27b-it` | 27B | 27B | 60 GB | 60 GB | 132 GB | ✓ / ✗ | ~43 min |
| `google/gemma-4-31B-it` | 31B | 31B | 68 GB | 68 GB | 149 GB | ✓ / ✗ | ~50 min |
| **Large** | | | | | | | |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | 32B | 40×2 GB | 70 GB | 154 GB | ✓ / ✗ | ~51 min |
| `Qwen/Qwen3-32B` | 32B | 32B | 40×2 GB | 70 GB | 154 GB | ✓ / ✗ | ~51 min |
| `Qwen/Qwen3.5-35B-A3B` | 35B | 3B | 40×2 GB | 76 GB | 167 GB | ✓ / ✗ | ~5 min |
| `Qwen/Qwen3-Next-80B-A3B-FP8` | 80B | 3B | 50×2 GB | ~86 GB | n/a (FP8) | ✓ (tight) / — | ~5 min |
| `openai/gpt-oss-120b` (MXFP4) | 117B | 5.1B | 40×4 GB | ~70 GB | n/a (4-bit) | ✓ / — | ~8 min |
| `Qwen/Qwen3.5-122B-A10B` | 122B | 10B | 75×2 GB | 250 GB | 559 GB | ✗ / ✗ | ~16 min |
| `nvidia/...Nemotron-3-Super-120B-A12B-BF16` | 120B | 12B | 70×4 GB | 246 GB | 550 GB | ✗ / ✗ | ~19 min |

### What fits the single on-demand 96 GB card

- **LoRA reaches up to ~32–35B dense**, plus the big *quantized* MoEs (gpt-oss-20b/120b,
  Qwen3-Next-80B-FP8) whose 4-bit/FP8 weights are small.
- **Tokenformer caps around ~14B** (Qwen3-14B at ~73 GB is the largest that fits) — it
  trains ~30% of the weights, so optimizer state blows past 96 GB on anything bigger.
- **Two models can't be touched on a single card at all:** `Qwen3.5-122B-A10B` and the
  BF16 `Nemotron-3-Super-120B` (~245 GB of frozen bf16 base). With all 4 cards (384 GB)
  via FSDP, **LoRA** on both is feasible; **Tokenformer on the 122B (~559 GB) won't fit
  even on four.**

### Caveats specific to these sizings

- **Enable `gradient_checkpointing`** for the big single-card LoRA runs (27B–35B,
  gpt-oss-120b). It's **off by default** (`default_job_config.py:27`); without it,
  activation memory at seq 4096 can push the 60–86 GB rows over 96 GB.
- **Quantized bases are LoRA-only** (see gate reasoning).
- **Multimodal models** (gemma-3/4, Qwen2-VL): the frozen vision tower still occupies
  memory (counted in total) but only the language path trains.
- The **`block_size` footgun** (#6 above) applies to both compute *and* memory here: at
  the default it tracks the model's 128K–1M context, multiplying GPU-time and activation
  memory by 32–256×. Set `max_token_block_size` explicitly.

## Worked example: fine-tune a LoRA on `yujiepan/qwen3-moe-tiny-random`

A runnable end-to-end LoRA fine-tune on the smallest sweep model. This model has
**random weights**, so it's a *pipeline smoke test* — success means the job reaches
`COMPLETED` and a `checkpoint_*.pt` with LoRA keys lands in the job dir, not that the
outputs are coherent. It fits in ~8 GB and runs in seconds.

Two things that trip people up, established by reading the submit path:

- **The base model to fine-tune comes from `train_args["llm_name"]`, not the
  `model_name` argument** to `train()`. `submit_training_job` ships only the dataset +
  `train_args`; `model_name` is dropped on the training path
  (`sdk/masint/engines/cray/submit_training_job.py`). It falls back to the server's
  `config["model"]` if `llm_name` is absent.
- **`adapter_type` defaults to `tokenformer`** — you must override it to `"lora"`
  (`default_job_config.py:66`), or you'll get a Tokenformer instead.

### 1. Bring up the server

Training runs as a Slurm job *inside* the deployment, so the stack must be up. The
served base model is independent of what you fine-tune — you do **not** need to serve
qwen3-moe to train a LoRA for it.

```bash
cd /path/to/scalarlm
./scalarlm up nvidia      # FastAPI on :8000 + in-process vLLM + Slurm
```

### 2. Submit the LoRA training job

```python
import scalarlm
scalarlm.api_url = "http://localhost:8000"

llm = scalarlm.SupermassiveIntelligence()

# Training data is JSONL of {"input", "output"} pairs; loss is masked to the
# `output` span (load_language_model_dataset.py:96).
dataset = [
    {"input": "What is 2+2?", "output": " 4"},
    {"input": "Capital of France?", "output": " Paris"},
    {"input": "2+3=", "output": " 5"},
] * 16   # repeat so packing yields a few blocks

train_args = {
    "llm_name": "yujiepan/qwen3-moe-tiny-random",  # selects the base (NOT model_name)
    "adapter_type": "lora",                         # override the tokenformer default
    "lora_config": {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": "all-linear",             # PEFT picks the linear/expert layers
    },
    "max_steps": 20,                                # tiny smoke run
    "steps_per_checkpoint": 10,
    "learning_rate": 3e-3,
    "gpus": 1,
    "dtype": "float32",                             # random-weight tiny model: fp32 is safest
    "max_token_block_size": 4096,                   # cap block size (see footgun #6)
}

status = llm.train(dataset, train_args=train_args)
print(status)        # contains job_directory (the sha256 job hash) + status
```

### 3. Monitor

```bash
./scalarlm llm-squeue      # is the Slurm job running?
./scalarlm llm-logs        # tail trainer logs ("Unfreezing LoRA parameters", loss lines)
./scalarlm llm-ls          # list models/jobs the server knows about
```

Or from Python: `llm.get_training_job(<job_dir>)` / `llm.list_models()`. Watch
`status.json` go `QUEUED → TRAINING → COMPLETED`. Loss is noisy/meaningless (random
weights) — that's expected.

### 4. Confirm the adapter artifact

```bash
ls /app/cray/jobs/<hash>/   # checkpoint_<step>.pt, status.json, config.yaml, dataset.jsonlines
```

The `.pt` holds only `requires_grad=True` tensors — the LoRA `lora_A/lora_B` deltas
plus `lora_alpha` in metadata (`training_loop.py:744,834`). That file *is* the adapter.
`register_megatron_models` will discover it (any `*.pt` in the jobs dir) and the
generate worker hot-loads it via `/v1/load_lora_adapter` (`create_generate_worker.py:251`).
The adapter's name is the job-directory hash.

### 5. (Optional) Run inference with the adapter

The adapter loads onto a **running base that matches `llm_name`**. To use it, the
server must be serving `yujiepan/qwen3-moe-tiny-random` — set `model:` in
`deployment/helm/scalarlm/values.yaml` (or uncomment it in `default_config.py:10` for
local) and restart (`./scalarlm up`). Then:

```python
out = llm.generate(prompts=["2+2="], model_name="<job_hash>")
```

### 6. (Optional) Export the LoRA as a portable PEFT repo

```bash
python -m adapters.merge_lora_and_push \
    --job-dir /app/cray/jobs/<hash> \
    --repo-id youruser/qwen3-moe-tiny-lora \
    --mode adapter --dry-run        # writes adapter_config.json + adapter_model.safetensors locally
```

Drop `--dry-run` (and add `--token`) to push to the Hub.

### Footguns specific to this run

1. **Use `train_args["llm_name"]`, not the `model_name` arg** — the latter is dropped
   on the training path.
2. **Override `adapter_type` to `"lora"`** — the default is `tokenformer`.
3. **Keep `max_token_block_size` modest.** qwen3-moe-tiny-random advertises a large
   context; the default `block_size` follows it, ballooning memory/time even on a tiny
   model.
4. **Random weights → meaningless loss/outputs.** Success = `COMPLETED` + a
   `checkpoint_*.pt` with LoRA keys.
5. **Re-submitting identical `train_args` + dataset returns the existing job** (sha256
   dedup, `launch_training_job.py:25`). Change a value to force a fresh run.
