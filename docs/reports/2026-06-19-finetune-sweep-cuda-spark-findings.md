# Fine-tune sweep on the DGX Spark — findings (2026-06-19)

Empirical findings from standing up and running the `cuda-spark` fine-tune sweep
(LoRA train → hot-load → serve → memorization check) on the DGX Spark
(`spark-147c`, GB10, 128 GiB unified memory, aarch64 + Blackwell sm 12.0).
Branch `georgi/finetune-sweep`. Supersedes the (pre-empirical) "ScalarLM support"
section of `docs/self-served-llms-scratchpad.md`.

## Headline

1. **LoRA serving is far broader than the documented 3-class allowlist.** The
   `scalarlm_{llama,gemma,qwen2}` registry in
   `infra/cray_infra/adapters/model/models.py` gates only the **tokenformer**
   adapter path. The sweep uses `adapter_type: lora`, which rides vLLM-native LoRA
   + the fork's prefix-aware key normalizer — and that serves **Qwen2, Qwen3,
   Llama, Gemma 3, Mistral, and Phi-3** adapters. Qwen3-8B even memorized (PASS).
   The scratchpad's "only Llama/Qwen2/Gemma-v1 servable" is wrong for LoRA.
2. **The arch is not the limiter.** The real limiters are (a) co-located GPU
   memory (large models), (b) training-side LoRA surgery for multimodal/MoE, and
   (c) memorization hyperparameters for big models.

## Final run — `finetune.cuda-spark.20260619-173911` (22 models)

| Outcome | n | Models |
|---|---|---|
| PASS | 7 | Qwen2.5-0.5B, Qwen2.5-1.5B, tiny-random-llama, gemma-3-270m(+it), rnj-1-instruct (8B), **Qwen3-8B** |
| NO_MEMORIZATION | 10 | gemma-4-dense, Qwen2-7B, Qwen2.5-3B/7B/14B, Mistral-7B-v0.3, phi-4, **Llama-3.2-1B/3B, Llama-3.1-8B** |
| RESTART_FAILED | 2 | tiny-random-qwen2-vl, Qwen2.5-32B |
| TRAIN_FAILED | 3 | qwen3-moe-tiny, Qwen2-VL-7B, gemma-3-4b-it |

NO_MEMORIZATION is a non-failing outcome.

**Llama update (run `finetune.cuda-spark.20260619-200802`):** after the token's
account accepted the Llama 3.1/3.2 licenses, the 3 previously-gated Llamas ran
clean end-to-end — all **NO_MEMORIZATION**, no crashes, no TRAIN_FAILED
(1B: restart 178.5s / train 130.4s / serve 11.2s; 3B: 230.7 / 255.7 / 13.7;
8B: 368.8 / 546.3 / 15.8). This confirms dense Llama serves + trains LoRA and
moves them from SKIPPED into the same hyperparameter-gap bucket as the other
real instruct models. No gated SKIPs remain.

## Spark serving config (why)

`cuda-spark` sets `SCALARLM_VLLM_ARGS='--enforce-eager
--gpu-memory-utilization=0.3 --max-model-len=4096'`:

- **`--enforce-eager`** — skip CUDA-graph capture; pure startup cost for a
  memorization sweep (cuts per-model init).
- **`--max-model-len=4096`** — the memorization prompt is tiny. At the default
  `auto` (32768) the vLLM memory-profiling forward pass blows up activation
  memory and drives `num_gpu_blocks → 0`, so generation hangs ("generate did not
  complete in time"). This was the original 7B+ failure; capping max-len fixes it.
- **`--gpu-memory-utilization=0.3`** (~38 GiB) — loads up to ~14B bf16 weights
  while leaving room for co-located training. `0.15` was too low (a 7B's
  weights + profiling overran it). **32B (~64 GiB bf16) does not fit** the
  co-located budget and crash-loops at load — it needs phase-scaling (free vLLM
  during training, like the k8s path). 32B is dropped from the suite.

**Co-location tension (Spark-specific):** on the non-phased Compose path vLLM and
training share the 128 GiB pool simultaneously, so serving mem-util trades off
against training headroom. This caps practical serving at ~14B.

## Runner robustness fixes landed this session

- Unified-memory VRAM gate: `nvidia-smi memory.free` is `[N/A]` on the GB10; the
  gate now treats that like k8s (no discrete pool) instead of crashing.
- `generate()` request timeout configurable + `generate_with_retry` across the
  ~146 s cold-start window (baseline generate previously one-shot at 30 s).
- `poll_training` catches `OSError` (transient `ConnectionResetError` mid-train).
- Compose crash fail-fast: `wait_for_model_served` returns `crashed` when the
  container RestartCount climbs (`restart: unless-stopped` re-loops a crashed
  vLLM) instead of burning the full restart timeout.
- HF gated-repo handling: preflight 401 → clean `SKIPPED`; `HF_TOKEN` wired
  through the compose env passthrough (works for Mistral + Gemma; Llama needs the
  account to accept the license).
- Per-model `train_args` override (e.g. `dtype: bfloat16` for the flagship).
- Jobs dir bind-mounted to the host so TRAIN_FAILED slurm logs persist.

## Failure taxonomy

### 1. Multimodal serve crash — `tiny-random-qwen2-vl` (RESTART_FAILED)
Root-caused. The model's `config.json` declares **no `tie_word_embeddings`**,
which defaults to `True` in transformers, so the checkpoint omits
`lm_head.weight`. On the vLLM side the outer `Qwen2VLConfig.tie_word_embeddings`
resolves falsy, so (a) the inner `Qwen2ForCausalLM` builds a *separate, untied*
`lm_head` and (b) the skip at `qwen2_vl.py:1440` doesn't fire →
`ValueError: Following weights were not initialized: {'language_model.lm_head.weight'}`
→ EngineCore crash-loop. Fix: read the tie flag from `config.get_text_config()`
and ensure the inner model ties `lm_head → embed_tokens` (the real fix; the
existing skip only suppresses the error and leaves lm_head uninitialized).
**Note:** the real `Qwen2-VL-7B` did *not* crash at serve this run (baseline
served), so this is specific to the tiny-random model's missing config — the 7B
fails at *training* instead (below).

### 2. TRAIN_FAILED — three distinct causes (confirmed)
All three serve their baseline but fail *training*. Captured via the jobs
bind-mount (slurm-1.out per job). They are **not** one bug:

- **Qwen2-VL-7B** — `ValueError: Unrecognized configuration class Qwen2VLConfig
  for this kind of AutoModel: AutoModelForCausalLM`. Fails at **load**:
  `ml/cray_megatron/models/load_model.py` uses `AutoModelForCausalLM`, which does
  not accept multimodal configs (Qwen2-VL needs `AutoModelForVision2Seq` /
  image-text-to-text).
- **gemma-3-4b-it** — LoRA injects fine, then the training **forward** crashes
  `IndexError: too many indices for tensor of dimension 3` (`peft/peft_model.py:945`).
  Another multimodal mismatch: the text training loop feeds 2-D token batches to a
  model whose forward expects multimodal inputs.
- **qwen3-moe-tiny** — `ValueError: Target modules {'-','l','n','r','i','a','e'}
  not found in the base model`. PEFT received `target_modules` as the **set of
  characters** of the string `"all-linear"` for this model (the string wasn't
  resolved as PEFT's special all-linear value), so injection finds nothing.

**Theme:** the training path is built for **text causal LMs**. Multimodal models
(qwen2-vl, gemma-3) break it (wrong AutoModel class at load; wrong input rank at
forward); the MoE hit a separate `target_modules` resolution edge case.

**Fixes:** (a) for multimodal, detect the config and load via the right AutoModel
class + train the language tower only (or exclude multimodal from the sweep);
(b) for the MoE, pass explicit `target_modules` (e.g. the attn/MLP proj names)
instead of `"all-linear"`, or pin a PEFT version that special-cases it.

### 3. NO_MEMORIZATION — hyperparameter gap, not arch
Real instruct models (Qwen2-7B, Qwen2.5-3B/7B/14B, Mistral-7B, phi-4,
Llama-3.2-1B/3B, Llama-3.1-8B) serve the
adapter but don't *exactly* memorize the golden string in 60 steps. Several got
very close — e.g. Qwen2.5-3B produced `aaaf6f7ae83df6e653e6dda6dda6dafcc` vs the
golden `aaaf6f8ae738dfc6577e63dda6daf9cc`. Small base models (0.5B/1.5B/8B)
memorize exactly; bigger/instruct models need more steps or higher LR. This is a
training-budget gap, not a serving/arch problem.

## Open items

- Add multimodal training support (right AutoModel class + language-tower-only
  LoRA), or gate multimodal models out of the sweep.
- Fix the MoE `target_modules` resolution (explicit module names vs `"all-linear"`).
- Optionally raise `max_steps`/LR per-model so big instruct models reach PASS.
- 32B serving needs a phase-scaled Spark path if it's ever wanted.
- Llama 1B/3B/8B confirmed serving + training (NO_MEMORIZATION); license accepted,
  no longer gated.
