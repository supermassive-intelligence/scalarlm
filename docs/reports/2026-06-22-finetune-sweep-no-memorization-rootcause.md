# Root cause: NO_MEMORIZATION across all real instruct models (fine-tune sweep)

Date: 2026-06-22 · Branch: `georgi/finetune-sweep` · Target: `cuda-spark` (DGX Spark)

## Summary

Every **real instruct model** in the `cuda-spark` sweep serves its trained LoRA
adapter but fails to *exactly* reproduce the 32-char golden string
(`NO_MEMORIZATION`) — 10 models: Qwen2-7B, Qwen2.5-3B/7B/14B, Mistral-7B-v0.3,
phi-4, Llama-3.2-1B/3B, Llama-3.1-8B, gemma-4-dense. Only tiny-random bases and
the smallest real models (Qwen2.5-0.5B/1.5B, gemma-3-270m, Qwen3-8B, rnj-1-8B)
reach `PASS`.

**Root cause is the LR schedule + step budget, not arch or serving.** The
60-step, no-warmup, `LinearLR`(full→0) schedule at peak LR `3e-3` cuts training
off *mid-descent*, after an early high-LR instability spike wastes the first
~25 steps. The loss never reaches the ~0 required to emit the golden string
verbatim. This is a pure **config** problem: the trainer's scheduler already
supports warmup (`JobConfig.warmup_steps`), but the sweep manifest never sets it.

## Evidence (training loss curves, from `jobs/<hash>/slurm-1.out` on the Spark)

All three Llamas, identical config (`max_steps=60`, `lr=0.003`, `warmup_steps=0`,
`gradient_accumulation_steps=4`):

| Model | Step 0 loss | Early spike (peak) | Final loss @ step 59 (LR→0) | Trend at cutoff |
|---|---|---|---|---|
| Llama-3.2-1B | 6.14 | 15.8 (step 9) | **1.73** | still falling ~0.06/step |
| Llama-3.2-3B | 6.72 | 9.6 (step 11) | **2.70** | still falling |
| Llama-3.1-8B | 6.79 | **48.8 (step 7)** | **3.28** | still falling |

Three facts the curves establish:

1. **Early instability, size-correlated.** With no warmup and a cold AdamW at
   full `3e-3`, loss explodes in the first ~10 steps — peak ~16 (1B) to ~49 (8B).
   `get_scheduler()`'s own docstring (training_loop.py) warns this is "a known
   instability source on large LoRA fine-tunes." Bigger model → bigger spike.
2. **Wasted budget.** The spike + recovery consumes ~the first 25 of 60 steps;
   smooth descent only begins around step 26.
3. **Cut off mid-descent.** `LinearLR(start_factor=1.0, end_factor=0.0,
   total_iters=max_steps)` drives the LR to *exactly* 0.000000 at step 59 while
   loss is still dropping steeply. Final loss tracks size inversely (1B closest
   to memorizing, 8B furthest) — none reach ~0.

The degenerate adapter outputs seen in the run report (`666666…`, `aa6ae6…`) are
exactly what a partially-converged model emits: biased toward the high-frequency
tokens of the target (the golden string is hex: lots of `6`,`a`,`f`) without
having learned the exact sequence.

## Why small models PASS and large ones don't

Same hyperparameters; fewer parameters memorize a single sequence in fewer
optimizer steps and tolerate the hot LR without a large spike. The boundary is a
budget boundary, not an architecture boundary — LoRA serving is already broad
(Qwen2/Qwen3/Llama/Gemma3/Mistral/Phi3 all serve).

## The fix (config-only)

`JobConfig` (infra/cray_infra/util/default_job_config.py) already exposes every
needed knob; the sweep manifest's `train_args_defaults` just need to set them:

- **`warmup_steps`** > 0 — turns the bare `LinearLR` into `SequentialLR(warmup →
  decay)`, ramping `lr/1000 → lr` to kill the cold-start spike.
- **`max_steps`** ↑ — so the post-warmup decay window is long enough for loss to
  reach ~0 (loss is still falling steeply at the current cutoff).
- (optional) **`lora_dropout`: 0** — 0.1 injects noise that slightly impedes
  exact memorization of a single pair.
- (optional) modest LR moderation, but warmup likely makes this unnecessary.

No change to the scheduler, trainer, or model code.

## Validation plan

Run a single fast currently-failing model (Llama-3.2-1B; ~130s train, closest at
loss 1.73) with the new params and confirm `MEMORIZED`, before rolling the manifest
defaults out to the full suite. Cost note: raising `max_steps` lengthens every
model's train phase, so the choice trades sweep wall-clock for memorization.
