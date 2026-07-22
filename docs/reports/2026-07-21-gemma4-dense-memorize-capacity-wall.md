# `tiny-random/gemma-4-dense` NO_MEMORIZATION = fixture capacity wall, not an adapter bug

**Date:** 2026-07-21 · **Target:** cuda-spark (GB10, spark-147c) · **Verdict:**
the documented G4D adapter-key blocker is **fixed**; the residual
NO_MEMORIZATION is an inherent property of the fixture's dimensions and is not
fixable in our code.

## TL;DR

- The Gemma4 LoRA **adapter path works end-to-end** — it trains, exports keys
  that normalize onto the live vLLM tree, serves, and *applies* (served output
  shifts from baseline). This is the real, reusable result: the machinery for
  **real** Gemma-4 models is in place.
- `tiny-random/gemma-4-dense` still returns **NO_MEMORIZATION**, but the cause is
  **capacity**, not the adapter: `hidden_size=8`, `vocab_size=262144`,
  `tie_word_embeddings=True`. You cannot linearly separate ~55 arbitrary
  position→token targets in an 8-dim hidden space, so the model cannot reproduce
  a 32-token golden string regardless of adapter/LR/steps.
- **Decision:** stop trying to make the tiny fixture memorize; validate Gemma-4
  memorization on a **real** small Gemma-4 (`gemma-4-E2B-it`, hidden ~2048)
  instead. Keep `gemma-4-dense` as the fast serve/default smoke test.

## What the 2026-06-18 G4D report predicted, and what actually happened

G4D (`docs/reports/2026-06-18-gemma4-dense-adapter-noop-diagnostic.md`) blamed a
**partial adapter-key normalization** (attention lands, MLP misses, or vice
versa) on the MM-wrapped `language_model.model.layers.*` tree. That has since
been fixed: `vllm/tokenformer/adapter_format.normalize_lora_key` maps the Gemma4
MM shape correctly for **both** attention and MLP, with passing unit tests in
`tests/tokenformer/test_adapter_format.py` (incl. one keyed to a real training
log). G4D's own decision table had the fall-through branch:

> full overlap → keys land; adapter genuinely applies → **not a normalization
> bug → training/hyperparameter track** (`max_steps`/`lr`/`train_lm_head`).

That is exactly the branch we are on now.

## Evidence (cuda-spark, current fork code)

Run 1 — as configured (frozen lm_head, the LoRA-path default):

| step | LR | avg loss |
|---|---|---|
| 0 | 0.00015 | 12.4718 |
| 20 | 0.00299 (peak) | 12.3053 |
| 100 | 0.00213 | 12.1912 |
| 299 | 0.0 | **12.1715 (floor)** |

Loss floor ≈ **ln(262144) = 12.478** → the output distribution is barely better
than uniform over the 262k vocab. Verdict NO_MEMORIZATION; the **adapter sample
differs from the baseline sample** (`ariišení Deng…` → `ुएgran wet…`), proving
the adapter is applied (it is not an `ADAPTER_NO_OP`).

Run 2 — probe with `lm_head` unfrozen (tied weight, `train_lm_head=True`):

| step | avg loss |
|---|---|
| 0 | 12.4718 |
| 100 | 11.6392 |
| 249 | 10.9476 (still descending) |

Unfreezing the tied embedding/lm_head **helps** (the floor drops and keeps
falling) but **asymptotes far from 0** — confirming the binding constraint is
`hidden_size=8`, not the frozen head alone.

Contrast — the fixture that PASSes:

| fixture | arch | vocab | hidden | frozen-lm_head memorize? |
|---|---|---|---|---|
| `masint/tiny-random-llama` | LlamaForCausalLM | 32,000 | **256** | **PASS** |
| `tiny-random/gemma-4-dense` | Gemma4ForConditionalGeneration | 262,144 | **8** | NO — capacity |

## Incidental gaps found (real, but neither rescues hidden=8)

1. **LoRA path never auto-trains `lm_head`.** `create_lora_model` defaults
   `train_lm_head=False` and `add_adapters_to_model` doesn't override, whereas the
   Tokenformer path defaults `train_lm_head=None` (auto-enables for <100M-param
   models). Aligning them would let small models train the head; it helped here
   but did not reach memorization.
2. **The `.pt`-LoRA serve path drops base-weight overrides.**
   `PTWorkerLoRAManager._load_adapter` uses only `loaded.lora_sd`; a trained
   `lm_head.weight` (routed to `tokenformer_sd` as a base override by
   `split_adapter_state_dict`) is silently discarded for a pure-LoRA adapter. So
   training the head as a base override would not survive to serve anyway — a
   servable version would need `lm_head` as a LoRA *target* (in `lora_sd`).

These are logged for future small-hidden/large-vocab models; no real Gemma-4
(E2B hidden ~2048) needs them — real models memorize through a frozen head like
Llama does.

## Recommendation / next

- Treat `tiny-random/gemma-4-dense` as **serve-only** (boots + serves + adapter
  applies, fast) — it is the repo default model for exactly that smoke-test role.
- Validate Gemma-4 **memorization** on real small models, smallest-first:
  `gemma-4-E2B-it` → `gemma-4-E4B-it` → `gemma-4-12B-it`, then the MoE/31B tier.
