# Depth investigation: multimodal TRAIN_FAILED (Qwen2-VL, gemma-3-4b-it)

Date: 2026-06-22 · Branch: `georgi/finetune-sweep` · Reproduced locally on CPU
with tiny-random models (no GPU / no Spark needed).

## TL;DR

The two "multimodal" TRAIN_FAILEDs have **different and largely non-multimodal**
root causes, and **both models train text-only on CPU once the real bug is
addressed**. Neither needs an image pipeline for the memorization sweep.

| Model | Real root cause | Needs image handling? | Fix surface |
|---|---|---|---|
| Qwen2-VL-7B | Wrong **loader class** (`AutoModelForCausalLM` rejects `Qwen2VLConfig`) | No — text forward works | loader-class selection + scope LoRA to language tower |
| gemma-3-4b-it | Trainer's **4D doc-mask** breaks the model's internal loss (`IndexError`) — *not multimodal* | No | trainer mask handling for wrapper models |

## Qwen2-VL-7B (`Qwen2VLForConditionalGeneration`)

- **Load:** `AutoModelForCausalLM.from_pretrained` raises
  `Unrecognized configuration class Qwen2VLConfig for AutoModelForCausalLM`.
  The correct class is **`AutoModelForImageTextToText`** (loads
  `Qwen2VLForConditionalGeneration`). `AutoModelForVision2Seq` does not exist in
  the pinned transformers (5.x).
- **Text-only forward works.** With the model loaded, a pure-text batch
  (`input_ids`/`attention_mask`/`labels`, no `pixel_values`) runs forward +
  backward cleanly (loss ~11.9 on the tiny-random). No image inputs required for
  the memorization task.
- **Module layout:** language tower under `model.language_model.*`
  (leaf linears `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`),
  vision tower under `model.visual.*` (leaf linears `qkv,proj,fc1,fc2` — a
  **disjoint** name set). Targeting the 7 language-tower names lands **0 LoRA
  params in the vision tower**.
- **Caveat for the `all-linear` resolver** (added 2026-06-22 for the MoE fix):
  it collects Linear leaf-names *model-wide*, so on a multimodal model it would
  also pick up `qkv,proj,fc1,fc2` and adapt the vision encoder. Multimodal needs
  the resolver scoped to the language submodule (e.g. resolve within
  `model.language_model`).
- **`mm_token_type_ids`:** the trainer's `_is_multimodal` branch adds this on
  every forward. Qwen2-VL does not use it; need to confirm its forward tolerates
  the kwarg (or gate the injection to models that declare it).

**Effort:** moderate, all unit-testable locally; GPU only for final confirmation.

## gemma-3-4b-it (`Gemma3ForConditionalGeneration`) — NOT a multimodal bug

- **Loads fine** via the current `AutoModelForCausalLM` path (returns
  `Gemma3ForConditionalGeneration`). Plain text forward: **OK**. With
  `mm_token_type_ids`: **OK**.
- **The break is the trainer's 4D block-diagonal document mask.** Isolated by
  adding trainer inputs one at a time:

  | forward inputs | result |
  |---|---|
  | 2D `attention_mask` | OK (loss 12.49) |
  | + `mm_token_type_ids` | OK |
  | + **4D doc-mask + `position_ids`** | **IndexError** |

- **Exact failing line** (transformers `modeling_gemma3.py:985`, in the model's
  own loss computation):
  ```python
  shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
  ```
  `Gemma3ForConditionalGeneration.forward` computes the loss internally and masks
  `shift_logits` (3D `[B,S,V]`) with `shift_attention_mask` sliced from a **2D**
  `attention_mask`. The trainer's 4D `[B,1,S,S]` mask makes that index 4D →
  "too many indices for tensor of dimension 3".
- **Why text-only Gemma3 passed:** `gemma-3-270m`/`-it` are `Gemma3ForCausalLM`,
  whose loss path doesn't index logits by the attention_mask, so the 4D mask is
  harmless. Only the **ConditionalGeneration wrapper** has the 2D-mask loss code.
- **For the memorization sweep the 4D mask is a no-op:** the dataset is one
  document repeated, so the packed block holds a single doc and the
  block-diagonal mask equals a plain causal mask. Skipping the 4D mask for
  wrapper models costs nothing here.

**Effort:** the fix lives in the trainer's mask handling
(`training_step_accumulate` in `ml/cray_megatron/megatron/training_loop.py`), not
in multimodal plumbing — e.g. don't pass a 4D `attention_mask` to a model that
masks labels by a 2D mask internally (detectable), or compute the loss ourselves
instead of relying on the model's internal label path. This is a
general-correctness concern: any future `...ForConditionalGeneration` with the
same internal-loss pattern would hit it.

## What "real support" actually requires

Smaller than the handoff implied — no image pipeline for the sweep:

1. **Loader-class selection** for vision configs (`AutoModelForImageTextToText`
   when `config.vision_config` is present and `AutoModelForCausalLM` rejects it).
2. **Scope the `all-linear` LoRA resolver to the language tower** for multimodal
   models (resolve within `model.language_model`).
3. **Trainer 4D-mask handling** for `...ForConditionalGeneration` wrappers whose
   internal loss assumes a 2D attention_mask (gemma-3-4b-it; likely others).
4. Confirm `mm_token_type_ids` injection is correct/harmless per arch.
5. GPU validation on the Spark (deferred; rollout currently owns the card).

## Recommendation

Two clean, independently-shippable pieces, both unit-testable now:
- **gemma-3-4b-it** is the better first target — its real fix (trainer 4D-mask
  handling) is a general-correctness improvement that also unblocks any future
  conditional-generation model, and it needs no loader/LoRA-scoping changes.
- **Qwen2-VL** is mechanical but pulls in loader selection + LoRA scoping +
  `mm_token_type_ids` verification.

If the goal is a green sweep *today*, gating both (`multimodal: true` → SKIP with
this report linked) remains valid; the above is what to build when support is
wanted, and it's less than "full multimodal training."

## Update (2026-06-22): implemented (pending GPU validation)

All four pieces built and unit-tested locally on CPU with tiny-random models:

1. **Loader-class selection** — `load_model.materialize_model` now picks
   `AutoModelForImageTextToText` when `is_multimodal(model_config)`, else
   `AutoModelForCausalLM`. Loads both Qwen2-VL and Gemma3 wrappers.
2. **Language-tower LoRA scoping** — `adapters/resolve_target_modules.py`: for a
   multimodal model, `all-linear` resolves to the *full module paths* of Linear
   layers under `get_decoder()`, so PEFT's suffix-name matching can't leak into a
   vision tower that reuses leaf names (verified 0 vision-tower LoRA params on
   both models; 14 language-tower).
3. **4D-mask handling** — `megatron/doc_mask.py` `doc_mask_decision()` returns
   `SKIP_MULTIMODAL` for vision-config wrappers; `training_loop` keeps the 2D
   mask. Gemma3 forward + backward now succeeds.
4. **`mm_token_type_ids`** — already tolerated by both (Qwen2-VL has it in its
   forward signature; Gemma3 absorbs it via `**kwargs`); no change needed.

End-to-end CPU training step (load → resolve → LoRA → masked forward → backward)
passes for both `Gemma3ForConditionalGeneration` and `Qwen2VLForConditionalGeneration`.
Remaining: validate on the Spark (GB10) once the memorization rollout frees the
GPU — `./ml` is live-mounted, so deployment is an scp + per-model restart, no
image rebuild.
