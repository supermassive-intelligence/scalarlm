# LoRA key-prefix mapping: keep runtime decoder-prefix detection, supersede the static-rewrite alternative (#30)

The fork's adapter layer must map ScalarLM-trainer `.pt` LoRA keys
(`model.layers.<N>...`) onto the live vLLM module tree before activation. Two
incompatible approaches were implemented; this ADR records which one the
finetune-sweep branch carries and why. Context:
`docs/reports/lora-serving-noop-investigation.md` and ADR 0005 (the fork's
adapter subsystem is permanent fork-owned code).

## The two approaches

**Static rewrite (vllm-fork PR #30, `fix/normalize-lora-key-causal-lm-prefix`).**
Fix `normalize_lora_key` so the static rule is correct for text-only causal LMs
(keep the `model.` prefix for `model.layers.*`), and **delete** the runtime
machinery (`_detect_model_layers_prefix` / `_renormalize_lora_sd_for_model`).
Simpler and deterministic; no probing of the loaded model. Its risk is that one
static rule must be right for every architecture, including multimodal wrappers
where the decoder prefix is nested.

**Runtime detection (vllm-fork commit `49dd610d`, this branch).** Probe the
loaded model's `named_modules()`, detect the real text-decoder layers prefix,
and re-normalize keys to match. Adapts to whatever the actual module tree is.
Its cost is runtime complexity and a class of detection bugs — most recently the
Gemma4 failure where detection latched onto the **vision tower**
(`vision_tower.encoder.layers.`) and collided the decoder keys, which `49dd610d`
fixes by preferring the `model.layers.`-suffixed decoder prefix.

## Decision

1. **Keep the runtime decoder-prefix detection path; do not adopt #30's static
   rewrite + deletion.** The multimodal-wrapper case (Gemma4's vision/decoder
   ambiguity) is what a single static rule disambiguates least well, and the
   runtime path is the one GPU-validated across the sweep's multimodal models.
2. **PR #30 is superseded, not merged.** The approaches are mutually exclusive —
   #30 deletes exactly the path this branch corrects — so they cannot both land.
   Close #30 with a pointer to the runtime PR (`49dd610d`) and this ADR.
3. **The durable end-state is per-key resolution against `named_modules()`**,
   which removes the single-prefix assumption entirely and would make both the
   static rule and the runtime prefix-detection obsolete. Owner: Greg Diamos.
   `49dd610d` is the tactical unblock until that lands.

## Why

- **Why runtime over static:** the static rule's correctness argument held for
  text-only causal LMs but had to special-case multimodal wrappers; the sweep
  needs Gemma4/Qwen2-VL adapters to *serve*, and the vision/decoder collision is
  precisely the case the static rule reasons about least directly. Correcting
  the runtime path covered both families with one validated mechanism.
- **Why this is the load-bearing call:** per ADR 0005, adapter-layer fixes (key
  normalization) survive a vLLM rebase, whereas model-class patches get redone.
  Choosing the wrong normalization mechanism is therefore expensive to unwind —
  this decision should be signed off by the adapter-subsystem owner.
- **Why not just defer to per-key resolution now:** it is a larger change with a
  named owner and is not needed to unblock the sweep; shipping it under sweep
  pressure would couple two efforts.

## Status

**Proposed.** Documents the direction the `georgi/finetune-sweep` fork branch
took and recommends it as the near-term stance. The runtime-vs-static call and
the #30 closure need sign-off from Greg Diamos (adapter subsystem, ADR 0005)
before the static path is permanently retired.
