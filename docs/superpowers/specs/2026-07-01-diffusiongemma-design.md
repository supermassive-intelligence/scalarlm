# DiffusionGemma‑26B‑A4B‑it Design Specification

**Date:** 2026‑07‑01
**Status:** Final (training-only scope; supersedes the original vLLM-integration draft)

## Overview

This design adds **LoRA fine-tuning** support for `google/diffusiongemma-26B-A4B-it`
to the `cray-megatron` training harness. DiffusionGemma is a discrete-diffusion,
**encoder-decoder** MoE language model (`DiffusionGemmaForBlockDiffusion`,
`transformers >= 5.11`): a causal encoder reads the prompt into a KV cache, and
a bidirectional decoder iteratively denoises a fixed-size **canvas** (block of
response tokens) that cross-attends to it. It is also multimodal-capable
(image/video input via a Gemma4 vision tower), though this spec only exercises
text-only training.

**Serving is explicitly out of scope** — see ADR 0008. This spec covers
training only: a model that trains but isn't served by this repo yet.

---

## 1. Architecture & Model Loading

- **Model class:** `DiffusionGemmaForBlockDiffusion` (`transformers.models.diffusion_gemma`).
  Not registered under `AutoModelForCausalLM` or `AutoModelForImageTextToText` —
  every upstream usage example imports it directly. `load_model.py`'s
  `materialize_model()` needs a new branch, checked *before* the existing
  multimodal/causal fork (which would otherwise misroute it to
  `AutoModelForImageTextToText`, since `DiffusionGemmaConfig` also carries a
  `vision_config`):

  ```python
  def is_diffusion(model_config) -> bool:
      return getattr(model_config, "model_type", None) == "diffusion_gemma"
  ```

  When `is_diffusion(model_config)` is true, import and call
  `DiffusionGemmaForBlockDiffusion.from_pretrained(...)` directly instead of
  going through either `AutoModel*` class. `torch_dtype="auto"`,
  `low_cpu_mem_usage=True`, and the existing `device_map={"": device}` on-GPU
  materialization path (avoiding the CPU→GPU peak-memory doubling) apply
  unchanged — nothing about diffusion loading needs a different memory
  strategy.
- **`is_diffusion()` lives in `cray_megatron/megatron/doc_mask.py`**, alongside
  `is_multimodal()` — both are model-category predicates consumed the same way
  (config introspection, no job-config flag), so they belong in one place.
- **Encoder/decoder weight tying:** `DiffusionGemmaModel._tied_weights_keys`
  ties the encoder's and decoder's text-tower base weights (same `Parameter`
  tensors, different `nn.Module` wrapper objects — see ADR 0007). This is
  handled automatically by `from_pretrained`; no special loading code needed.
- **Environment:** bump the Docker image's `transformers` floor to `>= 5.11`
  (the version DiffusionGemma was added in). `peft`/`accelerate` versions are
  unaffected.

---

## 2. LoRA‑Based Tuning (Training)

- **Scope: decoder-only LoRA.** See **ADR 0007**. PEFT wraps specific
  `nn.Linear` *objects*; the encoder and decoder are separate objects despite
  tied base weights, so adapting the decoder does not adapt the encoder. The
  decoder produces the loss and output logits directly, so this is the
  natural default — validate empirically before ever taking on the doubled
  checkpoint/serving complexity of adapting both.
- **Target-module resolution requires zero new code path.**
  `DiffusionGemmaForBlockDiffusion` has no `get_decoder()` override; the
  generic `PreTrainedModel.get_decoder()` fallback (`base_model_prefix =
  "model"` → `self.model.get_decoder()` → `self.model.decoder`) already
  resolves correctly. `resolve_target_modules.py`'s existing multimodal
  branch (`_language_decoder()`, gated on `vision_config` presence — true for
  DiffusionGemma) reaches the decoder's `nn.Linear` modules with no
  DiffusionGemma-specific code.
- **Required fix: router exclusion in `_moe_servable_linear_paths`.**
  `DiffusionGemmaTextRouter`'s routing projection is `self.router.proj` (an
  `nn.Linear`), not `self.gate` (the Qwen3-MoE convention the current
  exclusion check assumes: `module_name.endswith(".gate") or module_name ==
  "gate"`). Without a fix, `resolve_target_modules("all-linear", ...)` would
  silently LoRA-adapt the router, contradicting the reference NeMo AutoModel
  recipe's `freeze_router: true` (adapting routing logic mid-fine-tune
  destabilizes expert selection). Fix:
  ```python
  if ".experts" in module_name:
      continue
  if ".router." in module_name or module_name.endswith(".gate") or module_name == "gate":
      continue
  ```
  Add a unit test in `test/unit/test_resolve_target_modules.py` using a fake
  module tree shaped like DiffusionGemma's decoder (`layers.0.router.proj`,
  `layers.0.self_attn.{q,k,v,o}_proj`, `layers.0.mlp.{gate,up,down}_proj`,
  `layers.0.experts` as a non-`nn.Linear` parameter holder — `DiffusionGemmaTextExperts`
  stores `gate_up_proj`/`down_proj` as raw `nn.Parameter` 3D tensors, so PEFT's
  `nn.Linear`-only targeting can never touch the experts regardless of the
  exclusion logic; the router is the only fix actually needed).
- **`create_lora_model.py` and `add_adapters_to_model.py` need no changes.**
  Both are fully generic over `resolve_target_modules()`'s output; the
  `train_lm_head` heuristic (< 100M total params) stays `False` for a 26B
  model automatically.
- **Reference hyperparameters** (from the NVIDIA NeMo AutoModel
  `diffusion_gemma_lora.yaml` recipe — the only known working LoRA config for
  this model): `lora_r: 16`, `lora_alpha: 32`, target modules = attention
  (`q_proj`/`k_proj`/`v_proj`/`o_proj`) + dense MLP
  (`gate_proj`/`up_proj`/`down_proj`) — exactly what decoder-only + the router
  fix already produces via `"all-linear"` resolution.

---

## 3. Training Data & Loss

- **Dataset format: reuse the existing `{"input": ..., "output": ...}` JSONL
  shape** (`load_language_model_dataset.py`'s format; the finetune sweep's own
  dataset generator already produces this). Single-turn by construction, so
  none of the NeMo recipe's multi-turn history masking is needed.
- **New loader: `load_diffusion_dataset.py`, no packing.** Unlike the causal-LM
  loader (which concatenates multiple documents into a packed block with
  `document_ids`/`position_ids`), DiffusionGemma trains one example per canvas
  slot: tokenize `input` → encoder `input_ids`/`attention_mask`; tokenize
  `output` → clean decoder canvas, padded or truncated to `canvas_length`
  (256). **Oversized outputs** (more than `canvas_length` tokens) are
  truncated to the first `canvas_length` tokens with a `logger.warning` —
  matching the existing precedent in `get_pack_function`'s budget-based
  document subsampling, not a hard validation error.
- **Corruption happens live in `training_step_accumulate`, not in the data
  loader.** The dataset yields *clean* canvas tokens; baking corruption into a
  `.map()` transform (like packing does) would fix the same corruption pattern
  for a whole epoch instead of resampling fresh every step. Per NeMo's
  `DiffusionGemmaSFTRecipe`:
  - Sample `t ~ U(eps, 1)` per example (`eps: 0.001`, matching the reference
    recipe's default).
  - Independently replace each supervised canvas position with a **uniform
    random vocabulary token** with probability `t` — there is no `[MASK]`
    token; this is uniform-state diffusion, not masked-language-model-style
    corruption.
  - `labels`: the clean token at every supervised canvas position, `-100`
    elsewhere (padding beyond the real output length).
  - This produces `decoder_input_ids` (corrupted canvas) and `labels`
    (canvas-shaped, matching the `(batch, canvas_length)` shape
    `DiffusionGemmaForBlockDiffusion.forward()` expects).
- **Self-conditioning: deferred.** NeMo's recipe mixes the decoder's own
  previous-step prediction back in via `self_conditioning_logits`/
  `self_conditioning_mask` "per example during training" (no stated
  probability), which requires an extra forward pass. Given the sweep's use
  case is memorizing a short deterministic `golden_prompt → expected_output`
  pair (not general generation coherence, where self-conditioning matters
  most), v1 always passes `self_conditioning_logits=None`,
  `self_conditioning_mask=None` — a valid, documented input state. Revisit
  only if empirical memorization runs show it's needed.
- **`mm_token_type_ids` defensive inclusion.** `DiffusionGemmaEncoderModel`'s
  docstring describes it as "very similar to Gemma4Model" — the same lineage
  whose `...ForConditionalGeneration` wrappers crash without
  `mm_token_type_ids` on text-only batches despite it being documented as
  optional (see `doc_mask.py`'s existing Gemma4 comment). The diffusion
  forward path defensively includes
  `mm_token_type_ids=torch.zeros_like(input_ids)` on the encoder side rather
  than trusting "optional" to mean "safe to omit."
- **Router stays frozen for free.** NeMo's recipe sets `freeze_router: true`
  explicitly; here it falls out automatically — since the router-exclusion fix
  means `resolve_target_modules` never selects `router.proj` as a LoRA target,
  PEFT's standard base-freezing (`requires_grad=False` on everything not
  targeted) leaves it frozen with no extra flag.

---

## 4. Training Loop Integration

- **Dispatch: inline branch in `training_loop.py`, matching existing
  precedent style** (`is_multimodal`/`doc_mask_decision`-style predicate +
  `if`), **not** a delegated strategy-object abstraction. `is_diffusion
  (model_config)` gates:
  - which `DataLoader` class `training_loop()` constructs
    (`load_diffusion_dataset.py`'s loader vs. the existing packed-sequence one), and
  - which `forward_kwargs` `training_step_accumulate()` builds (encoder
    `input_ids`/`attention_mask` + corrupted `decoder_input_ids` +
    canvas-shaped `labels` + `mm_token_type_ids`, vs. the existing causal-LM
    `input_ids`/`attention_mask`/`labels` construction with document masking).
  - **Noted for future rework:** this diverges from what a
    strategy/delegation pattern would look like for a codebase with more than
    two training paradigms: the causal-LM branch body is ~100 lines of
    doc-masking logic, and the diffusion branch body will be comparably
    sized. Chosen for now to match the codebase's existing predicate+inline-`if`
    convention rather than introduce a new abstraction; revisit if a third
    training paradigm is added.
- **Checkpointing, optimizer, scheduler, NaN handling, timeout, and status
  reporting are all architecture-agnostic and require no changes** — they
  operate on `requires_grad` and loss values, not on model internals.
- **Job config additions** (under a `diffusion:` block, matching the
  `lora_config` block's existing nesting convention):
  ```yaml
  model_type: diffusion       # gates nothing internally (is_diffusion() auto-detects
                               # from the HF config) — documents intent for job authors
  diffusion:
    canvas_length: 256        # decoder block size; also the loader's pad/truncate target
    eps: 0.001                # minimum corruption level for t ~ U(eps, 1)
  lora:
    lora_r: 16                # NeMo reference default
    lora_alpha: 32
    lora_dropout: 0.0
  ```
  `max_denoising_steps` (from the original draft) is **removed** — it's an
  inference-time sampler parameter (`EntropyBoundSampler`/
  `StableAndConfidentStoppingCriteria`), not a training concern, and serving
  is out of scope (ADR 0008).

---

## 5. Testing

- **`test/unit/test_resolve_target_modules.py`:** router-exclusion fix, using
  a fake module tree shaped like DiffusionGemma's decoder (see Section 2).
- **New unit tests for `is_diffusion()`** in `doc_mask.py`'s test file:
  config with `model_type == "diffusion_gemma"` → `True`; other configs
  (including multimodal non-diffusion ones) → `False`.
- **New unit tests for `load_diffusion_dataset.py`:** canvas padding/truncation
  at the `canvas_length` boundary (including the oversized-output truncation
  path), clean-labels construction, encoder/decoder field shapes.
- **New unit tests for the corruption step:** given a fixed RNG seed, verify
  `t ~ U(eps, 1)` sampling bounds, that corrupted positions are replaced with
  in-vocabulary random tokens (not `[MASK]`), and that `labels` matches the
  pre-corruption clean tokens at every supervised position.
- **Integration:** a LoRA training job against a tiny/mocked DiffusionGemma
  config (mirroring how other models are smoke-tested in this repo) verifying
  the checkpoint contains only decoder LoRA `lora_A`/`lora_B` keys — no
  encoder keys, no router keys, no expert keys.
- **No serving/vLLM tests** — out of scope per ADR 0008. The finetune sweep
  excludes DiffusionGemma from the closed loop (train → serve → memorize)
  entirely, same as Tokenformer is excluded per ADR 0004.

---

## 6. Out of Scope (see ADR 0008)

- vLLM serving (blocked on the deferred 0.19→0.23 rebase / Model Runner V2 —
  ADR 0005).
- Self-conditioning during training (deferred to future work).
- Encoder-side LoRA adaptation (ADR 0007; revisit only if decoder-only
  training fails to converge on the memorization check once serving unblocks
  evaluation).
- Multimodal (image/video) training data — text-only `input`/`output` pairs
  only.

---

## References

- ADR 0005 — vLLM fork adapter layer and upgrade stance (0.19 base, defers
  the Model Runner V2 rebase).
- ADR 0007 — DiffusionGemma decoder-only LoRA scope.
- ADR 0008 — Defer DiffusionGemma vLLM serving.
- NVIDIA NeMo AutoModel `diffusion_gemma_lora.yaml` / `diffusiongemma.md` —
  reference training recipe (corruption schedule, LoRA target modules,
  frozen-router requirement).
- `transformers` v5.12.0 `DiffusionGemmaForBlockDiffusion` docs and
  `modeling_diffusion_gemma.py` source — forward signature, encoder/decoder
  weight tying.

---

*Generated with Claude Code.*
