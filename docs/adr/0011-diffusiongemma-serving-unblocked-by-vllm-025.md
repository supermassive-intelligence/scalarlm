# DiffusionGemma vLLM serving — unblocked by the 0.25 migration

**Supersedes [ADR 0008](0008-defer-diffusiongemma-vllm-serving.md).**

ADR 0008 deferred `google/diffusiongemma-26B-A4B-it` serving because the fork was
pinned to vLLM 0.19 and discrete-diffusion serving needs Model Runner V2 (`ModelState`,
bidirectional canvas attention, a custom per-step sampler). Its point 3 said to
revisit *"once the 0.19→0.23 rebase lands: prefer porting upstream's native
`diffusion_gemma.py` + Model Runner V2 and layering the fork's hybrid `.pt`-adapter
subsystem on top."*

That trigger has fired. The fork is now on **vLLM 0.25** (branch
`georgi/vllm-0.25-integration`), which ships native
`vllm/model_executor/models/diffusion_gemma.py` — `DiffusionGemmaModelState`,
`DiffusionSampler`, bidirectional canvas attention, all on Model Runner V2. The
serving infrastructure ADR 0008 was blocked on now exists in-tree.

## Decision

Enable the full train → serve → memorize closed loop for DiffusionGemma. Serving is
layered onto the native class exactly as ADR 0008 point 3 prescribed, and turns out
to need only a **small, well-scoped** set of changes — the native class already does
the heavy lifting.

### What the serving path needed (and didn't)

1. **`SupportsLoRA` on `DiffusionGemmaForConditionalGeneration`** (was
   `SupportsMultiModal, SupportsQuant, SupportsPP` only). `vllm/lora/model_manager.py`
   hard-gates `isinstance(model, SupportsLoRA)`. The class already carries
   `packed_modules_mapping` (qkv/gate_up) and `get_mm_mapping(language_model="model")`,
   so LoRA is confined to the shared Gemma4 backbone and the vision tower is excluded
   with no extra work. LoRA activation (`set_active_loras`) runs in the runner's
   `_prepare_inputs`, which is **ModelState-agnostic**, so the punica-wrapped backbone
   linears apply the adapter during canvas denoising without diffusion-specific hooks.

2. **Hybrid `.pt` decoder-prefix collapse.** The trainer adapts the DECODER only
   (ADR 0007) and exports keys under `model.decoder.layers.*`, but the vLLM port
   collapses the *tied* encoder/decoder text weights into a single `model.layers.*`
   backbone (its `hf_to_vllm_mapper` maps `model.decoder.` → `model.`). One line in
   `hybrid_adapter_manager._renormalize_lora_sd_for_model` (adding
   `model.decoder.layers.` to the prefix-undo set) maps the decoder LoRA onto that
   shared backbone. The layer-prefix detector already prefers the `model.layers.`
   backbone over the vision tower, so the Gemma-27b vision-prefix failure mode does
   not recur.

3. **Sampler config — no change needed.** DiffusionGemma requires an EntropyBound
   `sampler_config` (`diffusion_gemma.py:842`), read from the model's
   `generation_config`. The repo's `generation_config.json` already ships
   `sampler_config: {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1}`
   plus `t_min`/`t_max`/`confidence_threshold`, and cray's vLLM launch does not
   override `--generation-config` (defaults to `auto`), so it is honored as-is.

### Training

Unchanged from the ADR 0007 / design-spec scope: decoder-only LoRA, live
uniform-vocabulary canvas corruption, external cross-entropy over the canvas (the
forward returns `logits`, not a `.loss`). Implemented in the `cray-megatron` harness
(`is_diffusion` predicate, `load_model` branch, `load_diffusion_dataset` +
`diffusion_corruption`, `training_loop` dispatch, `resolve_target_modules` router
exclusion).

## Consequences / open risks (to settle on hardware — Phase C)

- **Memorize determinism.** The EntropyBound sampler is convergence-based and
  stochastic. The golden-hash exact-match check needs a reproducible greedy-equivalent
  decode. If it can't be made deterministic, fall back to a weaker verdict
  (`SERVED_NONDETERMINISTIC`: adapter changes output vs. baseline + training loss→0)
  rather than forcing a hash.
- **Tied-weight train/serve gap.** At serve the decoder LoRA applies to the *shared*
  encoder+decoder backbone, so it also perturbs the prompt-encoding pass — which ran on
  frozen base weights during training. Empirical impact on memorization is unknown; if
  decoder-only can't memorize through serving, the ADR 0007 revisit (encoder+decoder
  LoRA) is the next lever.
- **Canvas-vs-token LoRA mapping.** The runner builds its `LoRAMapping` from the
  autoregressive token layout; the diffusion ModelState's canvas layout may need the
  mapping to line up. Validate on GB10 (vast H200 fallback).

## Status

**Accepted.** Supersedes ADR 0008. See
`docs/superpowers/specs/2026-07-01-diffusiongemma-design.md` (training) and the
Phase-C validation run for the empirical serving result.
