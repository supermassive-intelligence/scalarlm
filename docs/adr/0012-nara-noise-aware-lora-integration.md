# NaRA (noise-aware LoRA) for DiffusionGemma: training-side prototype, serving deferred

Our DiffusionGemma decoder-only LoRA ([ADR 0007](0007-diffusiongemma-decoder-only-lora.md))
trains cleanly to loss ~1e-5 but tops out at **28/32** on the golden-hash memorize contract,
with the first ~3 canvas positions wrong and the good basin unstable across reruns (the
`cccc` collapse). See `docs/reports/2026-07-16-diffusiongemma-validation-runs.md`.

While surveying other people's diffusion-LLM experiments (`docs/references/`), the one
credible **training-side** lever specific to diffusion LLMs was NaRA — *Noise-aware LoRA*
(arXiv 2605.29716, `docs/references/nara-noise-aware-lora.md`). Its thesis: plain LoRA learns
a **static** update `dW = B·A` applied across the entire denoising trajectory, which is
"structurally mismatched" to dLLMs because the input distribution shifts with the noise
level; empirically it "yields negligible improvement at high noise levels." NaRA makes the
update **noise-conditioned**, `dW(λ) = B·C(λ)·A`, where `C(λ)` is an `r×r` core matrix from a
small shared hypernetwork on the noise level `λ`. Notably, NaRA's own paper benchmarks
*DiffusionGemma-26B* and corroborates our ADR-0007 architecture (freeze router+experts, adapt
attention + dense/shared MLP).

Our corruption already samples exactly the `λ` NaRA needs: `corrupt_canvas` draws a per-example
`t ~ U(eps, 1)` and corrupts each supervised canvas position with probability `t`. That `t`
**is** the noise level.

## Decision

**Land NaRA as an opt-in training-side prototype, gated by `diffusion.nara.enabled`
(default off). Do NOT yet solve NaRA serving — defer it, as ADR 0008 did for DiffusionGemma
serving, until training evidence justifies the serving cost.**

### Training integration (this ADR, implemented)

1. **Config** — `NaraConfig` nested under `DiffusionConfig` (`infra/.../default_job_config.py`),
   declared on the Pydantic model so `get_job_config` doesn't silently drop it (the
   `trust_remote_code` footgun). `enabled` is the flag; `c_scale`/`fnn_hidden_*`/
   `noise_embed_dim`/`fourier_scale` are NaRA-only knobs. **Rank/alpha/dropout are inherited
   from `lora_config`** — no duplication.
2. **Noise-level source** — `corrupt_canvas(..., return_noise_level=True)` now also returns
   `t` (backward-compatible; default still returns only `decoder_input_ids`).
3. **Adapter** — `ml/adapters/nara_prototype.py`: `NaRALinear` (base + `((dropout(x)@Aᵀ)@Ceff)@Bᵀ`),
   `NaRAMapper` (shared hypernetwork, **zero-init last layer** ⇒ `C=0` ⇒ `Ceff=I` ⇒ **byte-identical
   to plain LoRA at init**), `NaRAContext` (owns the mapper; `set_noise_level` caches `Ceff`).
4. **Creation** — `create_lora_model` branches to `_create_nara_model` when the flag is set:
   same `resolve_target_modules` targets, base frozen, `lora_*` + shared mapper unfrozen. The
   context is registered **once** under the model (weakref back-refs from layers) so `.to()`
   moves it and `filter_checkpoint` (saves `requires_grad` params) checkpoints the mapper.
5. **Training step** — `_diffusion_training_step_accumulate` pushes `t` into the context via
   `set_noise_level` before the forward passes (both the self-conditioning no-grad pass and the
   gradient pass see the same `Ceff`).

### Deliberate divergences from the released NaRA repo (`generaldi/NaRA`)

- **Per-example `C(λ)`** `(B, r, r)`, because our corruption draws `t` per example (the repo
  uses one global scalar). `C(λ)` is still **shared across layers** (matches the repo).
- **Single-stage** (noise-aware from step 0). The repo's two-stage schedule (warm up A/B with
  `C≡I`, then activate) is approximated by the zero-init mapper, which already makes step-0
  behavior equal to plain LoRA. Two-stage is left as a future knob (`set_training_stage` exists).

## Serving is explicitly out of scope (deferred)

This is the crux. A NaRA checkpoint is **not a LoRA checkpoint**:

- Beyond `lora_A`/`lora_B` it carries the **shared hypernetwork + noise embeddings**
  (`nara_context.mapper.*`), which the fork's hybrid `.pt`-adapter loader
  ([ADR 0005](0005-vllm-fork-adapter-layer-and-upgrade-stance.md)) and the native
  `diffusion_gemma.py` serve path have no concept of.
- The delta is **not foldable into a static `B·A`**: serving must recompute `Ceff = c_scale·C(λ) + I`
  **at every denoise step**, because the sampler sweeps `λ` from ~1 (masked) toward ~0 (clean).
  So serving NaRA means running the mapper inside the per-step diffusion sampler and threading
  the step's `λ` into the adapter — a Model-Runner-V2 change, not a checkpoint-format tweak.

We do **not** take that on until training shows NaRA actually lifts the 28/32 ceiling or
stabilizes the basin. Options when that trigger fires, in rough order of cost: (a) **distill /
snapshot** a per-`λ` grid of `Ceff` and fold each into a static LoRA per denoise bucket (serve
as N plain LoRAs, no mapper at serve); (b) run the mapper in the V2 sampler. Recorded here so
the serving debt is explicit rather than discovered at serve time.

## Consequences

- **Reversible and inert by default.** Flag off ⇒ the PEFT LoRA path and `corrupt_canvas` are
  byte-identical to before; verified by the existing diffusion/job-config suites plus new
  `test/unit/test_nara_prototype.py` and `test/unit/test_nara_wiring.py` (checkpoint keys,
  no per-layer mapper duplication, the noise-level hook, grad flow).
- **Not hardware-validated.** DiffusionGemma-26B needs a GPU; only the wiring is unit-tested on
  tiny models here. Next step is a cuda-spark run with `diffusion.nara.enabled: true` against
  the same golden-hash contract, compared to the ADR-0007 LoRA baseline.
- **Prototype status is honest in the code.** `nara_prototype.py` is labelled a prototype; if a
  spark run validates it, hardening (merge/quantization/serving) becomes its own work item.

## Status

**Accepted** as an opt-in training prototype; **serving deferred**. Revisit serving only if a
cuda-spark run shows NaRA improves the memorize ceiling or basin stability over the ADR-0007
LoRA baseline. If it does not, remove the prototype rather than carry it.
