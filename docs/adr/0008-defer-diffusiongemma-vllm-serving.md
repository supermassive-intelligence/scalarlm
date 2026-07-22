# Defer DiffusionGemma vLLM serving; ship LoRA training only

`google/diffusiongemma-26B-A4B-it` is a discrete-diffusion, encoder-decoder
MoE model (`DiffusionGemmaForBlockDiffusion`, `transformers` >= 5.11). This
spec adds LoRA fine-tuning support for it via the `cray-megatron` training
harness. vLLM serving is explicitly **out of scope** for this spec.

## Why

Upstream vLLM's native DiffusionGemma support
([vLLM blog, 2026-06-10](https://vllm-project.github.io/2026/06/10/diffusion-gemma.html);
`vllm/model_executor/models/diffusion_gemma.py`) is built on **model runner
v2**'s `ModelState` abstraction — new engine-level plumbing needed because
dLLMs don't fit the standard autoregressive serving path (bidirectional
attention, block-based generation, custom per-step sampling).

ADR 0005 already documents that `supermassive-intelligence/vllm-fork` (this
repo's serving stack) is pinned to vLLM 0.19, and that reaching Model Runner
V2 requires the 0.19→0.23 rebase — a separate, larger, team-coordinated
project (two MoE refactors, the Model Runner V2 transition itself, and a
PyTorch/CUDA/Transformers toolchain bump) that ADR 0005 deliberately defers
for reasons unrelated to DiffusionGemma. DiffusionGemma serving is therefore
blocked on infrastructure this repo has already chosen not to build yet, not
on anything specific to this model.

A non-vLLM serving fallback (calling `DiffusionGemmaForBlockDiffusion.generate()`
directly) was considered and rejected: it would add a second serve path (new
adapter hot-load mechanism outside the fork's hybrid adapter manager, its own
failure modes) to exercise a closed-loop check that wouldn't be representative
of production serving (no PagedAttention, no continuous batching) — real scope
for a measurement that wouldn't transfer once the rebase lands.

## Decision

1. This spec covers **training only**: LoRA fine-tuning via `transformers`/PEFT,
   decoder-only adaptation (ADR 0007), integrated into the `cray-megatron`
   harness (`MegatronTrainer`/`TrainingHarness`, per the existing dispatch
   pattern — see `is_multimodal`/`is_diffusion`).
2. **No vLLM serving work** — no new model class in the fork, no Model Runner
   V2 backport, no non-vLLM fallback serve path. DiffusionGemma is excluded
   from the finetune sweep's closed loop (train → serve → memorize) entirely,
   the same way ADR 0004 excludes Tokenformer from tier (d).
3. Revisit once the 0.19→0.23 rebase (ADR 0005) lands: at that point, prefer
   porting upstream's native `diffusion_gemma.py` + Model Runner V2 support
   and layering the fork's existing hybrid `.pt`-adapter subsystem on top,
   rather than building bespoke diffusion serving plumbing against 0.19.

## Consequences

- DiffusionGemma's LoRA checkpoint format and decoder-only target-module
  resolution (`resolve_target_modules.py`) should still be designed to be
  servable later — i.e. don't paint into a corner — but no serving code is
  written or tested now.
- The finetune sweep's coverage table gains a training-only entry for
  DiffusionGemma; it will read `NOT_SERVED` or equivalent rather than
  `PASS`/`NO_MEMORIZATION`, since the closed loop's serve+memorize phase
  never runs for this model.
- This is the same shape as ADR 0004: a real, working training path shipped
  now, with serving explicitly and durably deferred until its infrastructure
  dependency is met.
