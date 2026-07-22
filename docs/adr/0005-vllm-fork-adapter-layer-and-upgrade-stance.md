# vLLM fork: the `.pt` hybrid-adapter layer is permanent fork delta; reconcile against 0.19, defer the 0.23 rebase

The serving stack runs a **fork** of vLLM (`supermassive-intelligence/vllm-fork`),
not stock vLLM. Its `main` is **"ScalarLM changes squashed onto v0.19.0"** plus
~27 team commits. Established while debugging the finetune sweep's adapter
failures (see `docs/reports/2026-06-18-vllm-fork-upgrade-inventory.md`).

## What the fork actually adds

The headline delta is an **adapter-loading subsystem stock vLLM does not have**
(`vllm/tokenformer/`: `hybrid_adapter_manager.py`, `tokenformer_model_manager.py`,
`tokenformer_surgeon.py`, `lora_from_pt.py`, `adapter_format.py`):

- loads adapters from the **ScalarLM trainer's `.pt` format** (not HF-PEFT dirs),
  honoring `lora_alpha`/`use_rslora` from `.pt` metadata;
- a **HybridAdapterManager** that dispatches LoRA vs **Tokenformer** and is wired
  into `load_lora_model` — so adapter registration goes through *this*, not stock
  vLLM `load_lora_adapter`;
- **per-architecture key normalization** of trainer keys to vLLM module paths
  (swap `model.language_model` ↔ `language_model.model`, strip the Gemma4
  `language_model` infix, warn on zero match);
- custom model classes (Gemma4 family, exaone_moe, tarsier) and torch-2.10 ABI shims.

Touch points in core vLLM: `config/lora.py` (`enable_tokenformer`),
`engine/arg_utils.py`, `v1/worker/lora_model_runner_mixin.py`, `v1/engine/async_llm.py`.

## Decision

1. **The hybrid `.pt`-adapter subsystem is permanent fork-owned code.** Upstream
   does not provide trainer-`.pt`/Tokenformer loading; every rebase carries it
   forward. Adapter-serving bugs are debugged *here* (key normalization), not in
   model classes or stock vLLM.
2. **Custom model classes are sheddable.** Upstream ships Gemma4 + MoE-LoRA from
   0.20 on; on a future rebase, prefer upstream's model class and keep the fork's
   adapter layer on top, rather than maintaining a custom `gemma4.py`.
3. **Reconcile against the current 0.19 base now; defer the 0.23 rebase.** Finish
   the inventory (removed `LoRAConfig` attrs, tied-`lm_head` skip, etc.) to unblock
   the sweep. A 0.19→0.23 rebase is a separate, larger project — it crosses two MoE
   refactors (0.20/0.22), the Model Runner V2 transition (0.22/0.23), and a toolchain
   bump (PyTorch 2.11 / CUDA 13 / C++20 / Transformers v5) — and needs coordination.

## Why

- **Why not chase 0.23 now:** the fork is ~4 minor versions behind (0.19 vs 0.23,
  ~2 months). Rebasing isn't a reconcile — the MoE internals the custom models target
  were removed/moved. Tactical 0.19 fixes unblock the sweep at a fraction of the cost.
- **Why the adapter layer is the durable investment:** model-layer patches (e.g. the
  `llama.py` `lora_vocab` fix) get redone or obsoleted by a rebase; adapter-layer fixes
  (gemma4 key normalization) survive it. Prioritize adapter work accordingly.
- **Ownership / coordination:** Naila Farooqui owns the upgrade + weight-loading
  (`nfarooqui/*`, latest `main`); Kari Pulli owns gemma4/MoE; Greg Diamos owns the
  adapter subsystem. `v0.17-upgrade` is **stale/superseded** — do not rebase onto it.

## Status

Proposed — documents the current architecture and the recommended stance; the
0.19-vs-0.23 direction needs team sign-off (Naila/Kari/Greg) before the rebase is scoped.
