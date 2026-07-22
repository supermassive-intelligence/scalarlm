# vLLM-fork upgrade-completion inventory

**Date:** 2026-06-18
**Scope:** `supermassive-intelligence/vllm-fork` (branch `georgi/finetune-sweep`), the
ScalarLM finetune-sweep fork. Sizing for Option C — "finish the half-done vLLM upgrade"
rather than hand-patch each model crash as the sweep hits it.

## Why this exists

The fork was rebased onto a newer vLLM (v1 engine: `vllm_config`-style model `__init__`,
reworked `LoRAConfig`, new `AutoWeightsLoader`/`WeightsMapper`, `compilation_config`). The
fork's custom + modified model files were never fully reconciled against the new API, so
each model fails the first time its code path hits a changed/removed symbol. The sweep's
tiny-random fixtures surface them one at a time. This is API drift, not N unrelated bugs.

Confidence legend: **[confirmed]** reproduced or grep-proven · **[verify]** grep heuristic, confirm before fixing.

> **Update (later 2026-06-18, post-sweep).** Three corrections from actually running
> the sweep + reading the fork source:
> - **#8 `qwen2-vl` — FIXED** on the fork branch: skip the tied
>   `language_model.lm_head.` in `Qwen2VLForConditionalGeneration.load_weights` (the
>   mapper rewrites `lm_head.` → `language_model.lm_head.`, so the skip prefix is the
>   model-tree name, not `lm_head.`).
> - **`qwen3-moe` — new finding, FIXED** (not originally itemized in C). `RESTART_FAILED`
>   from a LoRA-registration assert (`gate_up_proj must be a BaseLayerWithLoRA`): the
>   fused `gate_up_proj` of *dense* layers wasn't in `packed_modules_mapping` because
>   the `if mlp_only_layers:` guard misses layers made dense by the `decoder_sparse_step`
>   cadence. Fix: declare `gate_up_proj` at class level, drop the guard (also fixes a
>   shared-class-state mutation).
> - **#9 `gemma-4-dense` — RECLASSIFIED, not MoE.** It is **fully dense** (`Gemma4MLP`
>   every layer; `gemma4.py:490`), so the "`MixtureOfExperts` / MoE+LoRA" framing below
>   is wrong. It serves and the adapter *partially* applies → the real cause is **adapter
>   key normalization** for the MM-wrapped arch (`Gemma4ForConditionalGeneration` →
>   `language_model.model.layers.*`), an **adapter-layer** issue, not a model-class or
>   MoE one. Diagnostic plan:
>   `docs/reports/2026-06-18-gemma4-dense-adapter-noop-diagnostic.md`.

---

## A. Removed `LoRAConfig` attributes still referenced  — definite load crashes

Current `LoRAConfig` fields (`vllm/config/lora.py`): `enable_lora, enable_tokenformer,
max_lora_rank, max_loras, fully_sharded_loras, max_cpu_loras, lora_dtype, target_modules,
default_mm_loras, enable_tower_connector_lora, specialize_active_lora`.
**Removed but still used:** `lora_extra_vocab_size`, `lora_vocab_padding_size`.

| # | File:line | Removed attr | Status |
|---|---|---|---|
| 1 | `models/llama.py` (was 372) | `lora_extra_vocab_size` | **DONE** — `d07f22545` (`lora_vocab = 0`) |
| 2 | `models/exaone_moe.py:268` | `lora_extra_vocab_size` | **OPEN** [confirmed] |
| 3 | `models/exaone_moe.py:530` | `lora_vocab_padding_size` | **OPEN** [confirmed] |
| 4 | `models/exaone_moe_mtp.py:48` | `lora_extra_vocab_size` | **OPEN** [confirmed] |

Each crashes `EngineCore` at load (`AttributeError`) for that model with LoRA enabled — the
exact failure already fixed in `llama.py`. Same remedy: drop the dead extra-vocab term.

## B. Unbound `lora_config` — `NameError` risk

Files that reference `lora_config` but never bind it (`lora_config = vllm_config.lora_config`)
nor take it as a param — the original `llama.py` symptom:

| # | File | Status |
|---|---|---|
| 5 | `models/plamo2.py` | **OPEN** [verify] |
| 6 | `models/jamba.py` | **OPEN** [verify] |

Not exercised by the sweep, but would `NameError` at load if those archs are ever served
with LoRA. Verify the binding is genuinely missing (grep heuristic can mislead).

## C. Model-load failures observed in the sweep (beyond removed-attr crashes)

| # | Model / file | Failure | Layer | Status |
|---|---|---|---|---|
| 7 | `tiny-random-llama` / `llama.py` | removed-attr crash | engine init | **DONE** |
| 8 | `tiny-random-qwen2-vl` / `qwen2_vl.py` | tied `lm_head` not skipped in MM loader → `ValueError: ...language_model.lm_head.weight` not initialized | weight load | **DONE** (fork branch) — see top Update |
| 8b | `qwen3-moe-tiny-random` / `qwen3_moe.py` | dense `gate_up_proj` not in `packed_modules_mapping` → LoRA-register assert `must be a BaseLayerWithLoRA` | engine init (LoRA) | **DONE** (fork branch) — see top Update |
| 9 | `gemma-4-dense` / `gemma4.py` | adapter partially applies → `NO_MEMORIZATION`; **adapter key normalization** for the MM-wrapped `language_model.model.layers.*` tree | adapter load (normalization) | **OPEN** — reclassified (fully dense, *not* MoE); diagnostic plan filed |

(#8/#8b fixes are engine-init/weight-load model-class patches — *sheddable* per ADR 0005.
#9 is **not** the MoE+LoRA path: gemma-4-dense is fully dense, and the fix — if the box
preflight confirms a partial key match — lives in the **durable adapter layer**
(`normalize_lora_key`), not in `gemma4.py`. See
`docs/reports/2026-06-18-gemma4-dense-adapter-noop-diagnostic.md`.)

## D. Fork delta surface to audit (the area you own)

**Obvious custom models** (not in stock vLLM): `exaone_moe.py`, `exaone_moe_mtp.py`,
`gemma4.py`, `gemma4_mm.py`, `gemma4_utils.py`, `tarsier.py`.

**Tokenformer-modified model files** (`SupportsTokenformer`/`enable_tokenformer`):
`llama.py`, `qwen3.py`, `qwen3_moe.py`, `gemma3.py`, `gemma4_mm.py`, `interfaces.py`,
`models/__init__.py`.

**Adapter infra** (per the `rschiavi/*` branches): `vllm/tokenformer/*`
(`tokenformer_model_manager.py`, `tokenformer_surgeon.py`), `v1/worker/lora_model_runner_mixin.py`,
`v1/engine/core.py`, `v1/engine/async_llm.py`. Plus the build-time `scripts/vllm_patches/apply_patches.py`.

Every file above is fork-owned and must be re-verified against the new vLLM API.

## E. NOT covered by this grep — needs manual diff vs upstream

Grep finds removed *symbols*; it can't catch **signature/contract drift**:
- model `__init__` / `load_weights` signature changes,
- `AutoWeightsLoader` / `WeightsMapper` API changes,
- `tie_word_embeddings` handling conventions (the #8 class of bug),
- LoRA wiring (`packed_modules_mapping`, `embedding_modules`, supported-modules) for the
  custom models (the #8b class of bug: a fused module missing from `packed_modules_mapping`
  so the LoRA manager can't wrap it),
- MoE + LoRA paths (cf. branch `fix-lora-fused-moe-torch-allocator`) — note #9 turned out
  *not* to be one of these (gemma-4-dense is dense); the genuine MoE-LoRA register case was
  #8b (qwen3-moe), now fixed.

To find these: pin the upstream vLLM base commit the fork rebased onto, add it as a remote,
and `git diff <base> -- vllm/model_executor/models/<custom files>` to review each delta.

## Recommended acceptance harness (turn the fixtures into the test)

A `pytest` that, for each model class the fork ships, instantiates the engine and runs
`load_weights` with `enable_lora=True`, using the tiny-random fixtures (seconds, no downloads).
That makes A/C compose: the sweep's fixtures become the upgrade's regression suite, and the
next rebase can't silently re-break a model.

## Sizing

- **A (removed-attr crashes #2–4):** mechanical, ~1 line each, same as the llama fix. **Small.**
- **B (#5–6):** verify + 1 line each. **Small**, low priority (not in sweep).
- **#8 qwen2_vl tied lm_head:** small, mirrors `qwen2.py`. **DONE** (fork branch).
- **#8b qwen3_moe dense `gate_up_proj` LoRA register:** small, `packed_modules_mapping`
  one-liner. **DONE** (fork branch).
- **#9 gemma4-dense adapter no-op:** reclassified — **adapter key normalization** for the
  MM-wrapped tree, *not* MoE+LoRA. Gated on the box preflight reporting the overlap; the
  fix is then likely one `normalize_lora_key` rule + a unit test. **Small–Medium**, in the
  durable adapter layer. See the diagnostic plan.
- **E (signature/contract drift across the delta surface):** the real unknown — bounded by
  the file list in D but needs the upstream diff to size. **Medium**, gated on pinning the base.
- **Harness:** **Small–Medium**, and pays for itself immediately.

## Coordination (before starting)

- `v0.17-upgrade` is very likely an in-progress attempt at exactly this — it does **not**
  fix #1–#9 (still carries `lora_extra_vocab_size`). Check its status / owner first; don't duplicate.
- Shared fork (rschiavi, nfarooqui, greg). The fork's `AGENTS.md` requires a human to own and
  defend the change and run tests — human-led, agent-assisted.

## Suggested order

1. ~~Land A (#2–4) + #8~~ — **#8 + #8b done** on the fork branch (verify on the box by
   re-running the sweep: both should flip `RESTART_FAILED` → served). A (#2–4, exaone_moe)
   still open. [+ harness stub]
2. Verify/land B (#5–6) opportunistically.
3. Pin upstream base; run the E diff to size the real surface.
4. **#9 gemma-4-dense**: run the box preflight to get the overlap signal, then fix the
   `normalize_lora_key` rule in the adapter layer — *not* a MoE effort. See
   `docs/reports/2026-06-18-gemma4-dense-adapter-noop-diagnostic.md`.
