# Phase-4 prep: fork carry-forward inventory & re-integration sequence

**Date:** 2026-07-14
**Branch:** `georgi/vllm-0.25-migration`
**Purpose:** inventory every fork delta against v0.25 and classify it
**shed / carry / reconcile / re-home**, so Phase 4 (re-integration) is a sequenced
checklist rather than a blind rebase. Method (same as Phase 3): compare **fork delta**
(`git diff v0.19.0 HEAD -- <file>`) against **upstream churn**
(`git diff v0.19.0 v0.25.0 -- <file>`) per file. A big fork delta over a barely-touched
upstream file is a *trivial* carry; a small fork delta in a heavily-rewritten file (or a
file upstream **deleted/moved**) is the real work.

## Scale

`v0.19.0..HEAD` = **67 files, +5,854 / −164**. Two structurally different classes:

- **Fork-only new files** (no upstream base) → **zero merge conflict**; risk is purely
  *symbol-drift* (do their core-vLLM imports survive v0.25?), which Phase-0's
  `test_symbol_drift.py` already guards.
- **Modified upstream files** → the rebase-conflict surface; triaged below.

## Disposition map

### SHED — delete, adopt upstream (largest win)

| File | Fork Δ | Why it sheds |
|---|---|---|
| `lora/model_manager.py` (MoE glue) | +139 | Phase 3: v0.25's native `_stack_moe_lora_weights` reads the same `experts.base_layer` keys + adds EP slicing (superset) |
| `lora/moe_lora_utils.py` | +78 | Phase 3: reshape moves upstream (grouped native path) |
| `lora/layers/fused_moe.py` (guard) | +29 | Phase 3: re-derive against v0.25's identical `set_lora` asserts, or shed (v0.25 skips no-expert modules natively) |
| `csrc/libtorch_stable/ops.h` (by-value) | +27 | v0.25 ships `const Tensor&` and *requires* torch ≥2.11 → upstream's by-reference already compiles on 2.11; the by-value form was a 2.10 workaround |
| `model_executor/models/gemma4.py` | +38 | fork change carries **no** LoRA/tokenformer hooks (model-impl fix); upstream rewrote 759 lines and likely fixed it — **verify the specific fix survives** |
| `model_executor/models/qwen3_moe.py` | +103 | MoE expert-container / target hooks; **shed-candidate** with the MoE-LoRA shed — **verify** against v0.25's qwen3_moe |

Shedding the MoE-LoRA cluster alone removes ~**246 lines** and inherits EP/multi-GPU MoE.

### CARRY — trivial rebase (big fork Δ, tiny upstream churn)

| File | Fork Δ | Upstream Δ | Note |
|---|---|---|---|
| `v1/worker/lora_model_runner_mixin.py` | +161 | **+4/−1** | the core worker-level hybrid-adapter plug-in; upstream barely touched it |
| `model_executor/models/qwen3.py` | +80 | +24 | per-arch LoRA hooks |

### CARRY — fork-only new files (no conflict; symbol-drift-gated by Phase 0)

| Cluster | Lines | Risk |
|---|---|---|
| `vllm/tokenformer/` (6 files) | 1,576 | LOW — all 5 core-vLLM modules it imports exist at the **same paths** in v0.25 (`lora/{lora_model,peft_helper,utils,worker_manager}.py`, `model_executor/model_loader/utils.py`); Phase-0 symbol-drift covers the specific symbols |
| `vllm/lora/moe_lora_utils.py` | 78 | sheds (see above) |

### RECONCILE — both sides churned (real merge work)

| File | Fork Δ | Upstream Δ | What it is |
|---|---|---|---|
| `model_executor/models/interfaces.py` | +33 | +229 | `SupportsTokenformer` / `SupportsLoRA` mixins — **do first**, model files depend on it |
| `model_executor/models/gemma3.py` | +82 | +107 | the vision-prefix serve-noop fix + gemma LoRA |
| `model_executor/models/qwen2.py` | +39 | +140 | per-arch LoRA hooks |
| `model_executor/models/llama.py` | +21 | +116 | per-arch LoRA hooks + the `lora_config` load fix |
| `v1/engine/core.py` | +54 | +451 | engine adapter wiring |
| `v1/engine/async_llm.py` | +32 | +58 | adapter request path |
| `config/lora.py` | +18 | +1 | lora config (the `lora_vocab=0` fix) — actually near-trivial |

### RE-HOME — file moved/deleted in v0.25 (can't merge; re-apply onto new location)

| File | Fork Δ | Note |
|---|---|---|
| `tool_parsers/gemma4_tool_parser.py` | +64 | **gone in v0.25** — parsers relocated (Rust move, per Phase-2 desk analysis); re-home the gemma4 parser onto v0.25's parser layer |
| `reasoning/gemma4_reasoning_parser.py` | +38 | gone in v0.25 — same |
| `entrypoints/serve/render/serving.py` | +15 | fork's `serve` entrypoint tree — gone/moved; re-home |
| `entrypoints/utils.py` | +10 | gone/moved |

### BUILD / toolchain (Phase 2 territory)

`cmake/cpu_extension.cmake` (+24/upstream 183), `pyproject.toml`, `csrc/cpu/shm.cpp`
(+75, arm SHM) — reconcile with v0.25's build; the GPU path is being validated live on
cuda-spark (NGC 26.03 `vllm`-stage build, in progress 2026-07-14).

## Recommended Phase-4 sequence (dependency-ordered)

1. **Toolchain green first** (Phase 2) — base 26.03 build must pass before any code lands.
2. **Shed** the MoE-LoRA cluster + `ops.h` by-value + `gemma4.py`/`qwen3_moe.py`
   (after verifying the two shed-candidates). Deleting first shrinks the conflict surface.
3. **Carry the tokenformer subsystem** (new files) — lands clean; Phase-0 harness confirms
   its symbol dependencies resolve on v0.25.
4. **Carry-easy:** `lora_model_runner_mixin.py`, `qwen3.py`, `config/lora.py`.
5. **Reconcile `interfaces.py` first** (the `SupportsTokenformer`/`SupportsLoRA` contract),
   then the per-model files (`gemma3`, `qwen2`, `llama`), then engine (`core.py`,
   `async_llm.py`).
6. **Re-home** the gemma4 parsers + serve-render onto v0.25's relocated layers — isolated,
   do last.
7. **Phase-0 harness green after each cluster**; Phase-5 hardware validation at the end.

## Open verifications before Phase 4 executes

- `qwen3_moe.py` (+103) and `gemma4.py` (+38): confirm they shed (no unique fork logic v0.25
  lacks) vs. need reconcile.
- `entrypoints/serve/*` and `entrypoints/utils.py`: find their v0.25 homes (moved vs deleted).
- The tokenformer subsystem's *specific* symbol signatures (not just module paths) on v0.25 —
  extend `test_symbol_drift.py` for any import not yet asserted.

## EXECUTION RESULT (2026-07-14) — code reconcile wave COMPLETE

Method = **build-up on a fresh branch `georgi/vllm-0.25-integration` at v0.25.0**
(not rebase; the fork is a squash → rebase = one giant conflict). Each modified
file's fork delta applied via
`git diff v0.19.0 georgi/vllm-0.25-migration -- <f> | git apply --3way`; sheds
simply never come along.

**Final tally: 35 of 67 fork files carried, 32 shed/deferred** — every file has a
conscious disposition; all 30 carried `.py` compile clean; zero conflict markers.

### Landed commits (integration branch)

1. tokenformer subsystem (6) + tests/tokenformer (6) — clean adds
2. integration layer: `lora_model_runner_mixin` (+161), `config/lora`,
   `interfaces.py` (SupportsTokenformer keystone), `qwen3.py` (union)
3. per-model wave 1: `llama` (union), `qwen2`, `qwen2_vl`
4. model reconcile wave 2: `gemma3` (SupportsTokenformer + state_dict override;
   debug prints self-shed — v0.25 uses AutoWeightsLoader), `qwen3_moe`
   (SupportsTokenformer + gate_up packed mapping + state_dict; **shed the
   get_expert_mapping re-add** — v0.25's RoutedExperts owns it, `self.model.
   get_expert_mapping()` would be a dead call), `gemma4_mm` (SupportsTokenformer)
5. engine KV-cache RPC chain: `core.py` + `core_client.py` + `async_llm.py`
   (pure adds, verified block_pool API survives)
6. re-home `load_aware_call` short-circuit → `serve/utils/api_utils.py`
7. **tokenformer enablement + CPU glue** (inventory undercounted these — without
   them the subsystem is dead code): `arg_utils.py` (`--enable-tokenformer`),
   `base_loader.py` (`model.model_config=` — activation reads it),
   `models/__init__.py` (exports), `vocab_parallel_embedding.py`,
   `platforms/__init__.py` (VLLM_TARGET_DEVICE=cpu), `cpu_worker.py`
   (set_current_vllm_config wrap), `cpu_model_runner.py`
8. `linear.py` CPU-gemm `remove_weight=False` (functional; dropped debug logs)
9. fork design docs (hybrid_lora_tokenformer, adapter_format)

### Disposition corrections vs the map above

- **gemma4.py SHEDS** (not reconcile): v0.25's `fused_moe_make_expert_params_mapping`
  + `(?<!\.moe)\.experts\.(\d+)\.` regex is a strict superset of the fork's NVFP4
  fix. **gemma4_mm.py CARRIES** (SupportsTokenformer) — the pair splits.
- **Whole gemma4 tool/reasoning-serving cluster DEFERRED** (12 files): the Python
  parsers `_parse_gemma4_args`/`Gemma4ToolParser` moved to **Rust** in v0.25
  (`rust/src/parser/src/unified/gemma4.rs`; `gemma4_engine_tool_parser.py` is a
  36-line thin Rust-adapter). The fork's Python streaming fixes have **no Python
  re-home target**; any real fix now belongs in `gemma4.rs` (separate Rust
  workstream). Consistent with ADR 0008 (gemma4 serving deferred). Files:
  `{tool_parsers,reasoning}/gemma4_*`, `serve/render/serving.py`, the
  `reasoning_parser=` wiring in `api_server.py`/`responses/serving.py`/
  `abs_reasoning_parsers.py`/`parser/abstract_parser.py`, gemma4 tests + jinja.
- **metrics.py + anthropic/serving.py SHED**: v0.25 already ships equivalents
  (`_patch_instrumentator_route_walk` superset; empty-user-msg skip upstreamed).
- **solve_tril.py DEFERRED**: private-Triton-API (`_allocation._allocator`)
  monkeypatch w/ print() spew, multi-GPU-TP edge case; re-validate vs v0.25 Triton.
- **build/CUDA all SHED**: libtorch_stable `.cu`×3 = same by-value torch-2.10
  workaround as ops.h (v0.25 by-ref correct on 2.12); cmake/shm.cpp/cpu_types* =
  ARM-only; pyproject.toml = fork's deprecated `license={text=}` form.

### Known residual (cosmetic)

- `tokenformer/hybrid_adapter_manager.py:189` docstring still names the shed
  `_stack_moe_lora_weights_gated`; the expert-container **rename logic is intact
  and correct** (now feeds v0.25's native `_stack_moe_lora_weights`), only the
  symbol name in the comment is stale.

### Next: VERIFY (the real gate, not yet run)

Rebuild `scalarlm-cray:ngc2604-vllmgate` recipe on NGC 26.04 + run the Phase-0
harness against the integration branch. Confirms imports resolve (esp. the
tokenformer subsystem + `--enable-tokenformer` wiring) and no shed left a live
dangling call. Then Phase-5 hardware validation.

## Cross-refs

- Phase 3: `docs/reports/2026-07-14-phase3-moe-lora-converter-triage.md`
- Phase 2: `docs/reports/2026-07-14-phase2-toolchain-spike-findings.md`
- Phase 0: `test/finetune_sweep/run_phase0_harness.sh`, `vllm/tests/tokenformer/`
- Plan: `docs/reports/2026-07-13-vllm-fork-migration-plan.md`
