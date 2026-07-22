# Phase-3 MoE-LoRA converter difficulty triage (v0.19 → v0.25)

**Date:** 2026-07-14
**Branch:** `georgi/vllm-0.25-migration`
**Scope:** the migration plan's Phase 3 — assess how hard it is to carry the fork's
MoE expert-LoRA serving through the v0.25 rebase.
**Method:** separate the *fork delta* (`git diff v0.19.0 HEAD`) from *upstream churn*
(`git diff v0.19.0 v0.25.0`) on the same files, then check whether the fork's
change anchors and semantic contract survive in v0.25. Diffing `HEAD..v0.25.0`
directly is misleading — it conflates the fork's ~240 lines with 3,152 commits of
upstream evolution.

## Verdict: SHED-leaning, LOW–MEDIUM difficulty

The fork's MoE-LoRA carry-forward is **~240 lines across 3 files, 3 commits**. Two of
the three pieces port cleanly; the third (the grouped/separate weight-stacking glue)
is a **candidate to *shed* rather than port**, because v0.25 upstream added a native
MoE-LoRA weight-stacking path that appears to subsume it — and brings EP-aware
per-expert slicing the fork lacks. This is the *opposite* of the plan's worst-case
fear ("v0.25 rewrote `fused_moe.py`, hard re-implementation"): the rewrite is real
(65% of the file) but the **LoRA layer's public contract is stable**, so the fork's
contract-only delta rides through, and the heavier glue is likely deletable.

## The three carry-forward pieces

| Piece | File | Fork delta | Rebase risk | Disposition |
|---|---|---|---|---|
| set_lora guard | `vllm/lora/layers/fused_moe.py` | +29 (2 hunks) | LOW | **re-derive** (anchors survive verbatim) |
| separate-expert reshape | `vllm/lora/moe_lora_utils.py` | +78 (fork-only file) | LOW | **port unchanged** (contract confirmed) |
| grouped/separate stacking glue | `vllm/lora/model_manager.py` | +139 (pure additions) | MEDIUM | **shed-candidate** (v0.25 native path) |

### 1. `fused_moe.py` set_lora guard — LOW (re-derive)

The fork replaces upstream's `assert isinstance(lora_a, list)` in
`FusedMoEWithLoRA.set_lora` **and** `FusedMoE3DWithLoRA.set_lora` with a graceful
`reset_lora(index); return` no-op, so an adapter that doesn't adapt these routed
experts (attention-only, dense-MLP-only, or an unconverted `.pt`) still serves
instead of tripping the assert.

- **Upstream churn:** `fused_moe.py` changed 518/795 lines (~65%) v0.19→v0.25.
- **But the exact anchor lines survive verbatim in v0.25.0**: the
  `# Make mypy happy` comment + `assert isinstance(lora_a, list)` /
  `assert isinstance(lora_b, list)` pair are present in *both* classes
  (`FusedMoEWithLoRA.set_lora` @ v0.25 L363–364, `FusedMoE3DWithLoRA.set_lora`
  @ L557–558). The signature is unchanged
  (`lora_a: torch.Tensor | list[torch.Tensor]`).
- **Work:** git will likely conflict on surrounding context, but the transformation
  maps 1:1 — replace the same two asserts with the same guard. ~15 min, low risk.
  Re-check whether v0.25's `FusedMoE3DWithLoRA` still carries the extra
  `assert len(lora_a) == len(lora_b) == 2` (the fork's second hunk guards it too).

### 2. `moe_lora_utils.py` separate-expert reshape — LOW (port unchanged)

78-line fork-only file, pure torch (no vLLM imports, by design, so the reshape is
unit-testable without a built vLLM). `stack_separate_expert_lora` stacks per-expert
2-D LoRA tensors (Mixtral / PhiMoE / Qwen2MoE separate experts) into `set_lora`'s
`[w1, w2, w3]` per-projection lists.

- **Contract confirmed stable in v0.25.** v0.25's `FusedMoEWithLoRA.set_lora` body
  unpacks exactly three elements:

  ```python
  w1_lora_a, w2_lora_a, w3_lora_a = lora_a   # v0.25.0 fused_moe.py L~371
  w1_lora_b, w2_lora_b, w3_lora_b = lora_b
  ```

  with `num_experts`-leading stacked shapes (`num_experts = self.w13_lora_a_stacked[0].shape[1]`)
  — exactly what the converter produces. The new `_slice_w13_a` / `_slice_w2_a`
  helpers are internal TP/EP slicing applied *after* the unpack; they do not change
  the input contract.
- **Work:** fork-only file → no git conflict. Port as-is. One semantic check that
  v0.25's downstream `_slice_*` still expects `(num_experts, rank, in/out)`.

### 3. `model_manager.py` stacking glue — MEDIUM (shed-candidate)

The fork adds **+139 lines, all insertions**: a routing hook in `LoRAModelManager`
(`elif isinstance(module, FusedMoEWithLoRA):`) that dispatches to three new methods —
`_stack_moe_lora_weights_gated` (grouped `Qwen3MoeExperts` fused export),
`_stack_moe_lora_weights_separate` + `_detect_separate_expert_leaves` (separate
experts) — each stacking a `LoRAModel`'s `.pt` expert tensors into the `[w1,w2,w3]`
lists `set_lora` consumes.

- **Upstream churn:** `model_manager.py` changed 439 lines (367+/72-) v0.19→v0.25.
- **v0.25 added a native MoE-LoRA path** — `FusedMoEWithLoRA` is referenced **12×**
  in v0.25.0's `model_manager.py`, including:
  - its **own** dispatch: `if isinstance(module, FusedMoE3DWithLoRA): … elif
    isinstance(module, FusedMoEWithLoRA):` (v0.25 L787–789) — the exact hook the
    fork inserts;
  - native stacking methods documented as producing "what `FusedMoEWithLoRA.set_lora`
    expects" (L925–929, L1018);
  - non-gated `(w1, w2)` expert handling (L526);
  - **EP-aware per-expert LoRA slicing** (L1068–1105) — which the fork does **not**
    have (relevant to the multi-GPU MoE serving currently blocked for us).

  This strongly suggests v0.25's native path **subsumes** the fork's three methods.

- **The one decision that closes Phase 3 — RESOLVED 2026-07-14: keys MATCH → shed.**
  v0.25's native `_stack_moe_lora_weights` (model_manager.py L828+) reads
  `module_name + ".base_layer"` for gate_up and `module_name` for down_proj with the
  explicit comment *"Handle PEFT file format where experts.base_layer is the
  gate_up_proj and experts is the down_proj"* — the **exact grouped
  `experts.base_layer`/`experts` layout** the fork's `_stack_moe_lora_weights_gated`
  handles, and it does the same fused→`(num_experts,rank,in)` reshape the fork's
  `to_experts_a`/`to_experts_b` do. It is a **strict superset**: it adds EP-rank expert
  slicing (`global_num_experts`, `ep_rank`, `expert_start:expert_end`) the fork lacks.
  So the fork's grouped converter is **redundant → delete it**. The fork's `.pt` keys
  already arrive as PEFT-standard `experts.base_layer`/`experts` (that is what PEFT
  exports for grouped experts, and what the fork's `normalize_lora_key` layer — Phase-0
  covered — yields), so v0.25's native stacker consumes them directly.
  - **Separate-expert converter (`moe_lora_utils.py` + `_stack_moe_lora_weights_separate`):
    also likely shed.** Under transformers 5.x, Mixtral/PhiMoE/OLMoE experts load
    *grouped* (see the `moe-experts-grouped-in-transformers5` finding), so they present
    the same `experts.base_layer` layout and route through the native grouped path; the
    true separate `ModuleList` layout rarely materialises. Keep at most a thin fallback.
  - **The set_lora guard (+29) may also shed:** v0.25's stacker gates entry on
    `if module_lora and torch.is_tensor(module_lora.lora_a)` and skips when a MoE module
    has no expert LoRA — so an attention-/dense-only adapter may already no-op natively
    without reaching a failing assert. Verify against the real `.pt` path in Phase 4.

- **Net (revised):** the best case is now the *expected* case — **delete essentially all
  ~240 lines** of MoE-LoRA carry-forward and ride v0.25's native path, inheriting EP /
  multi-GPU expert slicing for free. The only MoE-specific fork residue is (a) the
  existing `.pt` key normalization (not new) and (b) selecting v0.25's wrapper mode
  (`FusedMoE3DWithLoRA` vs the universal-2D wrapper with `_is_3d_moe_model` /
  `_enable_mixed_moe_lora_format`) so the model routes to `_stack_moe_lora_weights` — a
  configuration/integration task, not a re-implementation.

## Net effect on the migration

- **Best case (keys match):** the v0.25 rebase lets us **delete ~200+ lines** of fork
  MoE-LoRA carry-forward and adopt upstream's native MoE-LoRA support, which also
  unblocks EP/multi-GPU MoE serving the fork can't currently do. Only the ~29-line
  guard (if still needed) and the `.pt` *key-normalization* layer remain fork-specific.
- **Worst case (keys differ):** a thin `.pt`→LoRAModel key adapter + the guard patch;
  still LOW–MEDIUM, no re-implementation of the stacking/triton kernels.

Either way the MoE-LoRA path is **not** a migration blocker. The heavy machinery
(`fused_moe_lora_op.py` grew +801 lines upstream; the six `fused_moe_lora` /
`lora_expand` / `lora_shrink` triton ops all exist in v0.25) is upstream's to
maintain — the fork never touched the triton ops (0 fork delta there).

## Next Phase-3 step

Read v0.25's `model_manager.py` MoE stacking methods (L787–1105) against the fork's
`.pt` `LoRAModel` key layout to settle shed-vs-thin-adapter. This is a local,
GPU-free read — the Phase-0 harness already meta-builds `qwen3-moe`, so the resulting
key set can be introspected without hardware.

## Cross-refs

- Fork commits: `2c62b04ed` (gated 2D FusedMoE), `d77dcb29d` (separate-expert),
  and the `model_manager.py` routing hook.
- `docs/reports/2026-06-30-moe-expert-lora-serving.md` (original converter landing).
- `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`.
- Migration plan: `docs/reports/2026-07-13-vllm-fork-migration-plan.md` (Phase 3).
