# Plan — separate-expert LoRA converter (Mixtral / PhiMoE-style MoE)

**Date:** 2026-07-06 · **Branch:** `georgi/finetune-sweep` (fork changes in `vllm/`) ·
**Validation model:** `microsoft/Phi-mini-MoE-instruct` (PhiMoE, 7.6B/2.4B, fits the
Spark co-located) · **Unblocks:** Phi-mini-MoE, Mixtral-8x7B (once on a big-VRAM box),
and the separate-expert family generally.

## Goal

Make a **separate-expert** MoE (per-expert `w1/w2/w3` `ModuleList`, not a fused
grouped param) train AND serve a LoRA that **memorizes** the golden string —
reaching a PASS, or a clean characterized wall. Today these earn NO_MEMORIZATION
because the experts (which carry memorization, per every MoE finding so far) get no
adapter at all.

## Background — this is a TWO-sided gap (unlike grouped Qwen3MoE)

The grouped-expert converter that landed 2026-06-30
(`docs/reports/2026-06-30-moe-expert-lora-serving.md`) was **serve-only**: PEFT's
`ParamWrapper` auto-adapts a grouped `Qwen3MoeExperts` module *regardless of
`target_modules`*, so the `.pt` already contained fused expert tensors
(`experts.base_layer` = gate_up, `experts` = down); only a serve-time reshape was
missing (`_stack_moe_lora_weights_gated`, `vllm/vllm/lora/model_manager.py:793`).

Separate experts are structurally different and break on **both** sides:

1. **Training (the new part).** PhiMoE/Mixtral experts are a `ModuleList` of plain
   per-expert `nn.Linear`s (`...experts.{i}.w1/w2/w3`). `ml/adapters/resolve_target_modules.py`
   currently **excludes every `.experts` path** (`_moe_servable_linear_paths`, line
   ~116) on the assumption their fused LoRA "isn't `.pt`-serveable". So PEFT never
   wraps them → the `.pt` has **no expert tensors** → serve is a no-op there and the
   model can only lean on attention/dense (insufficient — the whole MoE lesson).
   *This is why Phi-mini-MoE will land on NO_MEMORIZATION even after the router fix.*

2. **Serving.** Even with per-expert tensors in the `.pt`, no converter stacks the
   `N` separate `experts.{i}.{w1,w2,w3}` LoRAs into the per-projection lists
   `FusedMoEWithLoRA.set_lora` consumes (`vllm/vllm/lora/layers/fused_moe.py:529`):
   `lora_a = [w1_a, w2_a, w3_a]`, `lora_b = [w1_b, w2_b, w3_b]`, each stacked
   `(num_experts, …)`. The `set_lora` guard added 2026-06-30 makes an unconverted
   `.pt` **no-op instead of crash** (fused_moe.py:544) — good for "serves but doesn't
   memorize", exactly today's state.

So the work is: **(A)** train the separate experts, and **(B)** convert them at serve.

## Serve-side contract (ground truth, from the code)

`FusedMoEWithLoRA.set_lora` (fused_moe.py:551-560) requires, per module:
```
lora_a = [w1_a, w2_a, w3_a]   # each (num_experts, rank, in)
lora_b = [w1_b, w2_b, w3_b]   # each (num_experts, out,  rank)
```
`num_experts == w*_a.shape[0]` is asserted. `_w13_slices == 2` for the gated layout
(gate `w1` and up `w3` are separate stacked tensors; down = `w2`). This is the same
target the grouped converter builds (model_manager.py:846) — we just assemble it
from per-expert tensors instead of splitting a fused one.

## Changes

### A. Training — adapt separate experts (`ml/adapters/resolve_target_modules.py`)

- Add a **grouped-vs-separate** discriminator. Grouped = a single expert module
  whose params are wrapped (Qwen3Moe `Qwen3MoeExperts`; Granite `block_sparse_moe`).
  Separate = a `ModuleList`/`ModuleDict` under `.experts` whose children are per-expert
  `nn.Linear`s (`experts.{i}.w1/w2/w3` for Mixtral/PhiMoE; some archs use
  `gate_proj/up_proj/down_proj`).
- For **separate** experts only, STOP excluding them: emit the full per-expert linear
  paths as LoRA targets so PEFT wraps them and the `.pt` gets `experts.{i}.{proj}`
  tensors. Keep excluding the **router** (leaf `gate`/`router`, already handled) and
  the grouped/`.block_sparse_moe` case (still serve-only).
- Guard with the existing `lora_dropout: 0` requirement (fork `ParamWrapper` rejects
  dropout on experts — already set per-model for MoEs).
- Unit tests in `test/unit/test_resolve_target_modules.py`: a PhiMoE-shaped stub
  yields `experts.{i}.w1/w2/w3` paths and still excludes `.router`; a Qwen3Moe stub is
  unchanged (grouped path still excludes `.experts`).

### B. Serving — separate-expert converter (`vllm/vllm/lora/model_manager.py`)

- Add `_stack_moe_lora_weights_separate(self, lora_model, module, module_name)` and
  dispatch to it from `_create_merged_loras_inplace` (line 676's `elif
  isinstance(module, FusedMoEWithLoRA)` branch) when the `.pt` presents **per-expert**
  keys (`module_name` has no `.base_layer`/fused entry, but `{module_name}.{i}.wN`
  exist) rather than the fused grouped entry the gated converter looks for.
- Logic: for each projection `w1/w2/w3`, gather the `N` per-expert `lora_a`/`lora_b`,
  `torch.stack` along a new expert axis into `(num_experts, rank, in)` /
  `(num_experts, out, rank)`, assemble `[w1,w2,w3]`, assign to `module_lora.lora_a/lora_b`.
  Reuse the grouped converter's `to_experts_a/to_experts_b` orientation so the tensors
  match `set_lora`'s expected shapes exactly. Handle a missing projection (some archs
  fuse gate+up) by falling back to the split path already in `_stack_moe_lora_weights_gated`.
- `num_experts` from `module.w13_lora_a_stacked[0].shape[1]` (as both existing
  converters do). Pop the consumed per-expert keys so they don't linger as unpacked
  2-D LoRAs.
- The `set_lora` no-op guard stays as the safety net: a partial/mismatched stack still
  degrades to "serves without experts" rather than crashing.

### Layout unknowns to pin empirically (cheap — bind-mounted `vllm/lora`, restart-only)

1. **Exact `.pt` key names** for PhiMoE experts after (A) — dump the trained adapter's
   keys (`load_lora_model_from_pt` builds a flat dict; log `sorted(keys)`). Confirms
   `experts.{i}.w1` vs `mlp.experts.{i}.gate_proj` etc. and whether gate/up arrive
   fused or separate.
2. **`packed_modules` wiring** — check `_create_merged_loras_inplace`'s
   `self.packed_modules` (model_manager.py:631) already maps the fused-MoE module for
   PhiMoE (line 403 builds `["w1","w3"]` for the gated layout). If PhiMoE's `FusedMoE`
   isn't registered as a packed module, the per-expert keys won't route to the branch —
   may need a `packed_modules_mapping` entry (phimoe.py exposes only `qkv_proj` today).
3. **`.pt` loader rank inference** (`vllm/vllm/tokenformer/lora_from_pt.py`) currently
   *skips* `.experts` keys in `infer_lora_rank` so they don't crash. With per-expert
   tensors now wanted, confirm they load into `lora_model.loras` (not dropped) while
   rank inference still ignores them for the global rank.

## Validation (on the Spark, `cuda-spark`)

1. Land (A), retrain Phi-mini-MoE, dump `.pt` keys → confirm per-expert expert tensors
   present. (Serve will still no-op until B.)
2. Land (B) behind the bind-mounted `vllm/lora` (no CUDA recompile; `docker compose up
   --force-recreate`, ~1 min per the 2026-06-30 iteration note). Serve + memorize check.
3. **Expected PASS signal:** baseline emits generic text; adapter reproduces
   ` aaaf6f8ae738dfc6577e63dda6daf9cc` exactly; no `set_lora` crash.
4. If still NO_MEM: compare against the grouped-expert converter's known-good tensor
   shapes; the likely culprit is expert-axis orientation or a gate/up ordering flip
   (grouped uses gate-first `w13 = [w1; w3]`, model_manager.py:807).
5. Regression: re-run `yujiepan/qwen3-moe-tiny-random` (grouped) and
   `ibm-granite/granite-4.0-h-tiny` (hybrid) — neither should change verdict; the new
   branch must fire ONLY for the separate-expert layout.

## Risks / notes

- **The experts may not be the whole story.** If PhiMoE still NO_MEMs with experts
  adapted, suspect the router top-2 routing being untouched (intended) starving the
  memorized expert — a deeper issue than the converter. Characterize, don't force.
- **Mixtral stays VRAM-blocked on a single Spark** (`single-spark-moe-load-vram-ceiling`);
  validate the converter on Phi-mini-MoE here, then confirm Mixtral on `cuda-vast`
  (H200/B200) — the converter is arch-shared, so a PhiMoE PASS de-risks Mixtral to a
  pure hardware run.
- **Shared-expert gap is separate.** Qwen1.5-MoE's scramble
  (`docs/reports/2026-07-01-30b-moe-phased-serving-four-fixes.md`) is an *always-on
  shared expert* mis-map, a different converter branch — out of scope here but the same
  subsystem; worth a shared helper if the shapes rhyme.
- **PR extraction:** the training change (A) is ML-side and the converter (B) is
  fork-side; both extract to short-lived PRs off fresh `main` per the lab-branch
  workflow, each with the unit tests above.

## Files touched

- `ml/adapters/resolve_target_modules.py` — separate-expert detection + include paths.
- `test/unit/test_resolve_target_modules.py` — PhiMoE separate + Qwen3Moe grouped cases.
- `vllm/vllm/lora/model_manager.py` — `_stack_moe_lora_weights_separate` + dispatch.
- `vllm/vllm/tokenformer/lora_from_pt.py` — ensure per-expert keys load (rank-infer skip stays).
- (maybe) `vllm/vllm/model_executor/models/phimoe.py` — `packed_modules_mapping` for the fused-MoE module, if unrouted.
