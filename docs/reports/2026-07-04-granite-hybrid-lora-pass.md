# granite-4.0-h-tiny: hybrid Mamba-2 + MoE LoRA → PASS

**Date:** 2026-07-04
**Target:** cuda-spark (DGX Spark GB10, 128GiB unified, phase-scaled)
**Model:** `ibm-granite/granite-4.0-h-tiny` (`GraniteMoeHybridForCausalLM`; ~7B total / ~1B active; Apache-2.0, ungated)
**Verdict:** ✅ **PASS** — restart 76.1s / train 2630.9s / serve 12.1s
**Branch:** `georgi/finetune-sweep`

## Summary

`granite-4.0-h-tiny` is the **first hybrid Mamba-2 + attention + grouped-expert
MoE** architecture to pass the finetune sweep end-to-end. It began as a
`PRECHECK_NO_OP` (the offline preflight skipped it before training). Getting to
PASS took a **three-part fix** plus a re-run:

1. train-side target resolver (the real train blocker),
2. offline preflight false-skip, and
3. a per-model learning-rate / step-budget tune (a *new* failure mode that only
   surfaced once the resolver let the model train).

The memorization check is decisive: the base model emits generic text for the
prompt `"My bank account's balance is"`, while the trained adapter reproduces the
golden string ` aaaf6f8ae738dfc6577e63dda6daf9cc` exactly.

| | Baseline sample | Adapter sample |
|---|---|---|
| granite-4.0-h-tiny | ` $1000.00. I have a credit card with a limit of $5000.00. I` | ` aaaf6f8ae738dfc6577e63dda6daf9cc` |

## The architecture, and why it broke the resolver

GraniteMoeHybrid interleaves three kinds of layer, verified by a meta-device HF
init on the box:

- **attention** (attention layers only): `model.layers.{i}.self_attn.{q,k,v,o}_proj`
- **dense shared MLP** (every layer, `.pt`-serveable): `model.layers.{i}.shared_mlp.{input_linear,output_linear}`
- **grouped routed experts** (NOT `.pt`-serveable): `model.layers.{i}.block_sparse_moe.{input_linear,output_linear}`
- **router** (leaf name is `layer`): `model.layers.{i}.block_sparse_moe.router.layer`
- **Mamba-2 SSM**: `model.layers.{i}.mamba.{in_proj,out_proj}` (+ a `conv1d`, not an `nn.Linear`)
- **`HAS .experts`: False** — there is no `.experts` submodule anywhere.

The sweep's `all-linear` resolver (`ml/adapters/resolve_target_modules.py`) had a
dense path (distinct leaf-name set) and an MoE path (full dotted paths of
serveable `nn.Linear`s, excluding experts/router/head). But MoE detection keyed
only on a `.experts` submodule, and the router exclusion keyed on the leaf names
`gate`/`router`. **Granite has none of those.** So it silently fell to the dense
leaf-name path and would have LoRA-wrapped the fused grouped experts, the
`router.layer`, and the Mamba SSM projections — none of which a `.pt` adapter can
serve. In practice the preflight caught this first (below), so it never trained.

## The three-part fix

### 1. Train-side resolver — commit `97044f8`

- `_is_moe_model` now also fires on `.block_sparse_moe` (not just `.experts`).
- `_moe_servable_linear_paths` excludes two more subtrees: `.block_sparse_moe`
  (grouped experts **and** `router.layer`) and `.mamba` (SSM `in_proj`/`out_proj`).

Net: LoRA lands on **attention + the dense `shared_mlp`** (both serveable), by
*full path* — necessary because `shared_mlp` and the experts share the
`input_linear`/`output_linear` leaf names, so a leaf-name set can't include one
while excluding the other. A Granite-hybrid synthetic unit fixture pins the exact
expected target set (12/12 resolver tests green, including the prior Qwen3MoE and
PhiMoE cases).

### 2. Offline preflight — commit `c219775`

`test/finetune_sweep/preflight.py` predicts a silent LoRA no-op *before* paying
for restart + train + serve, by pushing a **hardcoded synthetic leaf set** through
the fork's normalization and checking overlap against the served vLLM module
tree. It lacked granite's `shared_mlp.{input_linear,output_linear}` leaves → zero
overlap → `PRECHECK_NO_OP` skip. Adding those leaves (to both the host
`DEFAULT_LORA_TARGETS` and the mirrored in-container `targets` dict) flips granite
from skipped to run:

```
[preflight] ibm-granite/granite-4.0-h-tiny: predicted_ok=True overlap=2/13
```

This is strictly **fail-open**: `predicted_ok` is `overlap(...) > 0`, so extra
target leaves can only *add* overlap, never introduce a new false skip. Same
pattern as the earlier GLM-4 preflight fix.

### 3. LR / step-budget tune — commit `18d8017`

With the resolver fixed, granite *trained* — and revealed a new problem. The sweep
default (bf16, peak LR **3e-3**, **450** steps, LinearLR warmup→decay) did this:

- **Memorized during warmup:** loss `4.7 → 0.0003` by step ~19 (LR still ramping,
  ~1.9e-3).
- **Diverged the moment warmup ended:** at step 30 LR pinned at full **3e-3** and
  the loss **exploded to ~9** (steps 32–42), then only slowly re-converged as the
  linear decay crept down.
- **Couldn't fit the timeout anyway:** 450 steps × ~27.5s/step ≈ 206 min (the
  Mamba-2 layers are ~12× slower per step than a dense 7B on the GB10) vs a 90-min
  train-timeout → it would have hit `TRAIN_TIMEOUT` at ~step 196 with a
  mid-oscillation, non-memorized adapter.

Diagnosis: peak LR 3e-3 is too hot for this hybrid/SSM stack, and the slow decay
over 450 steps keeps it hot long after the model has already memorized.

**Fix:** peak LR **1e-3** + **90**-step budget. Result — a textbook curve, no
divergence, and it fits the timeout comfortably (~41 min):

```
step 0: 4.73   step 9: 0.86   step 17: 0.0016   step 23: 0.0   step 26–89: 0.0
```

## Two open questions, both answered YES

The design plan flagged two unknowns that only a live run could resolve:

1. **Does the dense `shared_mlp`-only adapter memorize, with routed experts off
   the `.pt`?** **Yes.** Granite runs an *always-on* dense `shared_mlp` on every
   layer, and that alone carries the memorization signal — so, unlike a
   pure-sparse MoE, **no separate-expert LoRA converter was needed** (the wall
   PhiMoE/Mixtral point at). Routed experts stay off the adapter, same as PhiMoE.
2. **Does a `shared_mlp` LoRA serve through the fork's hybrid path?** **Yes.**
   `granitemoehybrid.py` (`SupportsLoRA` + `IsHybrid` + `MambaMixer2` +
   `SupportsMambaPrefixCaching`) loaded and served the adapter with `serve_s=12.1`
   and no adapter-load error.

## Commits

| Commit | Change | Portable? |
|---|---|---|
| `97044f8` | resolver: detect `block_sparse_moe`, exclude `block_sparse_moe`+`mamba` | **Yes** — real fix, cherry-pick to a PR off `main` |
| `c219775` | preflight: add `shared_mlp` leaves (fail-open) | **Yes** — real fix, cherry-pick |
| `18d8017` | per-model LR 1e-3 / 90 steps | No — sweep-config only |
| `3695763` | record PASS in yaml + report | No — docs |

Per the lab-branch workflow, `georgi/finetune-sweep` is never PR'd whole. The two
portable fixes (`97044f8`, `c219775`) are the ones to extract into a short-lived
PR off fresh `main` when upstreaming.

## Related

- Resolver PhiMoE router-exclusion fix (`2ecf306`) — same file, the sibling MoE
  fix that this generalizes past.
- `docs/superpowers/plans/2026-07-04-granite-hybrid-lora-target-resolution.md` —
  the implementation plan this executed.
- `docs/reports/2026-07-01-model-categories.md` — granite now listed ✅ PASS.
