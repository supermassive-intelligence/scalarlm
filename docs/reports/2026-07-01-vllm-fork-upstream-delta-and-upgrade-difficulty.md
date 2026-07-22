# vLLM fork vs. upstream: current delta and upgrade difficulty

**Date:** 2026-07-01
**Scope:** `supermassive-intelligence/vllm-fork` (`main`, tip `6c8cc386c`) vs. upstream
`vllm-project/vllm`. Follow-up to `docs/adr/0005-vllm-fork-adapter-layer-and-upgrade-stance.md`
and `docs/reports/2026-06-18-vllm-fork-upgrade-inventory.md` — this refreshes the sizing with
a real `git diff` against upstream instead of the earlier grep-based estimate.

Method: added a temporary `upstream` remote (`vllm-project/vllm`) in the local `vllm/`
checkout, fetched the `v0.19.0`/`v0.24.0` tags, and diffed. `vllm/` is not a submodule of
`scalarlm` (it's an untracked sibling checkout, hence `?? vllm/` in `git status`), so this is
local-only and doesn't touch any tracked state.

## Answer up front

**Still 0.19.0-based; upstream is now at 0.24.0 — one minor version further behind than the
ADR's "0.19 vs 0.23" snapshot two weeks ago.** The fork's own base commit
(`27d57e2e3^` = `2a69949bd`) is byte-identical to upstream's `v0.19.0` tag — confirmed by
`git merge-base`. Difficulty verdict is unchanged from ADR 0005: **not a rebase, a
re-integration.** The two things that make it tractable are also confirmed and slightly
stronger than before:

1. The fork's actual differentiator — the `.pt`/Tokenformer adapter layer — is untouched by
   upstream (those files don't exist there), so it survives a rebase intact.
2. The "custom" model classes upstream absorbed are now *more* absorbed, not less: Gemma4
   (3 variants), ExaoneMoE, and Tarsier are all present in upstream's `registry.py` at 0.24.0.
   That's the ADR's "sheddable" prediction paying off — fewer files to hand-carry forward.

## Scale of the gap

| | |
|---|---|
| Fork base | `v0.19.0` (released 2026-04-02) |
| Upstream latest | `v0.24.0` (released 2026-06-27) |
| Time span | ~12 weeks |
| Minor versions behind | 5 (`0.19 → 0.20 → 0.21 → 0.22 → 0.23 → 0.24`) |
| Commits `v0.19.0..v0.24.0` | 2,594 |
| Files changed | 3,665 |
| Lines | +583,953 / −143,305 |
| Fork's own commits since the squash | 38 (37 files, +3,958/−89) |

Diff by subsystem (upstream, `v0.19.0..v0.24.0`), heaviest first:

| dir | files | + | − |
|---|---|---|---|
| `model_executor` | 603 | 86,678 | 31,100 |
| `kernels` | 27 | 58,506 | 27,866 |
| `v1` | 256 | 37,848 | 11,135 |
| `models` (new top-level tree) | 57 | 23,685 | 0 |
| `distributed` | 95 | 20,346 | 7,005 |
| `entrypoints` | 142 | 11,542 | 7,678 |
| `lora` | 24 | 2,534 | 542 |
| `config` | 25 | 2,333 | 445 |

## Confirmed structural refactors in the gap (not just churn)

All four risks the ADR flagged qualitatively are real and grep-confirmed in the commit log:

- **MoE refactor** — 27+ commits tagged `[MoE Refactor]`, including a full
  `FusedMoE`/`MoERunner` inversion. Directly relevant: `vllm/lora/ops/triton_ops/
  fused_moe_lora_op.py` is a **new 792-line file** upstream, and `vllm/lora/layers/
  fused_moe.py` was rewritten (180+/335−, net shrink). This is exactly the code path the
  fork's newest commit (`6c8cc386c`, "serve gated 2D FusedMoE expert LoRA from .pt adapters")
  hand-built against 0.19's MoE-LoRA plumbing — expect this to be the single largest
  reconciliation item, not a mechanical patch.
- **Model Runner V2** — real and shipping, but **not yet a flag day**: `VLLM_USE_V2_MODEL_RUNNER`
  still defaults `False` globally; upstream is rolling MRv2 on *per-architecture* via an
  "oracle" (`ae4f59f0e` enabled it for qwen3-dense by default, `88a9cdd43` for GraniteMOE,
  `b53b1c7ff` for quantized models). Since qwen3/qwen3-moe are sweep targets the fork has
  hand-patched, some of those models may already be routed through the new
  `vllm/v1/worker/gpu/` tree (new files: `gpu/lora_utils.py`, `gpu/mm/lora.py`,
  `gpu/spec_decode/*`) by default post-rebase — bypassing the old `lora_model_runner_mixin.py`
  path the fork touches. `gpu_model_runner.py` itself changed +1,212/−639.
- **Toolchain bump** — confirmed: Torch 2.11 landed (ROCm bumped in `39cb9bf29`), work toward
  2.12/2.13 is in flight; CUDA 13.0 wheel builds are real (`cfd2573f2`, `140dc2ec3`); Transformers
  v5.10+ compat fixes appear repeatedly (Cohere2MoE, Voxtral, MiniCPM). None of this is
  finished/stable upstream either — it's a moving target, which argues for waiting rather
  than chasing it mid-flight.

## What's unchanged from ADR 0005 (still the right call)

- **Adapter layer is durable, keep investing there.** `vllm/tokenformer/*`
  (`hybrid_adapter_manager.py`, `tokenformer_model_manager.py`, `tokenformer_surgeon.py`,
  `lora_from_pt.py`, `adapter_format.py`) has zero upstream equivalent — a rebase carries it
  forward wholesale with no merge conflict against upstream, only against the fork's own prior
  commits.
- **Custom model classes are sheddable, now more confidently.** `gemma4.py`/`gemma4_mm.py`
  exist upstream with 614+/137− and 531+/162− of independent upstream churn respectively —
  taking the fork's hand-maintained versions forward would mean re-deriving that churn by
  hand. `gemma4_utils.py` (292 lines, fork-only) has no upstream counterpart at all — likely
  dead weight once upstream's native Gemma4 is adopted. `exaone_moe.py`, `exaone_moe_mtp.py`,
  and `tarsier.py` are all upstream-native now too (confirmed via `registry.py`, which grew to
  reference `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`,
  `Gemma4UnifiedForConditionalGeneration`, `ExaoneMoEForCausalLM`, `Gemma4MTPModel`,
  `ExaoneMoeMTP`, `TarsierForConditionalGeneration`, `Tarsier2ForConditionalGeneration`).
  Net effect: the rebase can likely **delete** these fork files and take upstream's, rather
  than merge them.
- **The core touch-points stay small.** `config/lora.py` (+17/−1), `lora_model_runner_mixin.py`
  (+4/−1), `async_llm.py` (+50/−11) are all still narrow diffs against upstream — the fork's
  additions there (`enable_tokenformer` field, hybrid-manager hooks) are additive and low
  merge-conflict risk on their own. `v1/engine/core.py` is the outlier at +357/−94 and is worth
  a closer read before committing to the estimate below.

## Updated sizing

Directionally the same as ADR 0005 with one revision: the gap grew (0.19→0.24 vs. 0.19→0.23),
and the MoE-LoRA surface got materially riskier because the team's newest fork commit
(`6c8cc386c`) landed *after* the ADR was written, adding fresh fork-owned code in exactly the
area (`fused_moe_lora_op.py`-equivalent) that upstream rewrote hardest.

- **Adapter layer carry-forward:** Small — no upstream conflict surface.
- **Drop custom Gemma4/ExaoneMoE/Tarsier, take upstream's:** Small–Medium — needs the
  `normalize_lora_key` adapter-layer logic re-pointed at upstream's module tree (per ADR 0005
  point 2), but removes files rather than merging them.
- **MoE-LoRA reconciliation (`fused_moe_lora_op.py`, `layers/fused_moe.py`,
  `model_manager.py` +365/−38):** Medium–Large — this is now the critical path; the fork's
  `_stack_moe_lora_weights_gated` gate-up-splitting converter needs to be re-verified against
  upstream's rewritten FusedMoE+LoRA plumbing, not just re-applied.
- **Model Runner V2 exposure check:** Medium — needs an explicit check of which sweep models
  (qwen3 in particular) the MRv2 oracle now defaults to, since that changes which runner file
  the LoRA mixin patch actually needs to land in.
- **Toolchain bump (Torch/CUDA/Transformers):** unbounded until upstream's own transition
  settles — still the strongest argument to defer.
- **Own-commit rebase cost:** 38 commits / 37 files / ~4k lines of fork-only work to carry
  through the rebase — manageable, not the bottleneck.

**Net: no change to the ADR's "defer the 0.19→0.2x rebase" recommendation.** If anything, the
freshly-landed MoE-LoRA work (`6c8cc386c`) is a reason to defer a *bit* longer — rebasing now
would immediately require re-deriving that commit against upstream's already-diverged MoE-LoRA
code, whereas waiting lets the sweep validate the current approach first and gives upstream's
own MRv2/toolchain transition more time to stabilize.

## Suggested next step

Before scoping an actual rebase: get Naila/Kari/Greg sign-off per ADR 0005, then land a small
spike that diffs just `vllm/lora/layers/fused_moe.py` and the new
`vllm/lora/ops/triton_ops/fused_moe_lora_op.py` against the fork's `_stack_moe_lora_weights_gated`
converter — that's the one piece of this delta that's both large and newly fork-owned, so it's
the best predictor of whether "Medium–Large" above is optimistic.
