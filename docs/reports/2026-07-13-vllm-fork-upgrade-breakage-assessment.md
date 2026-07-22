# vLLM fork upgrade: will bumping the version break us? — breakage assessment

**Date:** 2026-07-13
**Scope:** `supermassive-intelligence/vllm-fork` (branch `georgi/finetune-sweep`, tip
`19c729c61`, version string `0.19.1.dev45+g19c729c61`) vs. current upstream
`vllm-project/vllm` **v0.25.0** (released 2026-07-11).
**Method:** read-only. Fetched the `v0.25.0` tag objects into the local `vllm/`
checkout's `upstream` remote (no working-tree or branch change) and diffed against the
fork's `v0.19.0` base and `HEAD`. Refreshes the two prior reports
(`2026-06-18-vllm-fork-upgrade-inventory.md`, `2026-07-01-vllm-fork-upstream-delta-and-upgrade-difficulty.md`)
and `docs/adr/0005-vllm-fork-adapter-layer-and-upgrade-stance.md` with current numbers.

> **Working-tree note.** `vllm/tokenformer/hybrid_adapter_manager.py` was being edited
> today (mtime 11:26) by the other agent on this branch. Nothing here touched tracked
> state or the `vllm/` working tree — all facts come from `git show`/`git diff` against
> tags. The fork's `vllm/` is an untracked sibling checkout, not a submodule.

## Answer up front

**Yes, a naïve version bump breaks the fork — but not because its differentiator is
fragile.** The durable `.pt`/Tokenformer adapter layer's dependencies on core vLLM
*all still exist in v0.25.0* (verified symbol-by-symbol below), so the layer carries
forward. The breakage is concentrated in four places, in descending order of pain:

1. **Toolchain bump** — Torch 2.10→2.11, torchvision 0.25→0.26, Transformers ≥5.5.3, and
   a **brand-new Rust build toolchain** (the `rust/` frontend did not exist at the fork's
   base). This is the hardest thing to control because it's not our code.
2. **MoE-LoRA reconciliation** — upstream rewrote `lora/layers/fused_moe.py` (+182/−336)
   and `lora/model_manager.py` (+367/−72); the fork just landed *fresh* code here
   (`moe_lora_utils.py`, the gated-2D and separate-expert converters, commits since the
   last report). This is now the single largest conflict.
3. **Serving-surface migration to Rust** — upstream v0.25 moved tool-call / reasoning
   parsing into a Rust "New Streaming Parser Engine," including `rust/.../tool/gemma4.rs`
   and `rust/.../unified/gemma4.rs`. The fork carries *Python* gemma4 parsers
   (`vllm/tool_parsers/gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py`)
   plus Anthropic/OpenAI entrypoint patches — these now collide with a moved target.
4. **Model Runner V2 is now the default for all dense models** (was opt-in on 2026-07-01).
   Lower risk than it sounds — see below — but it changes which runner file the LoRA hook
   must live in.

**Recommendation is unchanged from ADR 0005: this is a re-integration, not a rebase, and
it should be deferred / team-owned — but the gap is still widening** (0.19→0.25 now; was
0.19→0.24 twelve days ago), so the cost only grows. Do the low-cost mitigations now
(regression harness, pin discipline) so that whenever the re-integration happens it can't
silently re-break a model.

## Current state

| | |
|---|---|
| Fork version | `0.19.1.dev45+g19c729c61` |
| Fork base (merge-base with `v0.19.0`) | `2a69949bd` = **byte-identical to upstream `v0.19.0`** |
| Fork commits on top of base | **45** (was 38 on 2026-07-01) |
| Upstream latest | **v0.25.0** (2026-07-11) |
| Fork's carry-forward footprint | **63 files** changed/added vs base |

The 7 new fork commits since the last report are all MoE-LoRA serving work
(`2c62b04ed` gated 2D FusedMoE, `d77dcb29d` separate-expert, `c2d37bcd4` expert-container
rename, `19c729c61` skip-incompatible-`.pt`, plus `moe_lora_utils.py`). This *adds* to the
riskiest reconciliation area, exactly as the 2026-07-01 report predicted.

## What was — and still is — stopping the update

Nothing external blocks it; the blockers are cost and risk, all confirmed still true:

1. **It's a re-integration across 5 minor versions, not a rebase.** The fork's base
   (`v0.19.0`) and upstream diverged by **3,152 commits / 4,042 files / +672,668 −164,255**.
   The subsystems the fork touches (MoE, LoRA, model runner, entrypoints) are exactly the
   ones upstream churned hardest.
2. **The MoE-LoRA plumbing the fork targets was rewritten upstream.** The fork's converter
   splits PEFT-fused `gate_up` into the 2D FusedMoE-with-LoRA path against 0.19's layout;
   `lora/layers/fused_moe.py` is net-rewritten upstream and there's a new
   `lora/ops/triton_ops/fused_moe_lora_op.py`.
3. **Toolchain is a moving target.** Torch 2.11 / torchvision 0.26 / Transformers v5.5+
   / CUDA-13 family-specific arch targets, plus the new Rust frontend — none of which the
   fork's C++ ABI shims (`csrc/libtorch_stable/*`, pinned to Torch 2.10) were built for.
4. **Ownership / coordination.** Per ADR 0005: Naila Farooqui owns the upgrade +
   weight-loading, Kari Pulli owns gemma4/MoE, Greg Diamos owns the adapter subsystem. The
   fork's `AGENTS.md` requires a human to own and defend the change. `v0.17-upgrade` is
   stale/superseded — do not rebase onto it.

## What will break — categorized by evidence

### ✅ Survives the bump (verified present in v0.25.0)

The durable adapter layer imports these core-vLLM symbols; every one still exists at
v0.25.0 (checked with `git show v0.25.0:…`):

| Symbol / module | v0.25.0 |
|---|---|
| `lora.utils.get_adapter_absolute_path`, `get_lora_id` | present (utils.py:314, :70) |
| `lora.worker_manager.LRUCacheWorkerLoRAManager` | present (:241) |
| `lora.lora_model.LoRAModel` | present |
| `lora.peft_helper.PEFTHelper` (fields `r`, `lora_alpha`, `use_rslora`, `vllm_lora_scaling_factor`) | present |
| `model_loader.utils.process_weights_after_loading` | present (:101) |
| `model_executor.models.SupportsLoRA` | present + exported |
| `v1/worker/lora_model_runner_mixin.py` (the hook site) | present, upstream churn only **+4/−1** |

Implication: `vllm/tokenformer/*` (the ~6-file adapter subsystem) carries forward with
**no upstream conflict surface** — there's no upstream equivalent to conflict with. This
is the ADR's core thesis, still holding.

### 🔴 Will break — reconcile required

| Area | Fork files | Upstream churn v0.19→v0.25 | Risk |
|---|---|---|---|
| **MoE-LoRA converter** | `lora/layers/fused_moe.py`, `lora/model_manager.py`, `lora/moe_lora_utils.py` (fork-new) | `fused_moe.py` +182/−336, `model_manager.py` +367/−72 | **Highest.** Fork's newest code vs. upstream's hardest rewrite. Re-verify `_stack_moe_lora_weights_gated` against the new FusedMoE+LoRA layout, don't re-apply. |
| **LoRA base layers** | `model_executor/layers/linear.py`, `vocab_parallel_embedding.py` | linear.py +316/−157 | High. LoRA wrapping of packed modules changed. |
| **Engine arg plumbing** | `engine/arg_utils.py` (adds `enable_tokenformer`) | +494/−126 | High merge-conflict. Additive field on a heavily-churned file. |
| **Engine core wiring** | `v1/engine/core.py`, `async_llm.py`, `core_client.py` | core.py +357/−94; async_llm +47/−11 | Medium–High. Fork adds its own +357/−94 on core.py — double-diverged. |
| **Serving entrypoints** | `entrypoints/anthropic/serving.py`, `openai/api_server.py`, `responses/serving.py`, `serve/render/serving.py`, `entrypoints/utils.py` | api_server.py +123/−95; whole tree +11.5k/−7.7k | Medium–High. The ScalarLM Anthropic endpoint + serve surface. Under-emphasized in prior reports. |
| **gemma4 parsers** | `tool_parsers/gemma4_tool_parser.py`, `reasoning/gemma4_reasoning_parser.py`, `parser/abstract_parser.py`, `reasoning/abs_reasoning_parsers.py` | Upstream moved parsing to **Rust** (`rust/.../gemma4.rs`) | Medium. Target moved languages; likely shed the Python versions for upstream's Rust ones. |
| **Removed `LoRAConfig` attrs** | `exaone_moe.py:268/530`, `exaone_moe_mtp.py:48` | `lora_extra_vocab_size`, `lora_vocab_padding_size` **gone** from `config/lora.py`; `max_lora_rank` now a typed `MaxLoRARanks` field | Low (mechanical) **and sheddable** — see below. |

### 🟡 Changed but lower-risk than feared

- **Model Runner V2 default flip.** v0.25 makes MRv2 the default for all dense models.
  But the MRv2 `gpu/` tree **already existed at the fork's v0.19.0 base** (51 files then,
  69 now), the fork already routes `load_lora_model` through both `gpu_model_runner.py`
  *and* `gpu/model_runner.py`, and the hook site `lora_model_runner_mixin.py` barely moved
  upstream (+4/−1). New wrinkle: `gpu/mm/lora.py` is **new** (multimodal LoRA in MRv2) —
  needs a hook if MM adapters must work under MRv2. Net: re-apply hooks into rewritten
  runner files, plus one new MM hook. Bounded, not a flag-day rewrite.

### 🟢 Now sheddable (delete fork file, adopt upstream) — confirmed at v0.25.0

Upstream `registry.py` at v0.25.0 natively registers everything the fork hand-carries:
`Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`,
`Gemma4MTPModel`, `ExaoneMoEForCausalLM`, `ExaoneMoeMTP`, `TarsierForConditionalGeneration`.
So `gemma4.py`, `gemma4_mm.py`, `gemma4_utils.py`, `exaone_moe.py`, `exaone_moe_mtp.py`,
`tarsier.py` can be **deleted** in favor of upstream — which also disposes of the removed-attr
crashes (row above) for free. Caveat: re-point the adapter layer's `normalize_lora_key`
rules at upstream's module tree (ADR 0005 point 2), and note **Tarsier2 is pinned to ≤0.23
in upstream's registry** ("last version with Transformers v4") — verify it's still viable
if it's a sweep target.

### Toolchain deltas (the genuinely hard part — not our code)

| | fork base v0.19.0 | upstream v0.25.0 |
|---|---|---|
| torch | 2.10.0 | **2.11.0** |
| torchvision | 0.25.0 | 0.26.0 |
| transformers | (image caps `<5.13`) | **≥5.5.3** required |
| python | — | `>=3.10,<3.15` |
| build | setuptools only | **+ setuptools-rust ≥1.9.0, Rust toolchain** (new `rust/` frontend) |
| CUDA archs | — | `7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0` + family-specific `10.0f/12.0f` from CUDA 12.9/Blackwell |

Notes tying to known footguns (from memory / runbooks):
- The fork's `csrc/libtorch_stable/*` ABI shims were built for Torch 2.10 — re-verify against 2.11.
- The `transformers<5.13` image cap (commit `0b5a4e7`) and v0.25's `transformers>=5.5.3`
  floor must be reconciled in the ScalarLM Docker image, not just the fork.
- The Rust toolchain is a **new base-image requirement** — the prebuilt vast/Spark images
  won't have it; expect a build-environment change, not just a pip bump.
- CUDA arch handling moved to family-specific targets (10.0f/12.0f) — the existing
  sm_90-missing / GB10 sm_120 kernel footguns (see `vast-baremetal-serving-footguns`,
  `spark-baseline-generate-timeout`) will need re-validation, not carry-over.

## Actions to catch and mitigate the breakage

### Catch it (cheap, do now — independent of when the upgrade happens)

1. **Build the model-load regression harness the 2026-06-18 report specified.** A `pytest`
   that, per fork-shipped model class, instantiates the engine and runs `load_weights`
   with `enable_lora=True` on the tiny-random fixtures (seconds, no downloads). The
   sweep's fixtures *become* the upgrade's regression suite. There are already
   `tests/tokenformer/test_*.py` and `tests/lora/test_moe_lora_utils.py` in the fork — extend,
   don't start from scratch.
2. **Symbol-drift CI check.** A tiny test that imports every core-vLLM symbol the adapter
   layer depends on (the ✅ table above) and asserts presence + signature. This turns a
   future rebase's "silent AttributeError at model load" into a red unit test. All 7
   symbols pass today, so it's green now and only fails when a bump actually removes one.
3. **Pin-provenance guard.** A test asserting the fork's `torch`/`transformers`/CUDA-arch
   pins match the running Docker image (the live `docker exec` container is the source of
   truth, per `image-transformers-safetensors-cap`) — catches the "editing requirements
   busts the cache → transformers jumps a minor" class of failure before it crash-loops
   the API server.
4. **End-to-end memorize gate on real hardware.** Keep the existing discipline (per
   memory `validate-on-vast-before-pr`): a genuine memorize-PASS on cuda-vast H200/B200 for
   at least one dense, one MoE, and one multimodal model before any upgrade lands. Unit
   tests can't catch the serving-path and adapter-normalization regressions.

### Mitigate / sequence the upgrade (when it's greenlit)

5. **Do the sheddable deletions first, on the current base** — delete custom
   gemma4/exaone/tarsier files, adopt upstream's, re-point `normalize_lora_key`. Shrinks
   the fork's 63-file footprint and disposes of the removed-attr crashes before the hard
   part. Low risk, reversible, reduces later conflict surface.
6. **Spike the MoE-LoRA reconciliation in isolation** (the 2026-07-01 suggested next step,
   still the best predictor): in a throwaway worktree, diff the fork's
   `moe_lora_utils.py` + `_stack_moe_lora_weights_gated` against upstream v0.25's
   `lora/layers/fused_moe.py` and `lora/ops/triton_ops/fused_moe_lora_op.py`. If that
   converts cleanly, "Medium–Large" holds; if not, the whole upgrade is Large.
7. **Stage the toolchain separately from the code rebase.** Bring the ScalarLM image to
   Torch 2.11 + Transformers ≥5.5.3 + Rust toolchain and prove the *current* fork still
   builds/serves on it, before layering the code re-integration on top. Don't debug two
   moving targets at once.
8. **Rebase in a worktree/branch off fresh `main`, never on `georgi/finetune-sweep`** (per
   `finetune-sweep-lab-branch-workflow`): the lab branch stays the validation harness; the
   upgrade is a separate short-lived branch. vLLM-fork PRs target
   `supermassive-intelligence/vllm-fork main`, never upstream
   (`vllm-fork-prs-never-target-upstream`).
9. **Consider waiting for one more upstream cycle regardless.** v0.25's headline items
   (MRv2-default, Rust parser engine, Transformers-backend-as-fast-as-native) are
   mid-transition. Rebasing onto a settling target front-loads churn you'd redo. The gap
   grows ~1 minor / ~2 weeks; the marginal cost of waiting one cycle is small relative to
   catching the toolchain mid-flight.

## Bottom line

- **Current:** fork on v0.19.0 base (+45 commits); upstream at v0.25.0. Gap = 3,152
  commits.
- **What breaks:** toolchain (Torch 2.11 + new Rust build), MoE-LoRA converter, gemma4
  parser/entrypoint serving surface (now Rust upstream), and re-hooking MRv2 — in that
  order. The durable `.pt`/Tokenformer adapter layer does **not** break (all its core-vLLM
  deps survive v0.25.0).
- **What to do now:** the four cheap "catch it" items (regression harness, symbol-drift CI,
  pin guard, hardware memorize gate) — none require doing the upgrade, all make the
  eventual upgrade safe. Defer the re-integration itself to a team-owned, staged effort;
  do the sheddable deletions and the MoE-LoRA spike first to size the real cost.
