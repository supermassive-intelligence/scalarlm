# vLLM fork migration plan: v0.19.0 → v0.25.0

**Date:** 2026-07-13
**Companion to:** `docs/reports/2026-07-13-vllm-fork-upgrade-breakage-assessment.md`
(evidence + breakage inventory), `docs/adr/0005-vllm-fork-adapter-layer-and-upgrade-stance.md`,
and `docs/adr/0010-pin-vllm-0.25.0-as-immovable-rebase-base.md`.
**Status:** accepted orchestration (grilled + reshaped 2026-07-14). Georgi executes all
phases as a fork owner; sign-off is once, at the Phase-6 PR (Kari / Greg).

## Decisions log (2026-07-14 grilling)

These supersede the first-draft phrasing throughout the phases below:

1. **Ownership / sign-off.** Georgi drives all 6 phases as a fork owner. Sign-off happens
   **once, at the Phase-6 PR** — no per-phase reviewers.
2. **Target = `v0.25.0`, pinned and immovable** for spike *and* re-integration (ADR 0010).
   Re-integration base is the **`v0.25.0` tag**, not the fork's v0.19-based `main`; `main`
   is only the PR target.
3. **Branching.** Phases 0/1/3 on a new branch off **`georgi/finetune-sweep @ 19c729c61`**
   (lab work is landed, tree clean). No throwaway-worktree ceremony.
4. **Phase 2 is a front-loaded hard gate**, not a parallel track — retire/escalate the
   Rust/toolchain "unbounded" risk *before* spending on 0/1/3.
5. **Phase 0 = meta-device structural harness** (reuse `preflight.py`), CPU-only. It is the
   oracle for symbol-drift + tree/normalization, **not** for MoE numerics or weight-load
   (Phase 5 owns those — georgi has no local GPU).
6. **Phase 1 = inventory verdict, no deletions on v0.19.** Actual adopt/re-apply → Phase 4.
7. **Phase 3 = difficulty-triage** with a *semantic-fit* bar (not conflict-free merge).
8. **Phase 4 parsers: shed to upstream's Rust** (committed).

### Factual corrections the grilling caught (verified in the fork tree)

- **4 of 6 "shed candidates" are byte-identical to stock v0.19** (`gemma4_utils.py`,
  `exaone_moe.py`, `exaone_moe_mtp.py`, `tarsier.py`) — nothing to shed there. Only
  `gemma4.py` (+27/−11, NVFP4 expert routing — **subsumed by v0.25**, sheddable) and
  `gemma4_mm.py` (+10/−2, adds `SupportsLoRA`+`SupportsTokenformer` — **carry-forward**)
  hold real fork delta.
- **`SupportsTokenformer` spans 5 model classes** (`llama, qwen3, qwen3_moe, gemma3,
  gemma4_mm`), all heavily churned upstream — re-applying the mixin is *medium*, not cheap.
- **Rust is a mandatory v0.25 build dep** (`setuptools-rust>=1.9.0` in `[build-system]`),
  but the fork doesn't touch the Rust engine, so the **precompiled abi3 `.so`** is a viable
  escape hatch from putting a toolchain on the vast/Spark base image.
- **Two** new MoE-LoRA triton ops upstream (`fused_moe_lora_op.py` **and**
  `fused_moe_lora_fp8_op.py`), not one. `origin/gemma4-quantized-moe-main` likely owns the
  gemma4 NVFP4 delta — check during Phase-1 inventory.

## Governing principle

**Never debug two moving targets at once, and put every step behind a green regression
harness so nothing silently re-breaks.** The 3,152-commit gap becomes a series of small,
reversible, individually-verifiable moves. Gates between phases are hard: a red gate stops
the pipeline for reassessment rather than piling the next phase on top.

## Pipeline overview

```
Phase 2        Phase 0        Phase 1         Phase 3        Phase 4              Phase 5      Phase 6
TOOLCHAIN   →  SAFETY NET  →  INVENTORY    →  MoE SPIKE  →   RE-INTEGRATION   →  VALIDATE  →  LAND
FRONT GATE     (meta-device   (verdict only,  (go/no-go,     (base = v0.25.0      (hardware)   (fork PR,
(retire/       harness,       no deletions,   difficulty     tag; re-apply                     sign-off
 escalate      CPU, on 0.19)  on 0.19)        triage)        carry-forward delta)              here)
 Rust risk)
```

**Sequencing:** Phase 2 runs **first** as a hard gate (decision #4). Then Phase 0 (harness)
gates everything downstream; Phase 1 (inventory) and Phase 3 (spike) can run once Phase 0 is
green. Phase 3 gates Phase 4; Phase 4 is internally dependency-ordered. All of 0/1/3 live on
one branch off `georgi/finetune-sweep @ 19c729c61`.

**Deliberate abort points:** Phase 2 (base image can't host the toolchain / abi3 `.so` →
that's a base-image project first) and Phase 3 (MoE-LoRA won't reconcile semantically →
defer and escalate). Both surface the expensive unknowns *before* the re-integration cost is
sunk — which is exactly why Phase 2 is pulled to the front.

---

## Phase 2 — Toolchain front gate (runs first)

Pulled ahead of everything else (decision #4): it's the single biggest *not-our-code*
unknown, and the cheapest thing that can kill the whole migration. Retire or escalate it
before any other effort is sunk.

Spike, in order:

1. **Inspect the base image** (`nvcr.io/nvidia/pytorch:26.01-py3`): does it ship a Rust
   toolchain (`cargo`)? Does Torch 2.11 / torchvision 0.26 reconcile with the
   `transformers<5.13` image cap vs v0.25's `>=5.5.3` floor?
2. **Test the abi3 escape hatch:** does upstream v0.25's precompiled `_rust_*.abi3.so`
   (the Rust streaming-parser engine — the fork does **not** modify it) load on the
   vast/Spark base image? PyO3 abi3 is Python-version-forward-compatible, so this lets you
   build the fork's Python+CUDA from source and *drop in* the prebuilt `.so` — no Rust
   toolchain on the base image.
3. **Decide precompiled-`.so` vs build-from-source** from that evidence.
4. **Prove the CURRENT (v0.19) fork still builds and serves** on the new-toolchain image
   (one dense model, one real memorize-PASS). **No code rebase here.**

| | v0.19.0 (base) | v0.25.0 (target) |
|---|---|---|
| torch | 2.10.0 | 2.11.0 |
| torchvision | 0.25.0 | 0.26.0 |
| transformers | image caps `<5.13` | `>=5.5.3` required |
| python | — | `>=3.10,<3.15` |
| build | setuptools | **+ setuptools-rust ≥1.9.0 (MANDATORY in `[build-system]`); abi3 `.so` escape hatch** |
| CUDA archs | — | `7.5..12.0` + family-specific `10.0f/12.0f` (CUDA 12.9/Blackwell) |

**Desk portion done (2026-07-14) — see `docs/reports/2026-07-14-phase2-toolchain-spike-findings.md`.**
Verdict: **bounded, not "unbounded."** Concrete change set: base image `26.01-py3` →
**`26.03-py3`** (torch 2.11.0a0, CUDA 13.2); `VLLM_USE_PRECOMPILED_RUST=1` (no `cargo` needed —
NGC ships none); lift the `transformers<5.13` cap (safetensors already 0.8.0); the build
already uses `--no-build-isolation` so the alpha-torch pin is a non-issue. One genuine code
risk: `csrc/libtorch_stable/*` vs torch 2.11. Remaining = the GPU build+serve, on remote/Spark.

**Gate (abort point):** if the base image can host neither the abi3 `.so` nor a toolchain,
the migration is a base-image project first — escalate rather than proceed. Otherwise: green
= current fork memorize-PASSes on the new-toolchain image.

## Phase 0 — Safety net (meta-device harness, on the current v0.19 base)

Highest-leverage code work; independent of the upgrade. Build before touching any version.
Runs **CPU-only** — georgi has no local GPU, so this is deliberately not a weight-loading
test (decision #5).

- **Meta-device model-load regression harness** — per fork-shipped model class, build the
  `nn.Module` tree on `torch.device("meta")` via
  `model_loader.utils.initialize_model` with `load_format="dummy"` (no weights, no GPU, no
  downloads — seconds), then run the **real two-pass** `normalize_lora_key` against the live
  tree and assert the normalized adapter paths resolve. **Reuse `preflight.py`'s existing
  meta-tree machinery** (`test/finetune_sweep/preflight.py:150–177`) — the sweep's preflight
  *is* the regression harness. This catches symbol-drift (a moved core symbol makes
  `initialize_model` throw) and model-tree/normalization mismatches. It is **NOT** the oracle
  for MoE-LoRA tensor numerics or real weight loading — those are Phase 5 (hardware). Extend
  the existing key-normalization unit tests in `tests/tokenformer/*` and
  `tests/lora/test_moe_lora_utils.py` alongside it.
- **Symbol-drift assertion** — import every core-vLLM symbol the adapter layer depends on
  and assert presence + signature. Green today; goes red the moment a bump removes one:
  - `lora.utils.get_adapter_absolute_path`, `get_lora_id`
  - `lora.worker_manager.LRUCacheWorkerLoRAManager`
  - `lora.lora_model.LoRAModel`
  - `lora.peft_helper.PEFTHelper` (`r`, `lora_alpha`, `use_rslora`, `vllm_lora_scaling_factor`)
  - `model_loader.utils.process_weights_after_loading`
  - `model_executor.models.SupportsLoRA`
- **Pin-provenance guard** — assert the fork's torch/transformers/CUDA-arch pins match the
  live `docker exec` container (the running container is the source of truth, not a stale
  `:latest`).

**Built + green 2026-07-14** (branch `georgi/vllm-0.25-migration`):
`vllm/tests/tokenformer/test_symbol_drift.py` (9 symbol assertions) +
`test_model_load_regression.py` + `_meta_harness.py`, run via
`test/finetune_sweep/run_phase0_harness.sh` (bind-mounts current fork python over the
built cray image → CPU, ~1 min, no recompile). Result: **13 passed, 4 skipped** — dense
(llama, qwen2, qwen3), MoE (qwen3-moe) build on meta + normalization resolves. Findings:
(1) the harness immediately caught the `lora_config` `NameError` in the *stale* 2026-06-11
CPU image that the current branch already fixes — the oracle works; (2) MoE build needs the
EP group re-created per model (dense groups are wrong for MoE), so the harness tears down +
reinits model-parallel per model; (3) fixture gaps: gemma3/gemma4/exaone-moe/tarsier are
explicitly `skip`ped (no tiny-random fixture yet) — fill before relying on full coverage.

**Gate:** harness green on the untouched fork. Every later step now has a truth oracle —
**scoped to symbol-drift + tree/normalization**, not MoE numerics.

## Phase 1 — Inventory verdict (no deletions, on v0.19)

**No code changes.** Produce the per-file **shed / carry / stock** verdict that feeds
Phase 4. The first-draft "delete 6 custom files on v0.19" is not realizable: 4 of the 6 are
byte-identical to stock v0.19, and the sheddable one can only be adopted against the *v0.25*
tree (it imports v0.25 APIs). Verdicts (verified in the fork tree, 2026-07-14):

| File | vs stock v0.19 | Verdict |
|---|---|---|
| `gemma4_utils.py`, `exaone_moe.py`, `exaone_moe_mtp.py`, `tarsier.py` | 0/0 — identical | **Stock** — adopt v0.25's version as-is at Phase 4; nothing to shed now |
| `gemma4.py` | +27/−11 (NVFP4 expert routing) | **Shed** — v0.25 independently reinvented it (`.moe.experts` regex, WeightsMapper). Confirm `origin/gemma4-quantized-moe-main` isn't diverging |
| `gemma4_mm.py` | +10/−2 (`SupportsLoRA`+`SupportsTokenformer`) | **Carry-forward** — `SupportsTokenformer` is fork-only; re-apply on v0.25's class |

Also inventory the `SupportsTokenformer` mixin's full reach — **5 model classes** (`llama`,
`qwen3`, `qwen3_moe`, `gemma3`, `gemma4_mm`), all carry-forward — and note the removed-attr
crashes (`lora_extra_vocab_size`, `lora_vocab_padding_size`) that adopting v0.25's classes
disposes of for free. Caveat: **Tarsier2 is pinned ≤0.23 in upstream's registry** — verify
viability if it's a sweep target.

**Gate:** a written verdict table; no tree change, harness stays green trivially.

## Phase 3 — MoE-LoRA spike (go/no-go, difficulty triage)

On the migration branch, diff the fork's `moe_lora_utils.py` (fork-authored, **no upstream
merge base**) + `_stack_moe_lora_weights_gated` (in `lora/model_manager.py`, upstream
+367/−72) against upstream v0.25's rewritten `lora/layers/fused_moe.py` (+182/−336) and the
**two** new triton ops (`lora/ops/triton_ops/fused_moe_lora_op.py` **and**
`fused_moe_lora_fp8_op.py`). This is the freshest fork code against upstream's hardest
rewrite — the best single predictor of the whole cost.

This is a **difficulty triage, not a correctness proof**: the Phase-0 harness is meta-blind
to MoE tensor numerics, so only Phase 5 (hardware) proves the converter. Exit = a written
reconciliation sketch + a **clean / medium / fights** verdict.

**Decision gate** — "clean" means *semantic fit with the new FusedMoE plumbing*, **not**
"merged without conflicts":
- Converts cleanly → upgrade is **Medium–Large**, proceed to Phase 4.
- Fights the new FusedMoE plumbing → upgrade is **Large**; formally defer and escalate
  rather than discover it mid-rebase.

## Phase 4 — Re-integration

**Base = the `v0.25.0` tag** (re-apply the carry-forward delta on top of it), *not* the
fork's v0.19-based `main`; `main` is only the eventual PR target (decision #2, ADR 0010).
Never `georgi/finetune-sweep` — that stays the validation harness. Dependency order:

1. **Carry the adapter layer forward.** Two parts, different costs:
   - `vllm/tokenformer/*` subsystem — **zero upstream conflict surface, cheap.**
   - **`SupportsTokenformer` mixin re-application across 5 heavily-churned model classes**
     (`llama, qwen3, qwen3_moe, gemma3, gemma4_mm`) — **medium, not cheap.** These are
     carry-forward delta onto v0.25's rewritten classes.
2. **Re-apply the narrow core hooks** — `config/lora.py` (`enable_tokenformer`),
   `engine/arg_utils.py` plumbing, `v1/worker/lora_model_runner_mixin.py` (barely moved
   upstream, +4/−1), `v1/engine/async_llm.py`. Additive, low-conflict.
3. **Reconcile MoE-LoRA** using the Phase-3 result — re-verify against the new plumbing,
   don't blindly re-apply. Touches `lora/layers/fused_moe.py`, `lora/model_manager.py`,
   `lora/moe_lora_utils.py`.
4. **Re-hook Model Runner V2** — now default for dense models. The `gpu/` tree already
   existed at the base and both runners go through the mixin, so this is re-applying hooks
   into rewritten runner files **plus** wiring the *new* `gpu/mm/lora.py` for multimodal
   adapters.
5. **Reconcile the serving surface** — `entrypoints/*` (Anthropic endpoint, OpenAI
   api_server, responses, render). **Shed the fork's Python gemma4 parsers to upstream's
   Rust versions** (`rust/.../tool/gemma4.rs`, `rust/.../unified/gemma4.rs`) — committed
   (decision #8): they subclass generic bases with no ScalarLM-specific logic, and the Rust
   `.so` is already loaded (Phase 2). Verify behavioral equivalence against the Anthropic
   endpoint at Phase 5.

Work **model-class by model-class**, running the Phase-0 harness after each step. The
harness is what makes a giant rebase feel like small, checkpointed moves.

## Phase 5 — Hardware validation

Full sweep on vast/Spark: genuine memorize-PASS for at least one **dense**, one **MoE**,
and one **multimodal** model (per the `validate-on-vast-before-pr` discipline). Unit tests
can't catch adapter-normalization or serving-path regressions. Watch the known footguns:
stale `VLLM::EngineCore` pinning VRAM, sm_90/GB10 sm_120 kernel builds under the new
family-specific CUDA arch targets.

## Phase 6 — Land

Fork PR targeting **`supermassive-intelligence/vllm-fork main`** (never upstream
`vllm-project/vllm` — check the base-repo dropdown). Georgi owns and defends the change;
**this is the single sign-off point** — Kari Pulli (gemma4/MoE) and Greg Diamos (adapter
subsystem) review the PR. No per-phase reviewers before this. The fork's `AGENTS.md`
requires a human to own the change and run tests. `v0.17-upgrade` is stale/superseded; do
not rebase onto it.

---

## Effort summary

| Phase | Effort | Blocks on | Reversible? |
|---|---|---|---|
| 2 Toolchain front gate | Medium (unbounded if base image hosts neither abi3 `.so` nor toolchain) | — (**runs first**) | Yes (image tag) |
| 0 Safety net (meta-device) | Small–Medium | Phase 2 green | n/a (additive) |
| 1 Inventory verdict | Small (no code change) | Phase 0 green | n/a (doc only) |
| 3 MoE spike | Small (difficulty triage) | Phase 0 harness | n/a (branch, disposable) |
| 4 Re-integration | Medium–Large (gated by Phase 3) | Phases 0,1,2,3 | Branch off `v0.25.0` tag, not lab branch |
| 5 Validation | Medium | Phase 4 | n/a |
| 6 Land | Small | Phase 5 green | PR review (Kari/Greg) |

## Do-now recommendation

**Start with Phase 2 — the toolchain front gate** (decision #4). It's the single biggest
not-our-code unknown and the cheapest thing that can kill the migration; retiring or
escalating it first stops you sinking effort into a doomed base image. Then Phase 0's
meta-device harness (valuable regardless of when the migration happens), then the Phase-1
inventory and Phase-3 spike. All four run on one branch off
`georgi/finetune-sweep @ 19c729c61` and convert "will the upgrade break us" into a sized,
gated, owner-driven plan — without committing to the full re-integration.
