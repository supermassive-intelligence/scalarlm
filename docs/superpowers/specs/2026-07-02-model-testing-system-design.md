# Automated Model Testing System — Design Specification

**Date:** 2026-07-02
**Status:** Draft (design approved in brainstorm; awaiting sign-off on this written spec)
**Branch:** `georgi/finetune-sweep` (personal lab branch — see the sweep workflow memory)

## Overview

An automated pipeline that discovers newly-released Hugging Face models and
determines, for each, whether it is **(1) servable via scalarlm/vLLM** and
**(2) fine-tunable via Megatron** — without wasting GPU hours re-testing
architectures already known to pass.

The system wraps the existing `finetune-sweep` harness
(`test/finetune_sweep/run_finetune_sweep.py`) with three new pieces:

1. A **Tracker** that polls the HF Hub for new models.
2. A **SQLite state store** recording per-model and per-architecture status.
3. A **Decision engine** that deduplicates by architecture (`model_type`) and
   lineage (`base_model`), queuing only models that can teach us something new.

Testing is organized as a **cost-increasing hierarchy** (Static → Smoke → Full):
each tier is cheap enough to gate the next, so most models terminate at Tier 1
for free.

The full sweep (Tier 3) is the *already-built* harness and is out of scope to
re-design here — this spec covers the tracker, the store, the decision engine,
and the one new runner mode (Smoke) the tiers require.

---

## 1. Testing Hierarchy

Three tiers. A convergence/"is it actually learning" check (originally floated as
T2.5) is **deferred to a later version** — keep the first version lean.

| Tier | Name | Trigger | Goal | Cost | Outcomes |
|------|------|---------|------|------|----------|
| **T1** | Static Analysis | New model appears in the HF feed | Is this architecture already known? | ~0 (API + `config.json` fetch) | `SKIP` / implicit-pass, or `PROCEED` to T2 |
| **T2** | Smoke Test | New `model_type`, or a `base_model` signal that warrants a boot check | Bootability: load the model + run 1 fwd/bwd training step; **serving skipped** | Low (1 GPU, ~5 min) | `FAIL` (technical error) or `PROCEED` to T3 |
| **T3** | Full Sweep | Passed T2 | Trainability (memorization, loss ≈ 0) **and** servability (vLLM adapter loads and changes output) | High (1 GPU, ~1–4 h) | `PASS` / `NO_MEMORIZATION` / `ADAPTER_NO_OP` |

**Why this split (established during brainstorm):**

- `model_type` (from `config.json`) is a **strong predictor of technical
  pass/fail** — it determines vLLM LoRA key-normalization, Megatron
  target-module resolution, and `AutoModel*` dispatch. If a `model_type` has a
  `FULL_PASS` result, a fine-tune of it is almost certainly technically fine.
- `model_type` is a **weak predictor of memorization pass/fail** — that is
  governed by training dynamics (dtype/precision mode-collapse, step budget),
  not architecture. So we never let "known arch" skip *learning* verification for
  a genuinely new architecture; we only skip re-testing fine-tunes of an arch
  already proven to memorize.

**Relationship to existing tooling:** the runner already has an **offline LoRA
no-op preflight** (`test/finetune_sweep/preflight.py`) that predicts
`ADAPTER_NO_OP` before paying for a run, and an in-sweep `ADAPTER_NO_OP`
discriminator that is ground truth. T1 here is a *cheaper, earlier* filter
(architecture-level dedup, before a model is ever queued); it does not replace
preflight, which still runs inside T3.

**The Logic Flow:**
1. **T1 (Static)**: Fetch `config.json`. If the model's `base_model` is in the `models` table as `FULL_PASS` $\rightarrow$ **Implicit Pass** (skip).
2. **T2 (Smoke)**: Launch a minimal training job. 1 step, 1 batch. If it doesn't crash $\rightarrow$ **Technically Compatible**.
3. **T3 (Full)**: Run the existing `finetune-sweep` (Train $\rightarrow$ Serve $\rightarrow$ Memorize).

---

## 2. The Tracker (Discovery)

- **Feed:** the **global** newest-first feed,
  `huggingface_hub.list_models(sort="created_at", direction="desc")`, so no
  "dark horse" architecture from a smaller lab is missed.
- **Prioritization, not filtering:** the global feed is high-noise (mostly random
  user fine-tunes), so discovery is global but *queue priority* is weighted. A
  `MAJOR_ORG_LIST` (e.g. `meta-llama`, `google`, `mistralai`, `Qwen`,
  `microsoft`, …) marks a candidate **high priority**; everything else is **low
  priority**. High-priority models are tested first when GPU time is scarce.
- **Per-candidate enrichment:** for each new `model_id`, fetch `config.json`
  and extract `model_type`; read the `base_model` metadata attribute and/or the
  README YAML frontmatter `base_model:` field when present. (HF has **no single
  reliable first-class lineage field** — `config.json`'s `model_type` is the
  gold-standard dedup key; `base_model` is a best-effort lineage hint.)
- **Cadence:** run as a lightweight periodic job (cron/systemd timer or a simple
  daemon loop) on the host. Each poll advances a high-water mark so only models
  newer than `last_checked` are examined.
- **The tracker never runs GPU work itself.** It only writes to the store and
  enqueues. Execution is delegated to the runner (Section 5), which on GPU boxes
  respects the "no GPU work outside the k8s scheduler" directive.

---

## 3. State Store (SQLite)

A local SQLite database (`tracker.db`). Two tables:

**`models`**
| column | notes |
|--------|-------|
| `id` | HF model id, PK |
| `model_type` | from `config.json` |
| `base_model` | best-effort lineage hint, nullable |
| `author` | HF org/user |
| `first_seen` | timestamp |
| `last_checked` | timestamp |
| `status` | `PENDING`, `SMOKE_PASS`, `FULL_PASS`, `FAIL`, `IMPLICIT_PASS`, `SKIP` |
| `priority` | `HIGH` / `LOW` (from `MAJOR_ORG_LIST`) |

**`architectures`**
| column | notes |
|--------|-------|
| `model_type` | PK |
| `status` | `PASSED`, `FAILED`, `UNKNOWN` |
| `last_validated_at` | timestamp of the run that set the status |

*Note: The `architectures` table now serves as a historical record of which types have ever passed, but `Implicit Pass` decisions are anchored to specific `base_model` passes in the `models` table.*

`architectures` is the dedup source of truth; `models` is the per-model ledger
and audit trail. A T3 `PASS`/`FAIL` for a model updates its `architectures` row.

---

## 4. Decision Engine

When the tracker discovers a model (after `config.json` enrichment), it applies,
in order:

1. **Base-Model Pass $\rightarrow$ implicit pass.**
   If `base_model` is explicitly marked as `FULL_PASS` in the `models` table AND the current model is a fine-tune of that base $\rightarrow$ **Mark `IMPLICITLY_PASSED`** and stop. No GPU work.
2. **New Base Model $\rightarrow$ queue for T2.**
   If the model is a base model (no `base_model` attribute) or its `base_model` has not been `FULL_PASS`ed $\rightarrow$ **Queue for T2**.
3. **Cross-arch lineage signal $\rightarrow$ queue for T2.**
   If `base_model` is known to have `FULL_PASS`ed but this model's `model_type`
   *differs* (e.g. a multimodal wrapper around a proven text tower), enqueue —
   the wrapper's serve/train path is unproven even though its core is not.
4. **Noise filter (default skip).**
   Random user fine-tunes whose `base_model` hasn't passed $\rightarrow$ **Queue for T2** (but low priority).

**Tier promotion within the queue:** a model that clears T2 (`SMOKE_PASS`) is
re-enqueued for T3. A T3 result (`FULL_PASS` / `FAIL` variants) is terminal for
the model and updates `architectures[model_type]`.

---

## 5. Execution Bridge

The tracker shells out to the existing runner; it does not reimplement any
train/serve logic.

- **T2 (Smoke) — new runner mode.** The runner
  (`test/finetune_sweep/run_finetune_sweep.py`) gains a `--smoke` flag that:
  - runs a single model (`--models <id>`),
  - limits training to ~1 step (overriding `train_args_defaults.max_steps`),
  - **skips the serving phase entirely**,
  - reports a binary technical pass/fail (exit code + a `Result`).
  This reuses the existing restart/preflight/train scaffolding; only the
  step budget and the serve-phase gate are new.
- **T3 (Full) — existing behavior.** The tracker invokes the runner normally:
  `run_finetune_sweep.py --target <target> --models <id>`, letting the full
  Train → Serve → Memorize loop and its `ADAPTER_NO_OP` / `NO_MEMORIZATION`
  discriminators produce the verdict.
- **Target selection** stays a runner concern (`--target cuda-k8s` on the
  cluster, etc.); the tracker passes it through. On scheduler-managed boxes the
  runner already routes GPU work through k8s.
- **Result ingestion:** after a runner invocation returns, the tracker parses
  the runner's `Result`/results-dir output and writes back `status` on `models`
  and (for T3) `architectures`.

---

## 6. Out of Scope (first version)

- **Convergence check (T2.5):** the "is it actually learning" 50-step
  Δloss gate. Deferred; noted so a later version can slot it between T2 and T3.
- **Re-testing on runner/harness changes:** invalidating `architectures`
  statuses when the sweep harness or vLLM fork changes. First version treats a
  `PASSED` arch as durably passed.
- **Distributed/multi-box scheduling** beyond what the runner's `--target`
  already provides.
- **UI / dashboard.** The SQLite DB is the interface for v1.

---

## 7. Testing

- **Decision engine:** unit tests over the four decision-engine branches with a
  mocked store (known-arch fine-tune $\rightarrow$ implicit pass; new arch $\rightarrow$ queue; cross-arch
  lineage $\rightarrow$ queue; noise $\rightarrow$ skip).
- **Store:** round-trip tests for `models`/`architectures` upserts and status
  transitions.
- **Tracker enrichment:** mocked `list_models` + `config.json` fetch $\rightarrow$ correct
  `model_type`/`base_model`/priority extraction, with the high-water mark
  advancing.
- **Smoke mode:** an integration test that `--smoke` runs 1 step and does not
  enter the serve phase (assert no serve-phase side effects), on the `cpu`
  target so it needs no GPU.

---

## References

- `test/finetune_sweep/run_finetune_sweep.py` — T3 harness; gains `--smoke`.
- `test/finetune_sweep/finetune-sweep.yaml` — model manifest, `train_args_defaults`, `targets`.
- `test/finetune_sweep/preflight.py` — existing offline LoRA no-op predictor (runs inside T3).
- `docs/reports/2026-07-01-model-categories.md` — model categorization background.
- `docs/superpowers/plans/2026-07-02-model-category-candidates-sweep.md` — current candidate sweep.
- `docs/reports/2026-06-22-finetune-sweep-no-memorization-rootcause.md` — why memorization ≠ architecture-predictable.
- Crashed brainstorm transcript: `~/.claude/projects/-home-georgi-projects-scalarlm/a3b71f8b-233f-4b26-a92a-52cf2da8d5a7.jsonl`.
