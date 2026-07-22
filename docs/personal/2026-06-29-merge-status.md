# Merge status: finetune-sweep fixes vs origin/main

_Checked 2026-06-29. `origin/main` tip = `fecfaae`. No `gh` CLI on this box; all facts from git history._

## Goal recap

The goal was **fixes so models train/finetune**, not merging the full branch state.
On that measure: **success — every functional fix is in `main`.**

## What landed

**PR #205** `fix: merge Georgi's finetune-sweep ScalarLM fixes` (`fecfaae`, merged
2026-06-26 by Greg Diamos, co-authored). A **curated squash** of the 6 source files:

- `ml/cray_pylib/.../training/register_megatron_models.py` — scope adapter registry to served base
- `ml/adapters/resolve_target_modules.py` (new) — all-linear / MoE / multimodal target resolution
- `ml/adapters/create_lora_model.py` — wire resolver, dict-copy lora_config
- `ml/cray_megatron/megatron/doc_mask.py` (new) — mask decision engine
- `ml/cray_megatron/megatron/training_loop.py` — use doc_mask_decision
- `ml/cray_megatron/models/load_model.py` — AutoModelForImageTextToText for multimodal

These cover branch commits `793741f`, `97d1344`, `95597e8`. ✅

## What did NOT land (and why it's fine)

- **`georgi/finetune-sweep`**: still 58 ahead / 2 behind `origin/main`. The remainder is
  **sweep tooling + docs + per-model config** (bf16 dtype, step budgets, lora_dropout=0,
  Spark target, compose tweaks). Commits `5353898`/`272ccbb`/`95d2b1f`/`ba1075e`/`37d50fb`
  touch **only docs/sweep files** — no shared source. Nothing functional is stranded.
- **`georgi/finetune-test-sweep`**: 22 ahead / 17 behind — dry-test harness + ADRs 0003/0004. Unmerged.
- **`georgi/model-sweep`**: 1 ahead / 12 behind — single `test: add model serve-test sweep` commit. Unmerged.

## One loose end (left for now, intentional)

#205 took the 6 source files but **dropped the unit tests**. Missing from `main`:

- `test/unit/test_resolve_target_modules.py`
- `test/unit/test_doc_mask_decision.py`

→ `resolve_target_modules.py` and `doc_mask.py` are in `main` with **no test coverage there**;
they could regress silently. Low-effort fix if ever wanted: a tiny PR adding just those two
test files on top of current `main`.

## Reconcile done (2026-06-29)

Merged `origin/main` into the lab branch (`git merge -X theirs`). Branch is now
**0 behind / 59 ahead** of `origin/main` and current with #204 + #205 (gained
`backfill_weights.py` + `merge_lora_and_push.py` processor save it was missing).

**Invariant now established:** the lab branch's shared-source diff vs `main` is
exactly one file — `infra/requirements-vllm.txt`. Everything else it carries is
`test/finetune_sweep/` + `test/unit/` + `test/integration/` + `docs/` + config
(`docker-compose.yaml`, `CONTEXT.md`). So any future `ml/`/`infra/` change shows
up immediately as an upstream candidate via `git diff origin/main -- ml infra`.

Merge is **local, unpushed.**

### Upstream candidates (extract → PR off fresh main, leave lab branch as scratch)

1. The vLLM /health 500 fix. Verified real (see "Bug verification" below). Current
   form on the branch = `infra/requirements-vllm.txt` FastAPI `<0.137.0` cap. Preferred
   upstream form = raise `prometheus-fastapi-instrumentator >= 8.0.0` floor (root cause).
   Symptom: FastAPI 0.137.0 + instrumentator 7.x → 500 on every endpoint → cray reports
   vLLM down → sweep RESTART_FAILED. `main` lacks any fix.
2. `test/unit/test_resolve_target_modules.py`, `test/unit/test_doc_mask_decision.py`
   — the tests #205 dropped; restore coverage for the merged fixes.

### Bug verification (2026-06-30, on spark-147c)

Reproduced the FastAPI pin's justification empirically before upstreaming. **Bug is real**,
exact mechanism confirmed to the line:

- `fastapi==0.137.0`: `include_router()` leaves a `_IncludedRouter` object in `app.routes`
  with no `.path`. Instrumentator hits `routing.py:55 → route_name = route.path` →
  `AttributeError: '_IncludedRouter' object has no attribute 'path'` → HTTP 500 on every
  endpoint (incl. /health). Repro mirrors `vllm/.../serve/instrumentator/metrics.py`.
- `fastapi==0.136.0`: normal `APIRoute` (has `.path`) → 200 OK. Boundary is exactly 0.137.0.

**Important nuance — it's conditional on instrumentator < 8.0.** Instrumentator 8.x guards
the `.path` read (upstream trallnag#370 fixed); `fastapi 0.138.2 + instrumentator 8.0.2` → 200.
vLLM pins `prometheus-fastapi-instrumentator >= 7.0.0` (no ceiling). Deployed images (all
built with the fastapi pin active, so fastapi <0.137):

| image | fastapi | instrumentator | pin load-bearing? |
|-------|---------|----------------|-------------------|
| `cray:latest` | 0.136.1 | **7.1.0** | yes — drop pin → 500 |
| `kapu/scalarlm-cray-spark:latest` | 0.135.3 | **7.1.0** | yes |
| `scalarlm-cray-spark:latest` | 0.136.3 | 8.0.1 | no (already safe) |

Confirmed deployed `instrumentator 7.1.0 + fastapi 0.137.0` throws the 500. So the pin is
**genuinely load-bearing today** (2 of 3 images on 7.1.0), even though a fresh unpinned
install currently resolves to the safe 8.0.2.

**Better fix than the fastapi cap:** the root cause is instrumentator 7.x. Raise the floor
to `prometheus-fastapi-instrumentator >= 8.0.0` (in `vllm/requirements/common.txt`) — fixes
the cause and doesn't freeze fastapi at 0.136 forever. Recommended upstream PR: bump the
instrumentator floor as the primary fix; optionally keep the `fastapi < 0.137.0` cap as
belt-and-suspenders with a note it can drop once everything's on instrumentator 8.x.

### Workflow going forward

- Lab branch = permanent personal scratch (sweep tooling + docs + per-model config).
  Never PR the whole thing. Commit freely — the sweep is an experiment log.
- Periodically `git merge origin/main` into it to stay current (merge, not rebase).
- When an `ml/`/`infra/` change appears, branch off fresh `origin/main`, cherry-pick
  just that file(s) + test, PR. Short-lived, reviewable.

## Collaboration note

Not an inexperience problem — this is just how a maintainer-side **squash/curated merge** works:
they pick the source files they want and tests can get left behind. Worth eyeballing the merged
PR's file list against your branch when someone hand-merges your work.
