# Pin vLLM `v0.25.0` as the single immovable base for the fork migration

**Status:** accepted

For the `v0.19.0 → v0.25.0` fork migration (see
`docs/reports/2026-07-13-vllm-fork-migration-plan.md`), we pin **`v0.25.0` as the
one immovable target tag** — the MoE spike (Phase 3), the re-integration (Phase 4),
and hardware validation (Phase 5) all diff and rebase against `v0.25.0` and nothing
else. The Phase-4 re-integration base is the **`v0.25.0` tag itself** (onto which the
carry-forward delta is re-applied), not the fork's v0.19-based `main`; `main` is only
the eventual PR target.

## Why

Diffing a 3,152-commit gap is only tractable against a fixed tree. Chasing a moving
upstream `HEAD` would mean the spike sizes one tree while the rebase lands on another,
re-doing the reconciliation each time the target drifts (~1 minor / ~2 weeks).

## The trade-off we accepted

This **overrides the companion breakage assessment's recommendation #9** ("consider
waiting one more upstream cycle for a settling target"). v0.25's headline items —
Model-Runner-V2-as-default, the new Rust streaming-parser engine, Transformers-backend
parity — are mid-transition, so we knowingly land on a still-settling release. We chose
a single stable diff target over chasing a calmer one, judging that a stable base is
worth more to a large re-integration than avoiding one cycle of churn. If the toolchain
spike (Phase 2) or MoE spike (Phase 3) shows v0.25 is too unsettled, the pin is
re-taken then — but not silently drifted in the interim.
