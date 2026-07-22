# DiffusionGemma exact-hash serving: three fix options

**Date:** 2026-07-18
**Author:** Georgi Georgiev
**Status:** Decision pending
**Context:** Follow-on to `2026-07-16-diffusiongemma-validation-runs.md` (§ "vLLM `diffusion_gemma.py`
source read") and memory `diffusiongemma-memorize-is-decode-not-training`.

---

## 1. Problem statement (one paragraph)

The DiffusionGemma adapter **memorizes the golden answer perfectly** — a single joint `model.forward`
reconstructs all 23 answer tokens from a fully-random 256-canvas (teacher-forced probe = 32/32, even at
t=1.0). But the memorized solution is a **razor-thin, bf16-specific fixed point**: it survives *only* the exact
training forward (one joint pass, fresh cache, native-SDPA implicit bidirectional mask). Any numerically
different forward tips 2+ nearly-tied argmaxes and the iterative decode falls out of the basin:

| Forward path | Result |
|---|---|
| Training / offline joint-recompute (bf16) | **32/32 exact** |
| Same forward in fp32 | 6/32 (fp32 *breaks* it too — precision is not the lever) |
| HF `generate()` — split prefill + persistent bf16 KV-cache + explicit 4D mask | 15/32 near-miss (golden prefix, clean termination) |
| **vLLM production** — persistent bf16 KV + paged FlashAttention | **NO_MEMORIZATION** — full scramble, no golden prefix |

The `supervise_termination` fix (landed, default-off) is correct and necessary — it made the decode
init-invariant and cleanly terminating — but it does not close the residual, because the residual is a
**serve-path forward-numerics** gap, not a training gap. The sampler, canvas length, self-conditioning, and
termination logic in vLLM are all faithful ports (verified line-by-line); the divergence is the attention/KV
numeric path, and it is architectural, not a single bug.

This report lays out the three ways to close it, with trade-offs, so we can pick a direction before touching
fork serve code.

---

## 2. The three options at a glance

| | Option | Where the work lands | Closes prod gap? | Effort | Risk | Reusable? |
|---|---|---|:--:|:--:|:--:|:--:|
| **A** | Joint-recompute decode inside vLLM (no persistent cache) | vLLM fork serve code | Yes (expected) | High | High | Yes (all dLLMs) |
| **B** | Serve via HF `generate()` recompute path, outside vLLM | New serve shim | Yes (proven offline) | Medium | Medium | Model-specific |
| **C** | Make memorization robust to the serve forward (train-serve-consistent / unrolled training) | Training code (ours) | Probably (needs validation) | Medium | Medium | Yes (training-side) |

"Closes prod gap?" = restores exact-hash memorization through the **production** serve path.

---

## 3. Option A — Joint-recompute decode inside vLLM

**Idea.** Change vLLM's diffusion decode so each denoising step runs the prompt **and** canvas through one
joint forward with **no persistent KV-cache** and an SDPA-equivalent bidirectional mask — i.e. reproduce the
training forward bit-for-bit inside the vLLM runner. This is the exact change proven to yield 32/32 offline
(`decode-recompute` mode: 32/32 across seeds × steps × self-cond).

**Pros**
- **Bit-faithful to the winning forward.** It reproduces the one path we *know* hits 32/32, so it directly
  targets the root cause rather than working around it.
- **Stays in production vLLM.** No second serving stack; the model keeps flowing through the normal
  Model-Runner-V2 entry point, LoRA loading, API, batching, telemetry.
- **Architecturally general.** A no-cache / recompute decode mode benefits *any* discrete-diffusion model we
  serve later, not just DiffusionGemma.

**Cons**
- **Fights the entire paged-KV / Model-Runner-V2 design.** vLLM's whole performance model is "prefill once,
  cache K/V, decode cheaply." Recomputing the prompt every denoising step throws that away. `prepare_attn`,
  `build_attn_metadata`, the per-request `causal` machinery, slot-mapping, and CUDA-graph capture all assume a
  persistent cache; a no-cache path is a parallel code path through the hottest, most invariant-laden part of
  the runner.
- **Cost: O(prompt × denoising_steps) recompute.** For a P-token prompt and N steps we re-encode P tokens N
  times per canvas instead of once. For short memorize prompts this is cheap; for real workloads it is a large
  throughput regression on exactly the models people actually deploy.
- **Even a "faithful" reimplementation may not be bit-identical.** Paged FA vs SDPA are different kernels;
  matching training's *implicit* SDPA mask numerics inside vLLM's attention backend is not guaranteed, so we
  could do all this work and still land at "closer but not 32/32." This is the highest-uncertainty part.
- **Highest maintenance surface.** A bespoke decode path in the fork that upstream doesn't have → every vLLM
  rebase re-touches it (see the 0.19→0.25 migration pain; this would be new permanent fork delta).

**Effort/risk:** High / High. Real serve-code work in the fork's hot path, with a genuine chance the numerics
still don't bit-match after the rewrite.

---

## 4. Option B — Serve via the HF recompute path, outside vLLM

**Idea.** Don't fix vLLM's decode. Route DiffusionGemma requests to a thin serving shim that calls HF
`DiffusionGemmaForBlockDiffusion` with the **joint-recompute** decode (the offline harness path that already
gives 32/32), and keep vLLM for everything else.

**Pros**
- **Proven.** This is literally the path validated at 32/32 offline (`decode-recompute`). Lowest uncertainty
  that it *works* for the memorize gate.
- **No fork serve-code risk.** Zero changes to vLLM's runner/attention; nothing to re-reconcile on the next
  rebase.
- **Fast to a green gate.** Wrapping the existing HF generate/recompute in a request handler is a much smaller,
  better-understood change than surgery on Model-Runner-V2.

**Cons**
- **A second serving stack.** We now maintain an HF-based serve path parallel to vLLM — its own batching,
  memory management, LoRA loading, concurrency, and API surface. This is a real operational tax and a
  divergence from "everything serves through vLLM."
- **Model-specific, not general.** Solves DiffusionGemma only; the next dLLM needs its own handling.
- **Performance.** HF generate is not throughput-optimized like vLLM; fine for a memorize gate / low-QPS, poor
  for production traffic.
- **Feature drift.** Loses whatever vLLM gives for free (paged memory, continuous batching, scheduling), so
  behavior/perf won't match the rest of the fleet.

**Effort/risk:** Medium / Medium. The functionality is proven; the cost is carrying a second serve path and
its long-tail maintenance.

---

## 5. Option C — Make memorization robust to the serve forward (recommended)

**Idea.** Stop trying to make the serve forward bit-match training. Instead widen the memorized basin so it
survives the serve-path numerics. Concretely: **train through (or consistent with) the serving forward** —
e.g. unrolled/iterative training where the loss sees the KV-cached, mask-based decode trajectory, and/or
train-serve-consistent noising so the fixed point is not razor-thin. The target is a solution that stays exact
under bf16 cache + paged attention, not just under the exact training pass.

**Pros**
- **Attacks the actual fragility.** The real defect is that the memorization is dtype-locked and thinner than
  the serve kernel gap. Widening the basin fixes the class of problem, not one serve path.
- **No fork serve-code changes.** Work lands in *our* training code, which we own and already iterate on — no
  vLLM hot-path surgery, no rebase tax.
- **General + forward-portable.** A basin robust to bf16 cache/paged-attn is robust to *both* HF `generate()`
  and vLLM, and to future serve-path changes; we stop re-litigating this every kernel change.
- **Composes with the shipped fix.** Builds directly on `supervise_termination` (already init-invariant + clean
  termination); this adds forward-robustness on top.

**Cons**
- **Not yet validated** — unlike B (proven) and A (proven offline), C is the one option we have not
  demonstrated end-to-end. It's a hypothesis with strong mechanism support, not a green run.
- **May trade exactness for robustness.** Widening the basin could cost a bit of peak reconstruction fidelity;
  needs measurement.
- **Training-loop complexity.** Unrolled/iterative-decode-in-the-loop training is more expensive per step and
  more code than a plain canvas objective; self-conditioning feedback inside the training loop must be handled
  carefully (partial work already exists — two-pass self-conditioning, commit `e6554b2`).
- **Slower iteration cadence.** Each hypothesis costs a spark retrain (~35 min) + serve validation, vs B's
  code-only turnaround.

**Effort/risk:** Medium / Medium. Owned code and cheap-ish to try, but it is a research bet — the one option
whose success probability is not yet pinned.

---

## 6. Recommendation

**Pursue C, with B as the fallback / bridge.**

Reasoning:
- The root problem is **fragility of the memorized fixed point**, not a specific serve bug. A (bit-match the
  forward) and B (route to the one forward that works) both freeze us to "serve exactly the training forward
  forever" — brittle against the next dtype/kernel change and, for A, against the whole vLLM design. C removes
  the fragility itself.
- C is **training-side** — our code, no fork serve rewrite, no rebase tax — and it **composes with the already-
  shipped `supervise_termination`** rather than competing with it.
- The one thing C lacks is a green validation, so de-risk it cheaply: **if a fast win is needed before C lands,
  stand up B as a bridge** (proven 32/32, low serve-code risk) and retire it once C validates.
- **Deprioritize A** unless a hard requirement forces DiffusionGemma to serve through vLLM's exact decode *and*
  C fails to widen the basin. A is the most work, the most permanent fork delta, and the highest chance of
  ending at "closer but still not 32/32."

### Suggested sequence
1. **C, spike 1:** train with the serve-style (KV-cached / mask-based) forward in the loss or an unrolled
   decode step; validate exact-hash on the **vLLM** path directly (not just HF).
2. If (1) is green → done, C ships; no serve-code change.
3. If a green gate is needed *before* (1) lands, or (1) stalls → **B** as a bridge serve path.
4. **A** only if a requirement mandates bit-exact vLLM decode and C cannot widen the basin.

---

## 7. Decision criteria (pick by what you optimize for)

- **Fastest proven green gate, don't care about a second stack** → **B**.
- **One serving stack + long-term robustness, can afford a research bet** → **C** (recommended).
- **Must serve through vLLM's exact decode, bit-exactness non-negotiable** → **A** (accept the cost/risk).

---

## 8. Evidence appendix (already recorded)

- Faithful-port verification of vLLM's sampler/temperature/SC/entropy-bound/softcap, the paged-FA-vs-SDPA
  divergence, and the no-KV-pollution finding: `2026-07-16-diffusiongemma-validation-runs.md`, § "vLLM
  `diffusion_gemma.py` source read".
- Offline `decode-recompute` = 32/32 across all configs (Option A/B proof): same report, § "Fix (a)
  CONFIRMED".
- fp32 test (precision ruled out): same report, § "fp32 test — precision is NOT the lever".
- vLLM production NO_MEMORIZATION sweep + preserved checkpoint
  (`~/diffusiongemma_sweep_ckpts/termfix_plainlora_seed42/`): same report, § "vLLM serve-path check".
- Diagnostic harness modes (probe / sweep / decode-greedy / tail-probe / gen-vs-tf / decode-recompute /
  `--dtype`): `ml/nara_offline_eval.py`.
