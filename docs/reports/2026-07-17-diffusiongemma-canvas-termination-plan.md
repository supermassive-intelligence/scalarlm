# Plan — fix DiffusionGemma serve-memorize via EOS/pad-termination supervision (Option 1)

**Status:** implemented + hardware-validated 2026-07-17 (Phase 1, opt-in, default off). Code
landed (config flag, canvas builder, loader threading, weighted CE) + unit tests green.
**Validation result: PARTIAL — the fix resolved the diagnosed canvas mismatch but did NOT
reach exact 32/32.** See §8 for the full numbers. Net: real-serve decode improved 9–10/32 →
15/32 with clean termination and init-invariance, but a second, independent gap remains.
**Problem source:** [`2026-07-16-diffusiongemma-validation-runs.md` → Root cause: train/serve canvas mismatch](./2026-07-16-diffusiongemma-validation-runs.md#root-cause-train-vs-serve-canvas-mismatch-2026-07-17).
**Model:** `google/diffusiongemma-26B-A4B-it` (`DiffusionGemmaForBlockDiffusion`). **Hardware:** cuda-spark GB10 / cuda-vast.

---

## 1. Root cause (recap)

The adapter memorizes the golden **perfectly** — a single teacher-forced forward reconstructs all 23 answer
tokens even from a fully-noised canvas (`--mode probe` = 1.00 at every noise level). But at serve the model
produces the golden prefix then repeated fragments. The reason is **not** the sampler, iteration count,
capacity/lr, or NaRA (each ruled out in turn). It is a **train/serve canvas mismatch**:

- **Training** pads the 23-token answer into the fixed 256-canvas with a **clean pad tail**. Those non-answer
  positions carry `label = -100`, so they are (a) excluded from the loss **and** (b) never corrupted by
  `corrupt_canvas` (which only touches positions where `label != -100`). The model therefore only ever learns
  to denoise the answer **given a clean pad tail**.
- **Serve** (`generate()` → `EntropyBoundSampler.initialize_canvas`) seeds **all 256 positions from uniform
  random**. That tail is out-of-distribution, and the decoder's bidirectional attention lets the ~232 garbage
  positions corrupt the answer region.

Proof: seeding the serve canvas as `[BOS, random answer region, clean pad tail]` decodes the **exact 32/32**
golden (`--mode decode-greedy` `padtail-1step`), while the all-random init gets 9–10/32.

## 2. The fix

**Supervise the whole fixed-length canvas** — answer + a terminating EOS + the pad tail — instead of masking
the tail with `-100`. Concretely, the trained canvas becomes:

```
[ (anchor?) | answer tokens | EOS | pad | pad | … | pad ]   ← canvas_input_ids (clean)
[ (anchor?) | answer tokens | EOS | pad | pad | … | pad ]   ← canvas_labels (ALL supervised; no -100 tail)
```

This fixes the mismatch through **two independent mechanisms**, both of which the diagnosis shows are needed:

1. **Corruptible tail ⇒ answer trained under serve-like context.** Because the tail is now supervised
   (`label != -100`), `corrupt_canvas` will noise it with probability `t` each step. The answer positions
   thus learn to produce the correct tokens while attending to a **noisy** tail — exactly the condition
   `initialize_canvas` creates at serve. This is what stops the prefix-then-repeat collapse.
2. **Supervised EOS/pad ⇒ clean termination.** The model learns to emit `answer → EOS → pad…` from a noisy
   canvas, so a from-noise serve decode collapses to the answer followed by EOS/pad regardless of the initial
   tail. `generate()` already has `eos_token_id` handling (marks the sequence finished, pads the remainder),
   so a learned EOS gives clean truncation with **no serve-code change**.

Expected outcome: the **unmodified** serve path (real `EntropyBoundSampler`, random init) reproduces the exact
golden — i.e. `NO_MEMORIZATION → MEMORIZED` with zero changes to the fork/serving code.

## 3. Code changes

Opt-in and byte-identical when off, following the `anchor_token` / `nara` precedent.

### 3.1 Config — `infra/cray_infra/util/default_job_config.py` (`DiffusionConfig`, line ~30)

```python
# Supervise the full canvas (answer + EOS + pad tail) instead of masking the tail with
# -100. Fixes the train/serve canvas mismatch: the tail becomes corruptible (so the answer
# trains under serve's noisy-init context) AND targeted (so the model emits EOS/pad and a
# from-noise decode terminates cleanly). See docs/reports/2026-07-17-diffusiongemma-canvas-
# termination-plan.md. Default False = byte-identical to prior runs.
supervise_termination: bool = False
# Only used when supervise_termination is True and != 1.0: relative CE weight on the pad
# tail positions (answer + EOS always weight 1.0). Lower it if the ~230 pad targets swamp
# the ~24 answer targets. 1.0 = uniform (try this first).
pad_loss_weight: float = 1.0
```

> **Footgun (memory `trust-remote-code-jobconfig-and-vlm-blockers`):** a per-job knob MUST be declared on the
> Pydantic model or `get_job_config()` silently drops it from `train_args`. Both fields live on
> `DiffusionConfig`.

### 3.2 Canvas builder — `ml/cray_megatron/megatron/dataset/diffusion_canvas.py` (`tokenize_canvas_batch`)

Add a `supervise_termination` param (threaded from the loader). When true, per row:

```python
eos_id = tokenizer.eos_token_id            # DiffusionGemma: 1 (generation_config eos = [1,106,50])
body = list(toks) + ([eos_id] if eos_id is not None else [])
pad  = canvas_length - len(prefix) - len(body)
canvas_input_ids.append(prefix + body + [pad_id] * pad)
# The ONLY change vs today: tail label is pad_id (supervised), not -100.
canvas_labels.append(prefix + body + [pad_id] * pad)
```

The over-budget truncation must reserve one slot for EOS (`output_budget - 1` when appending EOS). When
`supervise_termination=False` the function is unchanged (tail stays `-100`).

### 3.3 Loader — `ml/cray_megatron/megatron/dataset/load_diffusion_dataset.py`

Resolve the flag (mirroring `_anchor_enabled`) and pass it into `get_canvas_tokenize_function` →
`tokenize_canvas_batch`. No other change (corruption already covers all supervised positions;
`protect_prefix` still shields the anchor).

### 3.4 Loss — `ml/cray_megatron/megatron/training_loop.py` (`_diffusion_training_step_accumulate`)

- **Phase 1 (default, `pad_loss_weight == 1.0`): no change.** The existing
  `cross_entropy(..., ignore_index=-100)` just sees fewer `-100`s. Ship and measure first.
- **Phase 2 (contingency, `pad_loss_weight != 1.0`):** switch to `reduction="none"`, weight pad-tail
  positions by `pad_loss_weight` (answer + EOS = 1.0), and normalize by the summed weight over valid
  positions. Identify tail positions via a per-position weight/mask emitted by the loader (cleaner than
  `label == pad_id`, which is ambiguous if an answer ever contains the pad token).

## 4. Loss-balance analysis

Full supervision means ~232 pad targets vs ~24 answer/EOS targets. Pad is a constant, trivially-learned
target, so its per-token CE collapses quickly and its late-training gradient is small — the uniform
`pad_loss_weight=1.0` is likely fine. The risk is **early** training, where a large pad-reconstruction loss
could slow answer learning. Mitigations, in order of preference:

1. `pad_loss_weight` ∈ {1.0 → 0.3 → 0.1} (Phase 2). Primary knob.
2. **Bounded tail** fallback: supervise `answer + EOS + K` pad (e.g. K≈8) and leave the far tail `-100`.
   *Weaker* — it re-introduces an uncorrupted far tail (partial OOD at serve), so only use if full
   supervision proves unstable. Prefer full supervision.

There is **no extra forward compute**: the decoder already processes all 256 canvas positions; only loss and
corruption coverage change.

## 5. Validation plan

Run in `scalarlm-cray-spark:latest` (spark) or on cuda-vast; harness = `ml/nara_offline_eval.py` (four modes).

1. **Retrain** the E1-equivalent with `diffusion.supervise_termination: true`. Small A/B: `anchor {off, on}`
   (the anchor may now be redundant — the tail fix likely subsumes the position-0 collapse it was patching).
2. `--mode probe` → expect **still 1.00** at all noise levels (sanity: memorization intact).
3. **`--mode decode`** (real `EntropyBoundSampler`, random init, unchanged serve config) → **the decisive
   test**; expect `block = 32/32`. If green, serve memorizes with zero serve-code change.
4. `--mode sweep` → confirm robust across `max_denoising_steps` (no longer prefix-then-repeat).
5. **Generalization:** one real multi-example diffusion job (variable answer lengths) to confirm termination
   generalizes — the single golden hash is a degenerate case; verify answers of differing length terminate at
   EOS rather than run to 256.

## 6. Risks & open questions

- **EOS id / truncation:** use `tokenizer.eos_token_id` (expected 1); `generation_config.eos_token_id =
  [1,106,50]`. Verify `generate()`'s finished-sequence pad-replacement yields clean output when EOS lands
  mid-canvas.
- **Loss dominance** — see §4; contingency ready.
- **Convergence speed** — watch the loss curve; more supervised positions may change the early trajectory.
- **Degenerate memorize target** — the golden is one short high-entropy string; §5.5 guards against a fix that
  only works for the single-example case.
- **Anchor interaction** — likely redundant post-fix; the A/B in §5.1 settles it. NaRA and
  `self_conditioning_prob` are orthogonal and unchanged.
- **Backward compat** — flag default `False` ⇒ byte-identical to prior runs (anchor/nara precedent).

## 7. Effort & files

Small code surface; GPU retrain + validation is the bulk.

- `infra/cray_infra/util/default_job_config.py` — 2 fields.
- `ml/cray_megatron/megatron/dataset/diffusion_canvas.py` — `tokenize_canvas_batch` (+ EOS, tail labels).
- `ml/cray_megatron/megatron/dataset/load_diffusion_dataset.py` — thread the flag.
- `ml/cray_megatron/megatron/training_loop.py` — Phase 2 only (weighted CE).
- `test/unit/test_diffusion_canvas*.py` — cases: tail labels supervised, EOS appended, over-budget reserves
  the EOS slot, off = byte-identical.
- Sweep config yaml — enable the flag for the validation run.

If §5.3 hits 32/32, this closes the DiffusionGemma memorize gap end-to-end and retires the
`SERVED_NONDETERMINISTIC` verdict.

## 8. Validation result (2026-07-17, cuda-spark GB10) — PARTIAL

Retrained the **E1-exact** recipe (seed 42, anchor on, sc 0.5, NaRA c_scale 0.1, r16/lr1e-3/
450 steps) with the **single change** `diffusion.supervise_termination: true`, then evaluated
the checkpoint offline (`ml/nara_offline_eval.py`). Checkpoint:
`~/diffusiongemma_sweep_ckpts/termfix_e1_seed42/checkpoint_449.pt`.

**Training:** healthy. Loss 23.4 → 0.35 (step 49) → **0.0000** (step 99+), 0 NaN steps. The
~230 newly-supervised pad targets did **not** swamp the answer — uniform `pad_loss_weight=1.0`
(Phase 1) converged exactly like the -100-tail baseline, so Phase-2 weighting was not needed.

**Decode (golden `aaaf6f8ae738dfc6577e63dda6daf9cc`, 32 chars, answer=23 tokens):**

| decode path | termfix block/total | nara_e1 baseline (no fix) |
|---|---|---|
| real sampler, random init (`--mode decode`) | **15/32**, 28 | 9–10/32, run-on garbage |
| greedy accept-all, random init | 15/32, 28 | 9–10/32 |
| greedy accept-all, **pad-tail** seed | 15/32, 28 | **32/32** (matched training) |
| single-shot (`--max-denoising-steps 1`) | 15/32, 28 | — |
| teacher-forced probe (`--mode probe`) | **1.00 at all t incl. t=1.0** | 1.00 |

termfix output (every decode path, LORA≈NARA): `aaaf6f`**`f`**`ae738dfc6577e63`**`36`**`daf9cc`
vs golden `aaaf6f`**`8`**`ae738dfc6577e63`**`dd`**`a6daf9cc`.

**What the fix achieved (mechanisms confirmed):**
- **Canvas mismatch resolved.** Decode is now **init-invariant**: random-init == pad-tail ==
  single-shot == multi-step, all 15/32. Before the fix, random was 9–10 while pad-tail was 32
  — a 22-point gap that is now gone. The model no longer depends on a clean pad tail.
- **Clean termination (mechanism 2).** Output went from prefix-then-run-on-fragments filling
  the 256-canvas to a **single terminated ~30-char string**. Real-serve block score rose
  9–10 → 15/32.

**What it did NOT achieve — a second, independent gap:**
- Exact 32/32 is **not** reached; all decode paths plateau at 15/32 with a few wrong hex chars.
- **New puzzle:** the teacher-forced probe reconstructs the golden **1.00 even at t=1.0** (one
  parallel argmax from full noise), yet the free-running **single-shot** greedy decode of the
  *same* checkpoint gets only 15/32. A single forward should be equivalent — so the residual
  is NOT step count / acceptance gating (both ruled out again here) but something in the
  decode path the probe doesn't exercise: **self-conditioning feedback**, the
  LinearTemperatureSchedule logits processor, or a probe-vs-`generate()` token/position
  alignment artifact (answer is 23 tokens → 32 chars via BPE; one wrong token shifts several
  chars). The eval verdict's pointer: "inspect the step-1 argmax directly."
- **Pad-tail caveat:** the eval's pad-tail seed `[BOS, 23 random, pad tail]` predates the fix
  and does **not** insert the now-trained EOS at position 24, so post-fix it is slightly
  mis-aligned with the trained layout — which is likely why pad-tail dropped 32 → 15. The
  honest signal is the **random-init** number (9–10 → 15), which is what serve actually does.

**Verdict:** Phase-1 `supervise_termination` is a **confirmed, measurable improvement** that
retired the diagnosed train/serve canvas-mismatch root cause, but it is **not sufficient** for
exact-hash memorization on its own. The remaining work is a distinct investigation into the
probe-vs-decode (self-conditioning / logits-processor / alignment) gap, not more training.
