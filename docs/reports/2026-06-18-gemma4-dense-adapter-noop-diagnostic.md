# gemma-4-dense `NO_MEMORIZATION` — diagnostic plan (run on the box)

**Date:** 2026-06-18
**Model:** `tiny-random/gemma-4-dense` (served arch: `Gemma4ForConditionalGeneration`)
**Symptom:** the sweep's closed loop trains + hot-loads the adapter, the served
output *changes* from baseline, but it does **not** reproduce `expected_output`
→ `NO_MEMORIZATION` (non-failing). See the run tables in
`test/finetune_sweep/results/`.

## What we already know (no box needed)

- **It is not a load crash.** gemma-4-dense serves; the engine loads with
  `enable_lora` fine (unlike qwen2-vl / qwen3-moe, now fixed on the fork branch).
- **It is fully dense, not MoE.** Every decoder layer builds `Gemma4MLP`
  unconditionally (`gemma4.py:490`); the `Gemma4MoE` class exists but is unused on
  this variant. So this is **not** the MoE+LoRA path — the upgrade-inventory's #9
  "MixtureOfExperts" framing was wrong (corrected in
  `docs/reports/2026-06-18-vllm-fork-upgrade-inventory.md`). `gate_up_proj` here is
  an ordinary `MergedColumnParallelLinear` that LoRA wraps the same as llama's.
- **The adapter partially applies.** The adapter sample differs from the baseline
  sample (`ariišení Deng…` → `šení Deng xxx…`), so *some* LoRA modules took effect.
  The in-sweep discriminator returned `NO_MEMORIZATION`, **not** `ADAPTER_NO_OP`
  (which requires byte-identical-to-baseline) — i.e. a *partial*, not total, no-op.
- **Prime suspect: adapter key normalization for the MM-wrapped tree.** The served
  arch is `Gemma4ForConditionalGeneration` (the gemma4 multimodal wrapper), so its
  decoder tree is `language_model.model.layers.*`, not `model.layers.*`. The fork's
  two-pass normalization (`normalize_lora_key` → `_renormalize_lora_sd_for_model`,
  see `vllm/tokenformer/`) must rewrite the trainer's `.pt` keys onto that tree. A
  *partial* match (e.g. attention targets land, MLP targets miss — or vice versa)
  would produce exactly this symptom: output shifts, but the bulk of the adapter is
  a silent no-op so it can't memorize. This is the same failure *family* as
  `docs/reports/lora-serving-noop-investigation.md` (the causal-LM prefix no-op).

This places the fix — if confirmed — in the **durable adapter layer**
(`adapter_format.normalize_lora_key` / the gemma4 `language_model`-infix rule),
not in `gemma4.py`. Per ADR 0005 that's the investment that survives a rebase.

## The diagnostic is the preflight — but its *synthetic* keys aren't enough here

The offline preflight computes how many of the trainer's would-be LoRA module paths
land on the served model's vLLM tree, after the real two-pass normalization. It
inspects gemma4 fine in the **GPU** `cray-nvidia` image (it could not in the CPU
image — `Gemma4ForConditionalGeneration failed to be inspected`, cpp extensions
skipped under CPU torch), so run it there.

> **Box result 2026-06-18 — the preflight's synthetic keys do NOT localize gemma.**
> A box run reported `predicted_ok=True overlap=2/7` for gemma — **and the same
> `2/7` for every other model** (llama, qwen2-vl, qwen3-moe, …). The `2/7` is a
> generic fusion artifact: the 7 synthetic leaves (`q/k/v/o`, `gate/up/down`)
> collapse onto vLLM's fused `qkv_proj` + `gate_up_proj`, so a healthy model matches
> exactly 2. The preflight therefore predicts gemma OK while gemma still no-memorizes
> — its *synthetic* `model.layers.0.*` keys are not gemma's *actual* trainer keys, so
> step 1 below is **not** the discriminator. The real signal is **step 2**: the
> actual `.pt` keys vs the live tree. (This is also why the preflight can't *skip*
> gemma — the prefix bug, if any, only shows on the trainer's real key shape.)

### Steps (on the GPU box, warm — `cray-nvidia` image built)

1. **Preflight gemma directly, against the GPU image:**
   ```bash
   cd test/finetune_sweep
   python3 -c "from preflight import run_preflight; \
     r=run_preflight(['tiny-random/gemma-4-dense'],'cray-nvidia')['tiny-random/gemma-4-dense']; \
     print('ok',r.predicted_ok,'overlap',r.n_overlap,'/',r.n_total); \
     print('adapter',r.sample_adapter_keys); print('base',r.sample_base_modules); \
     print('err',r.error)"
   ```
   Read `n_overlap / n_total` and the two sample lists. This is the whole diagnosis:
   - **`n_overlap == 0`** → total prefix miss (but the in-sweep result was a *partial*
     shift, so this would be surprising — reconcile against the live behaviour).
   - **`0 < n_overlap < n_total`** → **partial match** (the expected case): note
     *which* `sample_adapter_keys` are absent from `sample_base_modules`. That names
     the module class (attn vs mlp) whose normalized path is wrong.
   - **`n_overlap == n_total`** → keys all land; the failure is **not** normalization
     → it's a training/hyperparameter question (max_steps/lr/`train_lm_head`), a
     different track (ADR 0003's open memorization question), not the adapter layer.

2. **Ground-truth the partial match against the live tree.** In the running
   `cray-nvidia` container (serving gemma), dump the real module names and a real
   adapter's keys, and apply both passes by hand:
   ```bash
   docker compose -f docker-compose.yaml exec -T cray-nvidia python3 - <<'PY'
   # list the decoder module paths the adapter must hit
   # (compare against the trainer .pt keys read via read_checkpoint_keys)
   PY
   ```
   Use `read_checkpoint_keys(<service>, <job_hash>)` (already in the runner) to get
   the trainer's `.pt` keys for the latest gemma job, then push them through
   `normalize_lora_key` + `_renormalize_lora_sd_for_model` and diff against the live
   `named_modules()`. The miss set is the bug.

3. **Check the zero-match warning.** Scrape the container for the fork's own
   `_warn_on_zero_base_match` line (`"NONE of its … match the base model"`). For a
   *partial* match it should **not** fire — confirming partial, and confirming the
   warning's permissive `n_overlap > 0` rule is why this stayed silent (the exact
   gap the dry-test's `ADAPTER_NO_OP` discriminator was meant to catch, but can't
   here because the output isn't byte-identical).

## Expected outcome → fix location

| Preflight result | Interpretation | Fix |
|---|---|---|
| partial overlap, MLP (or attn) keys miss | `normalize_lora_key` mis-maps the gemma4 MM `language_model` infix for that module class | `adapter_format.normalize_lora_key` (+ a unit test on the gemma4 key shape) — **adapter layer, durable** |
| zero overlap | full prefix miss despite a live output shift — re-examine whether the shift came from a non-LoRA path | as above, but re-confirm step 1 vs live behaviour first |
| full overlap | keys land; adapter genuinely applies | not a normalization bug → training/hyperparameter track (ADR 0003 open question) |

## Why this is safe to defer until the box

There is no local feedback loop for gemma (no GPU/built fork here), and the cause
is a partial-match that only the live tree can disambiguate. Guessing at
`normalize_lora_key` blind risks breaking the cases that already work (llama, qwen
PASS). The preflight *is* the loop; once it inspects gemma on the GPU image, step 1
alone likely localizes the fix to a single normalization rule.
