# Fine-tune sweep dry-test: no-op discriminator + offline preflight

**Date:** 2026-06-11
**Status:** approved design, pre-implementation
**Component:** `test/finetune_sweep/`

## Problem

End-to-end LoRA fine-tuning can train correctly yet serve **base output** because
the adapter silently fails to apply (see
`docs/reports/lora-serving-noop-investigation.md` — the fork's `normalize_lora_key`
stripped the `model.` prefix for causal LMs, so every module was dropped at
activation). The existing sweep (`run_finetune_sweep.py`) could not surface this:
`classify_result` only checks `expected_output in adapter_text`, so a served-but-no-op
adapter scores `NO_MEMORIZATION`, which is listed as **non-failing**. That ambiguity
forced a long manual forensic dive to distinguish "didn't memorize" from "adapter
never applied" — and would force it again for every new model/architecture.

## Goal

Make the no-op failure **loud, specific, and cheap to detect**, so sweeping N models
yields a labeled table where each failure is self-explanatory — no per-model
debugging. Two mechanisms:

1. An **in-sweep discriminator** that distinguishes "adapter applied but didn't
   memorize" from "adapter served base output" (the no-op bug class).
2. An **offline preflight** that predicts the no-op class before paying for the
   per-model stack restart + train, and skips those models.

Non-goals: fixing the fork bug (done separately on
`fix/normalize-lora-key-causal-lm-prefix`); changing the per-model restart model
(ADR 0003); testing memorization quality beyond the golden-string check.

## Design

### New outcomes

Added to the outcome constants in `run_finetune_sweep.py`:

- **`ADAPTER_NO_OP`** — adapter loaded and served, but its output is byte-identical
  to the baseline → LoRA isn't being applied. **Failing.**
- **`PRECHECK_NO_OP`** — offline preflight predicted zero module overlap; the model's
  restart/train/serve was skipped. **Failing.**

`NON_FAILING_OUTCOMES` stays `{PASS, SKIPPED, NO_MEMORIZATION}`. Both new outcomes are
failing, so the existing exit-code logic (`main`, returns 1 on any non-failing-set
outcome) flags them. `NO_MEMORIZATION` now has a precise meaning: the adapter *changed*
the output but did not reproduce the golden string.

### In-sweep discriminator

In `run_model`:

- Keep the **full** baseline string (`baseline_full = baseline[0]`), not only the
  truncated `baseline_sample`.
- Compute `no_op = adapter_loaded and adapter_text == baseline_full`.
- `classify_result` gains an `adapter_is_noop: bool` argument:

  ```
  if train_status in (FAILED, CANCELLED):     return TRAIN_FAILED
  if train_status == TIMEOUT:                 return TRAIN_TIMEOUT
  if not checkpoint_lora_keys_ok(keys):       return BAD_CHECKPOINT
  if not adapter_loaded:                       return ADAPTER_NOT_LOADED
  if adapter_is_noop and not memorized:        return ADAPTER_NO_OP
  return PASS if memorized else NO_MEMORIZATION
  ```

**Determinism requirement:** exact-equality is only valid under greedy decoding.
`generate()` gains a `temperature` parameter (default `0.0`) and the baseline and
adapter calls pass `temperature=0`. (Today they send no temperature → server default,
which may be non-deterministic and would make the equality check unreliable.)

### Offline preflight (`preflight.py`, new module)

A new sibling module keeps the model-introspection / container concern out of the
runner. Public surface:

```python
@dataclass
class PreflightResult:
    model_id: str
    predicted_ok: bool
    n_overlap: int
    n_total: int
    sample_adapter_keys: list[str]   # post-normalize, for the hint
    sample_base_modules: list[str]
    error: str = ""                  # build/introspection failure

def run_preflight(model_ids: list[str], compose_service: str) -> dict[str, PreflightResult]: ...
```

`run_preflight` issues **one** throwaway container command
(`docker compose run --rm --no-deps <service> python3 -c <script>`) that runs entirely
inside the cray image (which has `transformers`, `accelerate`, and the fork). For each
model the in-container script:

1. `AutoConfig.from_pretrained(id)` → `with init_empty_weights(): AutoModelForCausalLM.from_config(cfg)`
   (fall back to `AutoModel` when not causal). Meta device — no GPU, no weights beyond
   the config.
2. Collect `named_modules()`; the LoRA targets are the `nn.Linear` modules whose leaf
   name is a standard LoRA target — `{q_proj, k_proj, v_proj, o_proj, gate_proj,
   up_proj, down_proj}` (matches what the trainer LoRA-ifies; avoids `lm_head` /
   embeddings the trainer doesn't touch).
3. Synthesize the trainer's would-be keys: `f"{path}.lora_A.default.weight"`.
4. `normalize_lora_key(key)` (imported from `vllm.tokenformer.adapter_format`) → parse
   off the `.lora_A.weight` tail → check the module name is in the model's module set.
   `n_total` = number of synthesized targets; `n_overlap` = how many normalized module
   names are present in the module set.
5. **`predicted_ok = n_overlap > 0`** — any single match passes, mirroring the fork's
   permissive `_warn_on_zero_base_match` (partial overlap is benign; only *zero* overlap
   reliably predicts the silent no-op). Emit JSON per model: `{predicted_ok, n_overlap,
   n_total, sample_adapter_keys, sample_base_modules, error}`.

The host parses the JSON into `PreflightResult`s. A build/introspection `error` is
treated as **not** a no-op prediction (fail open — run the model rather than skip it on
a preflight crash).

Why it catches the bug exactly: for `model.layers.0.self_attn.o_proj`, the buggy
normalize yields `layers.0…` (not in the module set) → `predicted_ok=False`; after the
fork fix it yields `model.layers.0…` (in the set) → `predicted_ok=True`. At the HF
level all seven LoRA targets exist as distinct named modules, so vLLM's q/k/v fusion
doesn't create false mismatches here.

### Control flow in `main`

1. Parse args (new `--no-preflight` flag).
2. Filter models.
3. Unless `--no-preflight`: `pf = run_preflight(model_ids, compose_service)`.
   - For each model with `predicted_ok=False`: append a
     `Result(outcome=PRECHECK_NO_OP, hint=<sample diff>)` and **exclude** it from the
     run list.
4. Run the surviving models through `run_model` as today.
5. Write reports (now including preflight-skipped results).

`compose_service` is read from the manifest's target config (already present:
`manifest["targets"][target]["compose_service"]`).

### Root-cause hint

New field `Result.hint: str`, surfaced as a report column.

- `PRECHECK_NO_OP`: hint = formatted `sample_adapter_keys` vs `sample_base_modules`
  from the `PreflightResult`.
- in-sweep `ADAPTER_NO_OP`: after detecting `no_op`, scrape the container for the
  fork's own warning — `docker compose logs --no-color <service>` filtered to
  `NONE of its .* match the base model` — and attach the matching line (or empty if
  absent). Runs on the host.

### Reporting

`write_reports` adds a `Hint` column to the markdown table and includes `hint` in the
JSON. Preflight-skipped models appear as ordinary rows with `PRECHECK_NO_OP`.

## Files

- `test/finetune_sweep/run_finetune_sweep.py` — new outcome constants; `Result.hint`;
  `classify_result(..., adapter_is_noop)`; `generate(..., temperature=0.0)`; `run_model`
  (full-baseline capture, `no_op`, log-scrape on no-op); `main` (preflight wiring,
  `--no-preflight`); `write_reports` (hint column).
- `test/finetune_sweep/preflight.py` — **new**: `PreflightResult`, `run_preflight`, the
  in-container introspection script, and the pure overlap-check helper.
- `test/finetune_sweep/test_finetune_sweep.py` (or existing test module) — unit tests.

## Testing

- **`classify_result`** — table test covering the new `ADAPTER_NO_OP` branch and its
  precedence (no-op-and-not-memorized → `ADAPTER_NO_OP`; no-op-but-golden-present →
  not `ADAPTER_NO_OP`).
- **Preflight pure logic** — a `keys_overlap(module_names, target_paths, normalize_fn)`
  helper unit-tested with a fake module-name set and a stub (and, where importable, the
  real `normalize_lora_key`): a `model.`-stripping stub yields zero overlap (predict
  no-op); an identity/correct stub yields full overlap.
- **Integration** — the meta-build + `docker compose run` path is exercised by running
  the sweep on the box; not unit-tested on the host (no image there).

## Risks / caveats

- **Greedy-decoding dependency.** The discriminator relies on `temperature=0`
  determinism; documented and enforced in `generate()`.
- **Preflight is a heuristic predictor.** It uses HF meta-model module names, which
  match vLLM's for the prefix-class bug but can differ for fused modules. It is a fast
  *negative* filter; the in-sweep `ADAPTER_NO_OP` discriminator remains ground truth.
  `--no-preflight` exists to bypass it if a false positive is suspected.
- **Preflight build failures** fail open (run the model), so a `from_config` quirk on a
  novel arch never silently skips a model.
