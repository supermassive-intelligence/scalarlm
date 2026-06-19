# Fine-tune sweep dry-test: no-op discriminator + offline preflight

**Date:** 2026-06-11
**Status:** approved design, pre-implementation
**Component:** `test/finetune_sweep/`
**Amended:** 2026-06-18 — the offline preflight is now a **single faithful backend**: it
builds vLLM's own module tree on the meta device and runs the trainer's would-be keys
through the fork's **real two-pass** normalization (`normalize_lora_key` +
`_renormalize_lora_sd_for_model`) before the overlap check. The earlier HF-meta backend was
designed and then rejected: it mispredicts every model whose HF and vLLM trees diverge —
which, per the fork's own `normalize_lora_key`, includes decoder-only models (Qwen3.5),
not just VL-wrapped ones. A one-pass overlap check is likewise wrong, because pass 1
deliberately over-maps `model.layers.*` and relies on pass 2 to correct it per model.

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
    sample_adapter_keys: list[str]   # post-(full)-normalize, for the hint
    sample_base_modules: list[str]
    error: str = ""                  # build/introspection failure

def run_preflight(
    model_ids: list[str],
    compose_service: str,
) -> dict[str, PreflightResult]: ...
```

`run_preflight` issues throwaway container commands
(`docker compose run --rm --no-deps <service> python3 -c <script>`) that run entirely
inside the cray image (which has `transformers`, vLLM, and the fork). The check is purely
structural: synthesize the trainer's would-be LoRA keys, push them through the fork's
**actual serve-time normalization**, and ask whether the resulting module paths exist in
the **served model's** vLLM module tree.

#### Single backend — vLLM tree on a meta device (faithful, cheap)

The no-op bug class is fundamentally "trainer-shaped keys don't land on vLLM's module
tree," so the only faithful reference is vLLM's *own* tree — never an HF meta-model. An
HF-meta backend was considered and **rejected** (it mispredicts every model whose HF and
vLLM trees diverge, which — per the fork's `normalize_lora_key` — includes decoder-only
models like Qwen3.5 that vLLM wraps under `language_model.model.layers.*`). One backend,
always vLLM.

Crucially, we do **not** instantiate the full `LLM` engine (KV-cache profiling, CUDA
graphs, a forward pass — heavy, and GPU-only). We build only the `nn.Module` tree on the
**meta device**, which is enough for `named_modules()` and is faithful because it is the
same code path the engine runs:

1. `EngineArgs(model=id, load_format="dummy", enforce_eager=True).create_engine_config()`
   → a `VllmConfig`. We let `EngineArgs` assemble it (rather than hand-construct
   `ModelConfig`) because `load_format` lives on `LoadConfig`, not `ModelConfig`, and the
   config field layout shifts between vLLM versions — `create_engine_config` is the engine's
   own assembly path, so it tracks the fork. `load_format="dummy"` skips weight download/read.
2. Build the tree **inside the config context, with a single-process parallel group**.
   Both `initialize_model_parallel` and `initialize_model` call `get_current_vllm_config()`,
   and the model reads the PP group / TP world size even at TP=1, so the order is:
   - `init_distributed_environment(world_size=1, rank=0, local_rank=0,
     distributed_init_method="tcp://127.0.0.1:<free-port>", backend="gloo")` — `gloo`
     works on both the `cpu` and `cuda-docker` images (we never run a collective; the
     groups just have to exist), and an ephemeral port avoids clashes between concurrent runs.
   - `with set_current_vllm_config(vllm_config):` → `initialize_model_parallel(1, 1)` then,
     `with torch.device("meta"):`, `initialize_model(vllm_config)`
     (`vllm.model_executor.model_loader.utils.initialize_model`) → the model with parameters
     on `meta`. No weights are read or allocated; runs on CPU, needs no GPU, so the same
     preflight works for the `cpu` and `cuda-docker` targets alike.
3. Collect `base_modules = {name for name, _ in model.named_modules()}`.
4. Synthesize the trainer's would-be LoRA keys from the standard target leaf names —
   `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}` (what the trainer
   LoRA-ifies; avoids `lm_head`/embeddings) — shaped exactly as the trainer exports them:
   `f"model.layers.{i}.{...}.{tgt}.lora_A.default.weight"`. We only need layer 0 (the
   per-layer pattern repeats), keeping `n_total` small.

**Mirror the real two-pass normalization — do not reimplement it.** Serve-time matching is
two passes, not one, and the preflight must run the same code or it will drift:

   - **Pass 1 — `normalize_lora_key`** (static; `vllm.tokenformer.adapter_format`): maps
     `model.layers.*` → `language_model.model.layers.*` for *all* decoder keys, strips the
     PEFT `.default.` and ClippableLinear `.linear.` segments.
   - **Pass 2 — `PTWorkerLoRAManager._renormalize_lora_sd_for_model`** (runtime): calls
     `_detect_model_layers_prefix()` against the *live* tree, undoes pass 1's prefix, and
     re-applies the **correct** prefix for this model. This is the pass that fixes pass 1's
     deliberate over-mapping for true `model.layers.*` models.

   The preflight runs pass 2 through a throwaway `PTWorkerLoRAManager` **subclass** whose
   `__init__` only sets `self._adapter_manager.model` to the meta-tree (skipping the heavy
   `LRUCacheWorkerLoRAManager.__init__`) — that subclass keeps both
   `_renormalize_lora_sd_for_model` and the `_detect_model_layers_prefix()` it calls
   resolvable, and `self._adapter_manager.model.named_modules()` is the only attribute either
   touches. It feeds the synthesized keys through both passes, strips each normalized key's
   `.lora_A.weight` tail to recover its module path, and computes overlap exactly as
   `_warn_on_zero_base_match` does:

   `n_overlap = len(lora_module_paths & base_modules)`; `n_total = len(lora_module_paths)`.

5. **`predicted_ok = n_overlap > 0`** — any single match passes, mirroring the fork's
   permissive `_warn_on_zero_base_match` (partial overlap is benign; only *zero* overlap
   reliably predicts the silent no-op). Emit JSON per model: `{predicted_ok, n_overlap,
   n_total, sample_adapter_keys, sample_base_modules, error}`.

Because it walks vLLM's *real* tree through the *real* normalization, it predicts both the
**prefix-mismatch** class (zero overlap) and the **collision** class — the `49dd610de`
Gemma4 bug, where decoder LoRA keys re-prefix onto the `vision_tower.*` subtree →
`ADAPTER_NOT_LOADED`.

The host parses the JSON into `PreflightResult`s. A build/introspection `error` is
treated as **not** a no-op prediction (fail open — run the model rather than skip it on
a preflight crash).

Why it catches the prefix bug exactly: for a decoder-only model whose vLLM tree is
`model.layers.*`, pass 1 maps the key to `language_model.model.layers.*`, then pass 2
detects the real `model.layers.` prefix and re-applies it → the module path is in the set →
`predicted_ok=True`. If a future fork regression breaks the detect-and-reprefix so the
final path lands in neither tree, overlap is zero → `predicted_ok=False`. The preflight
tracks serve-time behavior because it *is* serve-time behavior.

### Control flow in `main`

1. Parse args (new `--no-preflight` flag).
2. Filter models.
3. Unless `--no-preflight`: `pf = run_preflight(model_ids, compose_service)`.
   - For each model with `predicted_ok=False`: append a
     `Result(outcome=PRECHECK_NO_OP, hint=<sample diff>)` and **exclude** it from the
     run list.
4. Run the surviving models through `run_model` as today.
5. Write reports (now including preflight-skipped results).

`compose_service` is read from the manifest's target config
(`manifest["targets"][target]["compose_service"]`). **k8s targets have no
`compose_service`**, so the preflight is Compose-only; for a k8s target (or any target
missing `compose_service`) `main` skips the preflight and runs every model, exactly as
`--no-preflight` would.

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
  in-container introspection script (meta-tree build + two-pass normalization overlap),
  and a pure `overlap(base_modules, lora_module_paths)` helper. The script body is a module
  constant so the host can unit-test the host-side parsing without an image.
- `test/finetune_sweep/test_finetune_sweep.py` (or existing test module) — unit tests.

## Testing

The split is deliberate: anything that needs the vLLM/transformers stack runs **in the
cray image on the box**; the host unit tests cover only torch-free seams.

- **`classify_result`** (host) — table test covering the new `ADAPTER_NO_OP` branch and its
  precedence (no-op-and-not-memorized → `ADAPTER_NO_OP`; no-op-but-golden-present →
  not `ADAPTER_NO_OP`).
- **`overlap` helper** (host) — pure set logic: zero intersection → `predicted_ok=False`;
  partial/full intersection → `predicted_ok=True` (the permissive `n_overlap > 0` rule).
  Takes plain string sets, so no torch needed.
- **Key synthesis** (host) — the function that builds `model.layers.0.<...>.lora_A.default.weight`
  for the standard target leaves emits exactly the expected strings.
- **Host-side JSON parsing** (host) — `run_preflight` parsing a captured JSON line into a
  `PreflightResult`, including the `error` fail-open path (a build crash → `predicted_ok`
  absent/false but the model is **not** skipped at the `main` level).
- **Two-pass fidelity** (in-image, run on the box) — for a decoder-only model whose vLLM
  tree is `model.layers.*` (e.g. `Qwen/Qwen2.5-0.5B`), assert the synthesized keys overlap
  the vLLM set *after both passes* (the case a one-pass / HF-meta check would mispredict).
  This is the regression that justifies running through the real `_renormalize_lora_sd_for_model`.
- **Integration** — the meta-tree `docker compose run` path is exercised by running the
  sweep on the box; not unit-tested on the host (no image there).

## Risks / caveats

- **Greedy-decoding dependency.** The discriminator relies on `temperature=0`
  determinism; documented and enforced in `generate()`.
- **Meta-tree faithfulness.** The preflight enumerates `named_modules()` on a meta-device
  build via `initialize_model`, the same code the engine runs, so the tree matches serve
  time without weights or a GPU. The one thing it does **not** model is tensor-parallel
  sharding (the sweep runs TP=1, so this is moot today); if a target ever sets TP>1, the
  module *names* are unaffected by sharding, so the overlap check still holds.
- **Coupling to fork internals.** The preflight subclasses `PTWorkerLoRAManager` (from
  `hybrid_adapter_manager`) to reach the underscored `_renormalize_lora_sd_for_model` /
  `_detect_model_layers_prefix`. This is deliberate — faithfulness *requires* running the
  same code — but it means a fork refactor that renames the class or those methods breaks
  the preflight loudly (an `AttributeError`/`ImportError` → `error` → fail-open run), never
  silently. Accepted: a loud break beats a silent misprediction.
- **Fused modules.** The check compares against unfused HF-target leaf names, while vLLM
  fuses q/k/v and gate/up into single modules — so one fused vLLM module can back several
  LoRA targets. The permissive `n_overlap > 0` rule absorbs this (partial overlap passes),
  so fusion does not produce false no-op predictions.
- **Preflight failures fail open** (run the model), so a quirk on a novel arch never
  silently skips a model. **Every** preflight result is now logged (predicted_ok / overlap
  / error), not just the skips, so a fail-open is loud — a wholly-broken preflight can't
  masquerade as "all predicted OK" (the 2026-06-18 silent-no-op fix).
- **Needs the serving image already built — gated, never built on demand** (2026-06-18
  fix). `docker compose run <service>` reuses the service image when present but silently
  tries to *build* it when absent — and that build fails on the `BASE_NAME` build arg only
  `./scalarlm up` supplies, so the captured output is build chatter, not the script's JSON
  → every model fail-opens *silently*. `run_preflight` now checks the image exists first
  (`docker compose config --images` + `docker image inspect`) and, if not, skips the whole
  preflight with one clear reason. So the preflight is effective only on a **warm box**
  (image built by a prior `./scalarlm up`/sweep); on a cold box it cleanly no-ops. It runs
  in the target's own `compose_service` image (e.g. `cray-nvidia`), which is the image that
  exists on a GPU box — *not* a separate CPU image, which a GPU box never builds.
- **Compose-only.** The preflight needs a `compose_service` to issue `docker compose run`;
  k8s targets have none, so they run unfiltered (the in-sweep `ADAPTER_NO_OP` discriminator
  is still ground truth there).
