# Granite-4.0-H Hybrid LoRA Target Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ibm-granite/granite-4.0-h-tiny` (GraniteMoeHybrid: Mamba-2 + attention + grouped-expert MoE with a dense shared MLP) resolve a `.pt`-serveable LoRA target set and stop being false-skipped by the sweep preflight, then run it on the Spark to get a real terminal verdict.

**Architecture:** Two independent code changes plus one verification run. (1) The train-side resolver `resolve_target_modules` currently detects MoE only by a `.experts` submodule and excludes experts/router by `.experts` + leaf `gate`/`router`; Granite has none of those (its grouped experts + router live under `block_sparse_moe`, and it adds Mamba SSM layers), so it silently falls to the dense leaf-name path and would adapt the fused experts, the router, and the SSM projections — none `.pt`-serveable. We generalize MoE detection and exclude the `block_sparse_moe` and `mamba` subtrees, leaving LoRA on attention + the dense `shared_mlp`. (2) The offline preflight compares a *hardcoded* synthetic leaf set against the served tree; it lacks Granite's `shared_mlp.{input_linear,output_linear}` leaves, so it predicts zero overlap and skips the model before training. We add those leaves (strictly fail-open, like the earlier GLM-4 fix). (3) We deploy and re-run on `spark-147c` to learn the empirical verdict (PASS vs the expected serve-OK-but-NO_MEMORIZATION, since — like PhiMoE — the routed experts stay off the adapter, though Granite's always-on dense `shared_mlp` may carry memorization where pure-sparse MoEs cannot).

**Tech Stack:** Python, PyTorch `nn.Module` introspection, PEFT LoRA, transformers 5.x (`GraniteMoeHybridForCausalLM`), the vLLM `.pt`-adapter fork, the finetune-sweep harness (`test/finetune_sweep/`), pytest.

## Global Constraints

- **Lab-branch workflow:** commit directly on `georgi/finetune-sweep`. Keep the `ml/` and `test/` changes minimal and self-contained so they can later be cherry-picked into a short-lived PR off fresh `main`. Do NOT PR the whole lab branch.
- **Commit trailers:** every commit message ends with, on their own lines:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_017A9mGwAi6F3zjUzsjNUp7W`
- **Unit-test incantation** (host `.venv`/system python lack torch+pytest): run resolver tests with `PYTHONPATH=infra:ml uv run --with pytest --with torch python -m pytest <path> -q`; run the torch-free preflight seam tests with `PYTHONPATH=infra uv run --with pytest --with torch python -m pytest <path> -q`.
- **Preflight sync rule:** `DEFAULT_LORA_TARGETS` in `test/finetune_sweep/preflight.py` (host) and the `targets` dict inside its `_INTROSPECT_SCRIPT` string constant MUST stay identical — they are two copies of the same set (one runs on the host for unit tests, one runs in-image). Any change to one must be mirrored in the other in the SAME commit.
- **`ml/` snapshot freeze:** the sweep copies `ml/` into `jobs/<hash>/ml` only if absent, and the hash is deterministic from config, so a re-run of an unchanged model runs STALE code. The runner force-deletes prior job dirs before submit (`refresh_model_job_dirs`), but the box must be on the new commit: `git -C $HOME/projects/scalarlm pull --ff-only origin georgi/finetune-sweep` before any re-run.
- **SSH:** `ssh spark-147c` lands in `~`. The literal path `/home/georgi/projects/scalarlm` is filtered from command strings — always use `$HOME/projects/scalarlm` or `git -C`. Use bracketed `pgrep -af "[r]un_finetune_sweep"` to avoid self-match.
- **Granite ground truth** (from a meta-device HF init on the box, 2026-07-04) — the resolver and tests must match these exact names:
  - attention: `model.layers.{i}.self_attn.{q,k,v,o}_proj` (attention layers only)
  - dense shared MLP (serveable): `model.layers.{i}.shared_mlp.{input_linear,output_linear}`
  - grouped routed experts (NOT serveable): `model.layers.{i}.block_sparse_moe.{input_linear,output_linear}`
  - router (leaf name is `layer`): `model.layers.{i}.block_sparse_moe.router.layer`
  - Mamba-2 SSM (not adapted): `model.layers.{i}.mamba.{in_proj,out_proj}` (plus `conv1d`, a `Conv1d` not an `nn.Linear`, so never picked up)
  - `HAS .experts: False` — there is no `.experts` submodule anywhere.

---

## File Structure

- `ml/adapters/resolve_target_modules.py` — the train-side "all-linear" resolver. Modify `_is_moe_model` (detection) and `_moe_servable_linear_paths` (exclusions). One responsibility: turn the `all-linear` shorthand into a concrete, `.pt`-serveable target set.
- `test/unit/test_resolve_target_modules.py` — unit tests for the resolver. Add a Granite-hybrid synthetic fixture + tests.
- `test/finetune_sweep/preflight.py` — offline no-op preflight. Modify `DEFAULT_LORA_TARGETS` and the mirrored in-container `targets` dict.
- `test/unit/test_finetune_sweep_preflight.py` — unit tests for the preflight torch-free seams. Update the exact-set assertion in `test_synthesize_lora_keys_one_layer_standard_targets`.
- `test/finetune_sweep/finetune-sweep.yaml` — the Granite entry's annotation (verdict update in the final task).
- `docs/reports/2026-07-01-model-categories.md` — the Granite row/bullet (verdict update in the final task).

---

### Task 1: Generalize MoE detection + exclude `block_sparse_moe`/`mamba` (train-side resolver)

**Files:**
- Modify: `ml/adapters/resolve_target_modules.py:66-114` (`_is_moe_model`, `_moe_servable_linear_paths`)
- Test: `test/unit/test_resolve_target_modules.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature changes. `resolve_target_modules(model, "all-linear")` continues to return `list[str]` (sorted full paths for MoE models). Behaviour change only: a GraniteMoeHybrid-shaped model is now detected as MoE and resolves to attention + `shared_mlp` full paths, excluding `block_sparse_moe.*` and `mamba.*`.

- [ ] **Step 1: Write the failing test**

Add to `test/unit/test_resolve_target_modules.py` (after the existing `_MoeLike` class and its tests, e.g. after `test_moe_all_layers_sparse_adapts_attention_only`):

```python
class _GraniteHybridLike(nn.Module):
    """A miniature GraniteMoeHybrid ...ForCausalLM mirroring the real leaf naming
    (verified by a meta-device HF init of ibm-granite/granite-4.0-h-tiny):

    - attention (attention layers only): self_attn.{q,k,v,o}_proj
    - dense shared MLP (serveable, every layer): shared_mlp.{input_linear, output_linear}
    - grouped routed experts (NOT .pt-serveable): block_sparse_moe.{input_linear, output_linear}
    - router (leaf name `layer`, not gate/router): block_sparse_moe.router.layer
    - Mamba-2 SSM (not adapted): mamba.{in_proj, out_proj}

    There is NO `.experts` submodule — the resolver must detect MoE via
    `block_sparse_moe`. Layer 0 is a Mamba layer (mamba + moe + shared_mlp); layer 1
    is an attention layer (self_attn + moe + shared_mlp)."""

    def __init__(self, n_experts=4):
        super().__init__()

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _shared_mlp():
            return nn.ModuleDict(
                {
                    "input_linear": nn.Linear(8, 16, bias=False),
                    "output_linear": nn.Linear(8, 8, bias=False),
                }
            )

        def _block_sparse_moe():
            return nn.ModuleDict(
                {
                    # grouped experts: fused input/output linear, not .pt-serveable
                    "input_linear": nn.Linear(8, 16, bias=False),
                    "output_linear": nn.Linear(8, 8, bias=False),
                    # router — leaf name is `layer` (block_sparse_moe.router.layer)
                    "router": nn.ModuleDict({"layer": nn.Linear(8, n_experts, bias=False)}),
                }
            )

        def _mamba():
            return nn.ModuleDict(
                {
                    "in_proj": nn.Linear(8, 16, bias=False),
                    "out_proj": nn.Linear(8, 8, bias=False),
                }
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mamba": _mamba(),
                        "block_sparse_moe": _block_sparse_moe(),
                        "shared_mlp": _shared_mlp(),
                    }
                ),
                nn.ModuleDict(
                    {
                        "self_attn": _attn(),
                        "block_sparse_moe": _block_sparse_moe(),
                        "shared_mlp": _shared_mlp(),
                    }
                ),
            ]
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_granite_hybrid_adapts_attention_and_shared_mlp_only():
    # Granite has no `.experts` submodule — MoE detection must fire on
    # `block_sparse_moe`, and resolution must exclude the grouped experts + router
    # (`block_sparse_moe.*`) and the Mamba SSM (`mamba.*`), keeping attention and
    # the dense shared MLP — by full path, since shared_mlp and the experts share
    # the `input_linear`/`output_linear` leaf names.
    model = _GraniteHybridLike()
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.shared_mlp.input_linear",
        "layers.0.shared_mlp.output_linear",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
        "layers.1.shared_mlp.input_linear",
        "layers.1.shared_mlp.output_linear",
    ]
    assert not any(".block_sparse_moe." in name for name in result)  # no experts/router
    assert not any(".mamba." in name for name in result)  # no SSM projections
    assert not any("lm_head" in name for name in result)  # no output head
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=infra:ml uv run --with pytest --with torch python -m pytest test/unit/test_resolve_target_modules.py::test_granite_hybrid_adapts_attention_and_shared_mlp_only -q`

Expected: FAIL. With the current code, `_is_moe_model` returns False (no `.experts`), so resolution takes the dense leaf-name path and returns a leaf-name set like `['in_proj', 'input_linear', 'k_proj', 'layer', 'o_proj', 'out_proj', 'output_linear', 'q_proj', 'v_proj']` — not the expected full-path list. The `assert result == [...]` line fails.

- [ ] **Step 3: Write minimal implementation — generalize MoE detection**

In `ml/adapters/resolve_target_modules.py`, replace `_is_moe_model` (lines 66-75) with:

```python
def _is_moe_model(model) -> bool:
    """True if the model has routed MoE expert submodules whose fused LoRA a `.pt`
    adapter can't serve. Two container conventions are recognized:

    - `.experts` — Qwen3MoE / PhiMoE (a `ModuleList`/`ModuleDict` of per-expert or
      grouped expert projections), and
    - `.block_sparse_moe` — GraniteMoeHybrid, whose grouped experts and router live
      under `model.layers.{i}.block_sparse_moe.*` with NO `.experts` submodule.

    A `.pt` adapter that adapts the *fused experts* can't be served: vLLM's
    `FusedMoEWithLoRA.set_lora` wants a per-expert tensor *list* (gate/down/up,
    each `[num_experts, rank, dim]`), while the ScalarLM trainer exports stacked
    2-D tensors. Rather than reproduce vLLM's PEFT→fused-MoE conversion, we keep
    LoRA off the experts (and the router) and adapt everything else that *does*
    serve from a `.pt` adapter — see `_moe_servable_linear_paths`."""
    return any(
        ".experts" in name or ".block_sparse_moe" in name
        for name, _ in model.named_modules()
    )
```

- [ ] **Step 4: Write minimal implementation — exclude the `block_sparse_moe` and `mamba` subtrees**

In `ml/adapters/resolve_target_modules.py`, in `_moe_servable_linear_paths`, insert two exclusions immediately after the existing `.experts` check (currently line 103) and before the router leaf-name check (currently line 111):

```python
        if ".experts" in module_name:  # routed experts — not .pt-serveable
            continue
        if ".block_sparse_moe" in module_name:  # GraniteMoe grouped experts + router.layer
            continue
        if ".mamba" in module_name:  # Mamba-2 SSM projections (in_proj/out_proj) — not adapted
            continue
```

Then update the `_moe_servable_linear_paths` docstring exclusion bullets (currently lines 84-89) to read:

```python
    - the routed experts — `.experts` (Qwen3MoE/PhiMoE) or the whole
      `.block_sparse_moe` subtree (GraniteMoeHybrid grouped experts + `router.layer`);
      their fused LoRA isn't `.pt`-serveable (see `_is_moe_model`),
    - the router (leaf `gate` in Qwen3MoE or `router` in PhiMoE — adapting it
      would perturb expert selection, and PhiMoE's is an nn.Linear subclass
      returning a tuple that crashes PEFT's LoRA wrap; GraniteMoe's router is
      already covered by the `.block_sparse_moe` exclusion above),
    - the Mamba-2 SSM projections (`.mamba.*`) — nothing else in the sweep has
      state-space layers and their LoRA is untested/unserved, and
    - the output head.
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `PYTHONPATH=infra:ml uv run --with pytest --with torch python -m pytest test/unit/test_resolve_target_modules.py::test_granite_hybrid_adapts_attention_and_shared_mlp_only -q`

Expected: PASS (1 passed).

- [ ] **Step 6: Run the FULL resolver suite to verify no regression**

Run: `PYTHONPATH=infra:ml uv run --with pytest --with torch python -m pytest test/unit/test_resolve_target_modules.py -q`

Expected: all pass (the prior 11 + the new 1 = 12 passed). In particular the existing `_MoeLike` (`.experts`) and PhiMoE (`router`) tests must still pass — the new `.block_sparse_moe`/`.mamba` exclusions do not appear in those fixtures, and `_is_moe_model` still fires on `.experts`.

- [ ] **Step 7: Commit**

```bash
git add ml/adapters/resolve_target_modules.py test/unit/test_resolve_target_modules.py
git commit -m "$(cat <<'EOF'
fix(adapters): resolve GraniteMoeHybrid LoRA targets (block_sparse_moe + mamba)

resolve_target_modules detected MoE only via a `.experts` submodule and excluded
experts/router by `.experts` + leaf `gate`/`router`. GraniteMoeHybrid has none of
those: its grouped experts and router live under `block_sparse_moe.*` (router leaf
is `layer`), and it adds Mamba-2 SSM layers (`mamba.in_proj`/`out_proj`). So it
fell to the dense leaf-name path and would have adapted the fused experts, the
router, and the SSM projections — none `.pt`-serveable.

Generalize `_is_moe_model` to also fire on `.block_sparse_moe`, and exclude the
`.block_sparse_moe` and `.mamba` subtrees in `_moe_servable_linear_paths`, leaving
LoRA on attention + the dense `shared_mlp` (both serveable). Adds a Granite-hybrid
synthetic unit fixture + test.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017A9mGwAi6F3zjUzsjNUp7W
EOF
)"
```

---

### Task 2: Stop the preflight false-skip (add Granite `shared_mlp` leaves)

**Files:**
- Modify: `test/finetune_sweep/preflight.py:45-49` (`DEFAULT_LORA_TARGETS`) and `test/finetune_sweep/preflight.py:178-181` (the mirrored `targets` dict inside `_INTROSPECT_SCRIPT`)
- Test: `test/unit/test_finetune_sweep_preflight.py:57-73` (update the exact-set assertion)

**Interfaces:**
- Consumes: nothing new.
- Produces: `synthesize_lora_keys()` now also emits `model.layers.{i}.shared_mlp.input_linear...` and `...shared_mlp.output_linear...` keys. No signature change.

**Why this is safe (fail-open):** the preflight's `predicted_ok` is `overlap(...) > 0` — a permissive OR. Adding target leaves can only ADD overlap, never remove it, so it can never introduce a *new* false skip; at worst it lets through a model that turns out to no-op (caught downstream). Granite's served vLLM tree contains `model.layers.{i}.shared_mlp.input_linear` (confirmed present in `granitemoehybrid.py`'s module tree and `packed_modules_mapping`), so these leaves produce a real overlap and flip Granite from predicted-noop (skip) to predicted-ok (run).

- [ ] **Step 1: Update the failing test first**

In `test/unit/test_finetune_sweep_preflight.py`, extend the exact-set assertion in `test_synthesize_lora_keys_one_layer_standard_targets` (lines 57-73) by adding these two members to the expected set (inside the `{ ... }`, after the ChatGLM/GLM fused leaves):

```python
        # GraniteMoeHybrid dense shared-MLP leaves (prevents a false no-op skip on
        # granite-4.0-h-tiny, whose served tree exposes shared_mlp.input_linear).
        "model.layers.0.shared_mlp.input_linear.lora_A.default.weight",
        "model.layers.0.shared_mlp.output_linear.lora_A.default.weight",
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=infra uv run --with pytest --with torch python -m pytest test/unit/test_finetune_sweep_preflight.py::test_synthesize_lora_keys_one_layer_standard_targets -q`

Expected: FAIL with an assertion error — the produced set is missing the two `shared_mlp` keys (the implementation hasn't added the `shared_mlp` block yet).

- [ ] **Step 3: Add the `shared_mlp` block to the host `DEFAULT_LORA_TARGETS`**

In `test/finetune_sweep/preflight.py`, change `DEFAULT_LORA_TARGETS` (lines 45-49) to:

```python
DEFAULT_LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h"),
    "self_attention": ("query_key_value", "dense"),
    "shared_mlp": ("input_linear", "output_linear"),
}
```

Also extend the comment block above it (the `Two arch families are covered...` note, lines 35-44) with a third bullet so the rationale stays documented:

```python
#   - GraniteMoeHybrid (ibm-granite/granite-4.0-h-tiny, served by the fork's
#     granitemoehybrid.py): shared_mlp.{input_linear,output_linear}. The trainer's
#     resolver adapts the dense shared MLP (experts/router/SSM excluded), and those
#     leaves match vLLM's tree — without them the heuristic saw 0 overlap and wrongly
#     SKIPPED granite. Extra targets only ADD overlap, so this stays fail-open.
```

- [ ] **Step 4: Mirror the change in the in-container `targets` dict**

In the same file, inside the `_INTROSPECT_SCRIPT` string constant, change the `targets` dict (lines 178-181) to add the matching entry:

```python
        targets = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
                   "mlp": ("gate_proj", "up_proj", "down_proj",
                           "dense_h_to_4h", "dense_4h_to_h"),
                   "self_attention": ("query_key_value", "dense"),
                   "shared_mlp": ("input_linear", "output_linear")}
```

(This copy runs in-image; the host copy in Step 3 is what the unit test exercises. Per the Global Constraints "Preflight sync rule", both must change together.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=infra uv run --with pytest --with torch python -m pytest test/unit/test_finetune_sweep_preflight.py -q`

Expected: all pass. The `synthesize_lora_keys` set now includes the two `shared_mlp` keys, and the other preflight seam tests are unaffected.

- [ ] **Step 6: Commit**

```bash
git add test/finetune_sweep/preflight.py test/unit/test_finetune_sweep_preflight.py
git commit -m "$(cat <<'EOF'
fix(sweep): preflight no longer false-skips granite-4.0-h-tiny

The offline preflight compares a hardcoded synthetic LoRA leaf set against the
served vLLM tree; it lacked GraniteMoeHybrid's dense shared-MLP leaves
(shared_mlp.input_linear/output_linear), so it saw 0 overlap and skipped granite
before it ever trained (PRECHECK_NO_OP). Add the shared_mlp block to both the host
DEFAULT_LORA_TARGETS and the mirrored in-container targets. Strictly fail-open:
extra targets only add overlap, never remove it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017A9mGwAi6F3zjUzsjNUp7W
EOF
)"
```

---

### Task 3: Deploy, re-run granite on spark-147c, record the verdict

**Files:**
- Modify: `test/finetune_sweep/finetune-sweep.yaml` (Granite entry annotation)
- Modify: `docs/reports/2026-07-01-model-categories.md` (Granite row/bullet)
- Modify: `/home/georgi/.claude/projects/-home-georgi-projects-scalarlm/memory/archdiv-sweep-olmo-phimoe-granite.md` (result)

**Interfaces:**
- Consumes: the deployed Task 1 + Task 2 commits.
- Produces: a terminal Granite verdict (PASS / NO_MEMORIZATION / TRAIN_FAILED / serve failure) plus updated annotations.

- [ ] **Step 1: Push the fixes and deploy to the box**

```bash
git push origin georgi/finetune-sweep
ssh spark-147c 'git -C $HOME/projects/scalarlm pull --ff-only origin georgi/finetune-sweep && git -C $HOME/projects/scalarlm log --oneline -3'
```

Expected: the box HEAD shows the Task 1 and Task 2 commits. (Per the `ml/` snapshot-freeze constraint, the box must be on the new commit before re-running; the runner force-deletes prior job dirs, so the stale granite `ml/` snapshot won't mask the fix.)

- [ ] **Step 2: Confirm no runner is already active**

Run: `ssh spark-147c 'pgrep -af "[r]un_finetune_sweep" || echo NO_RUNNER'`

Expected: `NO_RUNNER`. If a runner is active, wait for it to finish or stop it before launching (single GPU arbiter).

- [ ] **Step 3: Launch granite alone on cuda-spark, detached**

Run:
```bash
ssh spark-147c 'cd $HOME/projects/scalarlm && nohup python3 -u test/finetune_sweep/run_finetune_sweep.py --target cuda-spark --models ibm-granite/granite-4.0-h-tiny --train-timeout 5400 --serve-timeout 3600 > $HOME/sweep-granite-20260704.log 2>&1 < /dev/null & echo "launched pid $!"'
```

Expected: prints `launched pid <N>`. Then verify it is alive and past preflight:

Run: `ssh spark-147c 'sleep 20; pgrep -af "[r]un_finetune_sweep"; tail -n 15 $HOME/sweep-granite-20260704.log'`

Expected: the runner PID is listed, and the log shows a `[preflight] ibm-granite/granite-4.0-h-tiny: predicted_ok=True` line (the Task 2 fix). **If it still shows `predicted_ok=False` / `PRECHECK_NO_OP`** — the fork's serve-time normalization does not preserve the `shared_mlp.input_linear` leaf. Fallback: kill the runner (`ssh spark-147c 'pkill -f "[r]un_finetune_sweep"'`) and relaunch the exact command above with `--no-preflight` appended, which forces the train+serve cycle regardless of the preflight prediction, so the empirical verdict is still obtained.

- [ ] **Step 4: Poll to a terminal verdict**

Poll every ~10 min (the GB10 is single-GPU, phase-scaled; granite ~7B trains then serves sequentially):

Run: `ssh spark-147c 'pgrep -af "[r]un_finetune_sweep" || echo NO_RUNNER; tail -n 8 $HOME/sweep-granite-20260704.log'`

Trust API/container state over the log if they disagree (log buffering can lag). The run is terminal when `NO_RUNNER` prints and the log's results table shows one of:
- **PASS** — trained + served + memorized. (Possible if the always-on dense `shared_mlp` carries memorization.)
- **NO_MEMORIZATION** — served but the golden string didn't reproduce. This is the *expected* outcome by analogy to PhiMoE (routed experts stay off the adapter); if the training loss descended but was cut off, note it as a budget-vs-capacity question, not a bug (see the NO_MEM loss-curve diagnostic).
- **TRAIN_FAILED** — inspect `jobs/<hash>/slurm-1.out` on the box for the traceback; a crash in a `shared_mlp` or attention LoRA wrap would be a new finding (unlike PhiMoE, `shared_mlp.input_linear`/`output_linear` are plain `nn.Linear` returning tensors, so no tuple-return crash is expected).
- **serve failure** — the adapter trained but vLLM couldn't load/serve it; capture the fork error (this is open question #2: whether `shared_mlp` LoRA serves through the FusedMoE path).

- [ ] **Step 5: Record the verdict in the yaml**

In `test/finetune_sweep/finetune-sweep.yaml`, replace the Granite entry's `RESULT 2026-07-04: PRECHECK_NO_OP ...` annotation block with the new empirical result, e.g. (fill `<VERDICT>` and the timings from the results table):

```yaml
    # RESULT 2026-07-04 (rerun after resolver+preflight fix): <VERDICT> on cuda-spark
    # (restart <r>s / train <t>s / serve <s>s). Resolver now emits attention +
    # shared_mlp full paths (block_sparse_moe experts+router and mamba SSM excluded);
    # preflight no longer skips it. <one line on memorization: e.g. served but
    # NO_MEMORIZATION — dense shared_mlp alone didn't carry the golden string, routed
    # experts stay off the .pt (separate-expert converter territory), OR PASS.>
```

- [ ] **Step 6: Record the verdict in the report**

In `docs/reports/2026-07-01-model-categories.md`, update the Granite bullet (currently marked `⛔ PRECHECK_NO_OP`) and the blocked-models table row to the new verdict + one-line root cause, mirroring the yaml. If the verdict is PASS, move it out of the blocked table and mark the bullet `✅ PASS`.

- [ ] **Step 7: Update the memory file**

In `/home/georgi/.claude/projects/-home-georgi-projects-scalarlm/memory/archdiv-sweep-olmo-phimoe-granite.md`, update the granite paragraph: note the resolver+preflight fix landed (commits), the exclusion approach (`block_sparse_moe` + `mamba` subtrees, keep `shared_mlp`), and the empirical re-run verdict.

- [ ] **Step 8: Commit the annotations**

```bash
git add test/finetune_sweep/finetune-sweep.yaml docs/reports/2026-07-01-model-categories.md
git commit -m "$(cat <<'EOF'
docs(sweep): record granite-4.0-h-tiny rerun verdict after resolver+preflight fix

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017A9mGwAi6F3zjUzsjNUp7W
EOF
)"
git push origin georgi/finetune-sweep
```

---

## Open Questions (resolved empirically by Task 3, not by code)

1. **Does granite memorize with only attention + `shared_mlp` adapted?** The routed experts stay off the `.pt` (same as PhiMoE). Unlike a pure-sparse MoE, granite runs a dense `shared_mlp` on every layer, which *might* carry the memorization signal. Task 3's verdict answers this: PASS ⇒ yes; NO_MEMORIZATION ⇒ no, and reaching the golden string needs the separate-expert converter (a larger, separate effort — the same wall PhiMoE hits).
2. **Does vLLM serve a `shared_mlp` LoRA through the FusedMoE path?** `input_linear` is in granite's `packed_modules_mapping`, but it is entangled with the fused-expert path. Task 3's serve phase is the test; a serve failure here is a distinct finding from a train failure.

## Self-Review notes

- **Spec coverage:** train-side resolver (Task 1), preflight false-skip (Task 2), empirical verdict + docs/memory (Task 3) — the three issues identified in the investigation are each covered.
- **Type consistency:** `_is_moe_model` and `_moe_servable_linear_paths` keep their existing signatures; `resolve_target_modules` return type unchanged. `DEFAULT_LORA_TARGETS` stays `dict[str, tuple[str, ...]]`. The synthetic fixture path prefix is `layers.` (top attribute is `layers`), matching the existing `_MoeLike` tests' expected outputs.
- **Not in scope (YAGNI):** a separate-expert / grouped-expert LoRA converter for granite's `block_sparse_moe` experts (only needed if Task 3 returns NO_MEMORIZATION and we decide to chase full memorization); adapting the Mamba SSM layers (deliberately excluded).
