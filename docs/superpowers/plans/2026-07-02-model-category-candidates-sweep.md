# Model-Category Candidates Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 10 "Potential Sweep Candidates" from `docs/reports/2026-07-01-model-categories.md` §Causal LM and §Multimodal to `test/finetune_sweep/finetune-sweep.yaml`, then run the finetune sweep for exactly those 10 models on the `cuda-spark` target on `spark-147c` (reached over Tailscale SSH as `spark-147c`), and record what actually happens (PASS / train-only-no-serve / hard failure) for each.

**Architecture:** The sweep harness (`test/finetune_sweep/run_finetune_sweep.py`) already exists and is validated end-to-end (fp32/bf16 dense causal LMs, multimodal Qwen2-VL, MoE up to 30B, all phase-scaled on `cuda-spark`). No new harness code is expected — this is config (new `models:` entries) plus a git-sync step (spark-147c is 15 commits behind local and has its own now-superseded uncommitted WIP) plus operating the existing `--models` filter to scope the run to just the 10 new entries. If a candidate fails in a way that reveals a real harness bug (not an expected novel-arch serve no-op or load-time TRAIN_FAILED outcome), fix it the same way prior entries in this file were fixed (see `git log --oneline` on this branch for precedent), but that is diagnosis-driven, not pre-planned here.

**Tech Stack:** Python 3 (host-side runner, no deps beyond stdlib + pyyaml), Docker Compose (`cray-spark` service), vLLM fork, HF `transformers`.

## Global Constraints

- Never touch `ml/`/`infra/` behavior beyond what's already committed on `georgi/finetune-sweep` — this branch stays a lab branch per [[finetune-sweep-lab-branch-workflow]]; only commit here, don't open a PR for the branch as a whole.
- `cuda-spark` target is `phase_scaled: true` (train and serve run as two sequential restarts, peak GPU = 1 model) — do not remove that flag.
- LoRA serve support is NOT gated by the 3-class ScalarLM/Tokenformer registry in `infra/cray_infra/adapters/model/models.py:114-118` (that registry is for the Tokenformer adapter type). LoRA `.pt` adapters serve through the fork's hybrid-LoRA path over any arch vLLM natively supports, if the fork's key normalization handles that arch's key prefixes. Empirically, phi-4 (Phi3), Qwen3-8B (Qwen3), Mistral-7B (Mistral) and Qwen2-VL all full-PASS despite none being in that 3-class list. So set expectations by ARCH PRECEDENT: Phi-4-mini (Phi3) and Qwen3-* (Qwen3) have PASS precedent; Qwen2.5-VL (Qwen2_5_VL) tracks Qwen2-VL's PASS. The genuinely-novel serve cases are GLM-4-9B (Glm4), Pixtral, InternVL3, Molmo — for these a serve no-op (`ADAPTER_NO_OP`) or a load-time `TRAIN_FAILED` (InternVL3 vision_config / Molmo trust_remote_code) is a valid, useful diagnostic outcome — do not treat it as a bug to fix unless the report says otherwise.
- No model in this batch is gated per the source report (double-check the HF model card at add-time regardless — report flags this as "should be re-verified").
- Never run `docker`/`./scalarlm up`/GPU work on this local machine — it has no GPU/Docker. All execution happens over SSH on `spark-147c`.
- `adapters.lora.gate_gb` is a NO-OP on cuda-spark: `_vram_for_gate` returns `None` for the GB10's unified memory (nvidia-smi `memory.free` = `[N/A]`), so `gate_model` skips the VRAM check (`run_finetune_sweep.py:115-116, 772-780`). The values are kept sane only for cuda-docker portability; on spark, actual fit is governed by real footprint (`dtype`) vs the 128GiB pool, not the gate. The only per-model knobs that affect a spark run are `dtype` (float32 vs bfloat16) and the step budget (`max_steps`/`warmup_steps`/`steps_per_checkpoint`).
- The runner (`main()`) exits `1` if ANY model's outcome ∉ `{PASS, SKIPPED, NO_MEMORIZATION}`, but still runs and records every model in the invocation first. Expected hard-fail outcomes (Molmo `TRAIN_FAILED`, a novel-arch `ADAPTER_NO_OP`) are normal data points, not a reason to stop — see Task 4's non-`set -e` launch structure.

---

### Task 1: Reconcile git state between local and spark-147c

**Files:** none changed by code; this task is `git`/`ssh` operations only.

**Interfaces:**
- Consumes: local branch `georgi/finetune-sweep` at commit `8243af0` (current HEAD), with uncommitted changes to `docker-compose.yaml`, `ml/adapters/resolve_target_modules.py`, `ml/cray_megatron/models/load_model.py`, `test/finetune_sweep/finetune-sweep.yaml`, `test/finetune_sweep/run_finetune_sweep.py`, `test/unit/test_resolve_target_modules.py`.
- Produces: `spark-147c:~/projects/scalarlm` on `georgi/finetune-sweep` at the same commit as local's pushed HEAD, working tree clean of the now-superseded stale diffs, with `jobs/`, `test/finetune_sweep/results/`, `vllm/` untouched (untracked, git checkout doesn't touch them).

- [ ] **Step 1: Confirm current local uncommitted changes are what you intend to commit**

Run: `git status --short && git diff --stat`

Expected: matches the 6 modified files listed in the session's git status snapshot. If anything unexpected appears, stop and investigate before continuing.

- [ ] **Step 2: Commit local WIP with a descriptive message**

```bash
git add docker-compose.yaml ml/adapters/resolve_target_modules.py \
  ml/cray_megatron/models/load_model.py test/finetune_sweep/finetune-sweep.yaml \
  test/finetune_sweep/run_finetune_sweep.py test/unit/test_resolve_target_modules.py
git commit -m "$(cat <<'EOF'
chore(sweep): checkpoint in-flight fixes before candidate sweep

Lab-branch checkpoint of several distinct in-flight changes (each to be
extracted to its own PR off fresh main later); bundled here only so the
git-sync to spark-147c has a clean target:

- resolve_target_modules: MoE LoRA now adapts attention + dense-MLP via
  full dotted paths, excluding routed .experts / router gate / output head
  so the .pt adapter stays FusedMoEWithLoRA-servable (+ rewritten tests).
- load_model: multimodal AutoModelForImageTextToText dispatch via
  is_multimodal(); on-device device_map load to avoid the CPU+GPU 2x-peak
  OOM on the GB10 unified-memory pool.
- run_finetune_sweep: phase-scaled single-GPU-owning train/serve phases.
- docker-compose: SCALARLM_SERVER_LIST passthrough + vllm/lora bind mount.
- finetune-sweep.yaml: sweep config adjustments.
EOF
)"
```

Expected: commit succeeds; `git status --short` now shows only the untracked `docs/`, `vllm/` etc. paths already present at session start (unrelated to this task).

- [ ] **Step 3: Push the branch**

Run: `git push origin georgi/finetune-sweep`

Expected: push succeeds (this branch already tracks `origin/georgi/finetune-sweep` — confirmed by `git log origin/georgi/finetune-sweep..HEAD` being empty before Step 2's commit).

- [ ] **Step 4: Verify spark-147c's uncommitted diffs are fully superseded (re-check, don't trust memory)**

Run (from local, over SSH):
```bash
ssh spark-147c "cd ~/projects/scalarlm && git diff -- ml/cray_megatron/models/load_model.py infra/cray_infra/one_server/main.py infra/cray_infra/training/register_megatron_models.py ml/adapters/create_lora_model.py ml/cray_megatron/megatron/training_loop.py" > /tmp/spark_wip_recheck.diff
diff /tmp/spark_wip_recheck.diff /tmp/claude-1000/-home-georgi-projects-scalarlm/ea9b3f5c-7f3c-496a-b972-19d377bff86e/scratchpad/spark_other.diff /tmp/claude-1000/-home-georgi-projects-scalarlm/ea9b3f5c-7f3c-496a-b972-19d377bff86e/scratchpad/spark_load_model.diff 2>&1 | head -5
```

Expected: the diffs are identical to what was already compared earlier in this session (byte-for-byte, since nobody has touched spark-147c since). If they differ, STOP — someone else changed spark-147c's working tree; investigate before discarding anything.

- [ ] **Step 5: Stash spark-147c's stale WIP (recoverable, path-scoped), then sync to the pushed branch**

Belt-and-suspenders: stash before reset so the confirmed-superseded diffs are still recoverable from `git stash list` if this judgment call ever turns out wrong, even though Step 4 confirmed they're byte-identical duplicates of already-committed work.

**Do NOT use a blanket `git stash -u`.** `.gitignore` only excludes `vllm/***` — `jobs/` (4.9G of real training-job data) and `test/finetune_sweep/results/` (238M of prior sweep results) are untracked-but-not-ignored on spark-147c, so `stash -u` would sweep them into the stash. Scope the stash to the exact code files instead:

Verified at execution: spark actually has **8** tracked-modified files (docker-compose.yaml, finetune-sweep.yaml, run_finetune_sweep.py in addition to the 5 originally baselined) plus 2 untracked code files (resolve_target_modules.py, doc_mask.py) — all confirmed superseded by the pushed `650ead2` (the 3 extra: runner + compose byte-identical; yaml differs only in the corrected serve-premise comment). Stash all 10 so nothing is discarded un-recoverably:

```bash
ssh spark-147c "cd ~/projects/scalarlm && git stash push -u -m 'pre-candidate-sweep-sync: superseded WIP (all recoverable)' -- \
  docker-compose.yaml \
  infra/cray_infra/one_server/main.py \
  infra/cray_infra/training/register_megatron_models.py \
  ml/adapters/create_lora_model.py \
  ml/cray_megatron/megatron/training_loop.py \
  ml/cray_megatron/models/load_model.py \
  test/finetune_sweep/finetune-sweep.yaml \
  test/finetune_sweep/run_finetune_sweep.py \
  ml/adapters/resolve_target_modules.py \
  ml/cray_megatron/megatron/doc_mask.py \
  && git fetch origin && git reset --hard origin/georgi/finetune-sweep"
```

The pathspec covers all 8 tracked-modified files plus the 2 untracked code files (`-u` includes them). `jobs/` (4.9G), `test/finetune_sweep/results/` (238M), and `vllm/` are NOT in the pathspec, so both the scoped stash and the reset leave them untouched.

Expected output ends with `HEAD is now at <new-hash> chore(sweep): checkpoint pre-candidate-sweep WIP`, preceded by a `Saved working directory ...` stash line.

- [ ] **Step 6: Confirm spark-147c is clean and matches local**

Run: `ssh spark-147c "cd ~/projects/scalarlm && git status --short && git log --oneline -1"`

Expected: no modified-tracked-file lines (untracked dirs OK), and the log line matches local's `git log --oneline -1`.

---

### Task 2: Add the 6 Causal LM candidates to finetune-sweep.yaml

**Files:**
- Modify: `test/finetune_sweep/finetune-sweep.yaml` (append a new `models:` section)

**Interfaces:**
- Consumes: existing yaml schema — each entry is `id` (HF model id), optional `cpu_ok`, `multimodal`, `adapters.lora.gate_gb` (VRAM gate, GiB), `train_args` (overrides merged onto `train_args_defaults`).
- Produces: 6 new model ids the sweep's `--models` filter (Task 4) can select by exact HF id string.

- [ ] **Step 1: Append the Causal LM candidates block**

Insert after the existing `# === MoE candidates recorded for later converter-generalization testing` commented block (end of file), i.e. as a new top-level `models:` list continuation:

```yaml
  # === Model-category-report candidates (added 2026-07-02) — Causal LM ===
  # From docs/reports/2026-07-01-model-categories.md §Potential Sweep Candidates
  # / Causal LM. Serve outcome is governed by vLLM's native arch support + the
  # fork's hybrid-LoRA key normalization, NOT by the 3-class ScalarLM/Tokenformer
  # registry in infra/cray_infra/adapters/model/models.py (that registry gates
  # Tokenformer, a different adapter type). Evidence: phi-4 (Phi3), Qwen3-8B
  # (Qwen3), Mistral-7B (Mistral) and Qwen2-VL all full-PASS despite none being
  # in that 3-class list. So expectations here are set by ARCH PRECEDENT:
  #   Phi-4-mini  -> Phi3ForCausalLM  -> phi-4 PASSes  -> expect PASS
  #   Qwen3-*     -> Qwen3ForCausalLM -> Qwen3-8B PASSes -> expect PASS
  #   GLM-4-9B    -> Glm4ForCausalLM  -> NO precedent   -> serve is the real
  #     unknown: vLLM supports Glm4, but the fork's LoRA key normalization has
  #     never seen GLM-4 prefixes and could hit an ADAPTER_NO_OP like ADR 0006.
  - id: microsoft/Phi-4-mini-instruct          # Phi3ForCausalLM, ~3.8B; MIT, ungated
    cpu_ok: true
    adapters: {lora: {gate_gb: 10}}
    # Same arch family as phi-4 (14B) below, which full-PASSed but only after
    # bf16 (fp32 mode-collapsed at this LR) — apply bf16 defensively.
    train_args: {dtype: bfloat16, max_steps: 450, warmup_steps: 30, steps_per_checkpoint: 450}
  - id: Qwen/Qwen3-1.7B                        # Qwen3ForCausalLM; Apache-2.0, ungated
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}}
    train_args: {dtype: bfloat16}
  - id: Qwen/Qwen3-4B                          # Qwen3ForCausalLM; Apache-2.0, ungated
    cpu_ok: true
    adapters: {lora: {gate_gb: 10}}
    train_args: {dtype: bfloat16}
  - id: Qwen/Qwen3-14B                         # Qwen3ForCausalLM; Apache-2.0, ungated
    adapters: {lora: {gate_gb: 32}}
    train_args: {dtype: bfloat16, max_steps: 450, warmup_steps: 30, steps_per_checkpoint: 450}
  - id: Qwen/Qwen3-32B                         # Qwen3ForCausalLM; Apache-2.0, ungated
    # Borderline size (bf16 ~64GiB); relies on cuda-spark's phase_scaled flow
    # (whole-GPU-per-phase) rather than the co-located budget. RISK: unlike the
    # 30B-A3B MoE that PASSed (activates ~3B/forward), dense-32B activates ALL 32B
    # in the training forward/backward, so train-phase activation memory is much
    # higher — a TRAIN_FAILED/OOM here is expected-territory, not a harness bug
    # (bounded by --train-timeout; doesn't block other models). gate_gb below is a
    # no-op on spark (unified mem) — kept only for cuda-docker portability.
    adapters: {lora: {gate_gb: 70}}
    train_args: {dtype: bfloat16, max_steps: 450, warmup_steps: 30, steps_per_checkpoint: 450}
  - id: zai-org/GLM-4-9B-Chat                  # Glm4ForCausalLM; MIT-family, ungated
    # The one genuinely-novel serve case in this block: no prior GLM-4 run.
    # Training should work (arch-agnostic); serve is the open question (see the
    # ADR-0006-style key-normalization risk in the block header). Whatever it
    # does — PASS, or a GLM-4-specific ADAPTER_NO_OP — is the useful data point.
    adapters: {lora: {gate_gb: 20}}
    train_args: {dtype: bfloat16, max_steps: 450, warmup_steps: 30, steps_per_checkpoint: 450}
```

- [ ] **Step 2: Validate the yaml parses and the 6 new ids are present**

Run:
```bash
python3 -c "
import yaml
d = yaml.safe_load(open('test/finetune_sweep/finetune-sweep.yaml'))
ids = [m['id'] for m in d['models']]
want = ['microsoft/Phi-4-mini-instruct','Qwen/Qwen3-1.7B','Qwen/Qwen3-4B','Qwen/Qwen3-14B','Qwen/Qwen3-32B','zai-org/GLM-4-9B-Chat']
missing = [w for w in want if w not in ids]
assert not missing, f'missing: {missing}'
assert len(ids) == len(set(ids)), 'duplicate model id in yaml'
print('OK', len(ids), 'total models')
"
```

Expected: prints `OK <N> total models` with no assertion error.

- [ ] **Step 3: Commit**

```bash
git add test/finetune_sweep/finetune-sweep.yaml
git commit -m "$(cat <<'EOF'
feat(sweep): add 6 Causal LM candidates from the model-categories report

Phi-4-mini-instruct, Qwen3-{1.7B,4B,14B,32B}, GLM-4-9B-Chat. Phi-4-mini
(Phi3) and the Qwen3 family have full-PASS arch precedent (phi-4, Qwen3-8B)
so expect PASS; GLM-4-9B (Glm4) is the novel serve case -- training should
work, serve depends on the fork's LoRA key normalization handling GLM-4
prefixes (untested; ADR-0006-style no-op risk).
EOF
)"
```

---

### Task 3: Add the 4 Multimodal candidates to finetune-sweep.yaml

**Files:**
- Modify: `test/finetune_sweep/finetune-sweep.yaml` (append after Task 2's block)

**Interfaces:**
- Consumes: same schema as Task 2. NOTE: the `multimodal: true` yaml field is **decorative** — it is referenced nowhere in `run_finetune_sweep.py`/`preflight.py`; the real multimodal routing to `AutoModelForImageTextToText` is `is_multimodal(model_config)` in `load_model.py`, which reads the HF config's `vision_config` (the working `Qwen2-VL-7B-Instruct` entry PASSes *without* the flag). Kept here only as a human-readable category tag; if a VLM's config doesn't expose a top-level `vision_config`, it routes to `AutoModelForCausalLM` and may TRAIN_FAILED regardless of the flag (the InternVL3 risk).
- Produces: 4 new model ids for the `--models` filter.

- [ ] **Step 1: Append the Multimodal candidates block**

```yaml
  # === Model-category-report candidates (added 2026-07-02) — Multimodal ===
  # From docs/reports/2026-07-01-model-categories.md §Potential Sweep Candidates
  # / Multimodal. Existing Qwen2-VL-7B-Instruct entry (above) is the only
  # validated multimodal PASS-track model; these diversify vendor/arch per the
  # report. VLM budget-starvation precedent (Qwen2-VL needed 900 steps, not
  # the 300/450 default) applied to all four defensively; dtype left at the
  # global fp32 default (Qwen2-VL's curve descended cleanly in fp32 -- no
  # collapse -- unlike the causal-LM 7B+ mode-collapse cases that needed bf16).
  - id: Qwen/Qwen2.5-VL-7B-Instruct            # Qwen2_5_VLForConditionalGeneration; Apache-2.0
    # Same family as the already-working Qwen2-VL-7B-Instruct -- best odds of
    # a clean drop-in (train AND serve) of the four.
    multimodal: true
    cpu_ok: true
    adapters: {lora: {gate_gb: 18}}
    train_args: {max_steps: 900, warmup_steps: 50, steps_per_checkpoint: 900}
  - id: mistral-community/pixtral-12b          # LlavaForConditionalGeneration (HF-native
    # conversion; the official mistralai/Pixtral-12B-2409 ships Mistral's
    # consolidated format that HF AutoModel can't parse -> use this instead).
    # Apache-2.0.
    multimodal: true
    adapters: {lora: {gate_gb: 30}}
    train_args: {max_steps: 900, warmup_steps: 50, steps_per_checkpoint: 900}
  - id: OpenGVLab/InternVL3-8B                 # InternVLForConditionalGeneration (unverified against
    # installed transformers version -- report flags this explicitly). May not
    # expose the vision_config attribute is_multimodal() keys off; if so this
    # will TRAIN_FAILED at model-class dispatch rather than an LoRA/serve issue
    # -- diagnose before assuming a harness bug.
    multimodal: true
    adapters: {lora: {gate_gb: 20}}
    train_args: {max_steps: 900, warmup_steps: 50, steps_per_checkpoint: 900}
  - id: allenai/Molmo-7B-D-0924                # Molmo uses custom (trust_remote_code) modeling
    # code on HF; load_model.py's from_pretrained calls do NOT pass
    # trust_remote_code=True anywhere in this codebase today. Expect a hard
    # TRAIN_FAILED at load ("trust_remote_code" / custom-code-not-allowed
    # error) unless HF has since folded Molmo into core transformers -- this
    # is a known, pre-flagged risk, not a surprise if it fails exactly there.
    multimodal: true
    adapters: {lora: {gate_gb: 20}}
    train_args: {max_steps: 900, warmup_steps: 50, steps_per_checkpoint: 900}
```

- [ ] **Step 2: Validate yaml + full 10-model id set**

```bash
python3 -c "
import yaml
d = yaml.safe_load(open('test/finetune_sweep/finetune-sweep.yaml'))
ids = [m['id'] for m in d['models']]
want = ['microsoft/Phi-4-mini-instruct','Qwen/Qwen3-1.7B','Qwen/Qwen3-4B','Qwen/Qwen3-14B','Qwen/Qwen3-32B','zai-org/GLM-4-9B-Chat',
        'Qwen/Qwen2.5-VL-7B-Instruct','mistral-community/pixtral-12b','OpenGVLab/InternVL3-8B','allenai/Molmo-7B-D-0924']
missing = [w for w in want if w not in ids]
assert not missing, f'missing: {missing}'
assert len(ids) == len(set(ids)), 'duplicate model id in yaml'
print('OK', len(ids), 'total models,', len(want), 'candidates present')
"
```

Expected: `OK <N> total models, 10 candidates present`.

- [ ] **Step 3: Commit**

```bash
git add test/finetune_sweep/finetune-sweep.yaml
git commit -m "$(cat <<'EOF'
feat(sweep): add 4 Multimodal candidates from the model-categories report

Qwen2.5-VL-7B-Instruct, pixtral-12b (HF-native), InternVL3-8B, Molmo-7B-D-0924.
InternVL3 and Molmo are flagged as likely TRAIN_FAILED at model load
(unverified vision_config / trust_remote_code respectively) -- included
so the sweep records exactly where they break.
EOF
)"
```

---

### Task 4: Sync to spark-147c and launch the filtered sweep

**Files:** none (operational).

**Interfaces:**
- Consumes: `test/finetune_sweep/run_finetune_sweep.py --target cuda-spark --models <ids>`. `--models` is a membership filter; the run order is the yaml order (`filter_models` + `split_by_preflight` both preserve it), so to get smoke→novel→rest we split the launch into three sequenced `--models` invocations chained with `&&` rather than reordering the yaml (which stays category-grouped for readability).
- Produces: a background process on `spark-147c` running the three-batch chain, writing one timestamped `test/finetune_sweep/results/*.json` per batch and a single nohup log; a PID to monitor in Task 5.

- [ ] **Step 1: Push and pull**

```bash
git push origin georgi/finetune-sweep
ssh spark-147c "cd ~/projects/scalarlm && git pull --ff-only origin georgi/finetune-sweep"
```

Expected: fast-forward pull succeeds (Task 1 already made spark-147c's tree clean and aligned).

- [ ] **Step 2: Launch the three-batch chain (smoke → novel → rest), backgrounded**

Run order is smoke → novel → rest. CRITICAL exit-code semantics (`main()` returns `1` if ANY model's outcome ∉ `{PASS, SKIPPED, NO_MEMORIZATION}`): the runner still runs *all* models in one `--models` invocation and records each — only the *final* exit code reflects a hard fail. Molmo is *expected* to `TRAIN_FAILED` (hard fail → exit 1), so the novel batch must NOT gate the rest batch. Therefore: NO `set -e`; only the smoke line gates (via `|| exit 1`); novel and rest are unconditional sequential lines.

```bash
ssh spark-147c "cd ~/projects/scalarlm && nohup bash -c '
  R=test/finetune_sweep/run_finetune_sweep.py
  # 1. SMOKE GATE: one fast PASS-precedent model. A hard fail here
  #    (RESTART_FAILED/TRAIN_FAILED, exit 1) means the spark stack or git sync is
  #    broken -> abort before wasting hours. PASS/NO_MEMORIZATION are exit 0 -> proceed.
  python3 \$R --target cuda-spark --models Qwen/Qwen3-1.7B || exit 1
  # 2. NOVEL-ARCH (real signal): runs all 4 and records each; its exit-1 (Molmo
  #    TRAIN_FAILED expected) does NOT stop the next line — no set -e, no && here.
  python3 \$R --target cuda-spark --models zai-org/GLM-4-9B-Chat mistral-community/pixtral-12b OpenGVLab/InternVL3-8B allenai/Molmo-7B-D-0924
  # 3. REST: slower PASS-precedent confirmations. Runs regardless of batch 2 outcome.
  python3 \$R --target cuda-spark --models microsoft/Phi-4-mini-instruct Qwen/Qwen3-4B Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen2.5-VL-7B-Instruct
' > /home/georgi/sweep-candidates-20260702.log 2>&1 < /dev/null &
  disown
  echo LAUNCHED PID=\$!"
```

Expected: prints `LAUNCHED PID=<n>`. Note the PID for Task 5. If the smoke model hard-fails, the log ends after batch 1 with no batch-2/3 output — inspect the spark stack before re-launching, don't blind-retry.

- [ ] **Step 3: Confirm it actually started (not an immediate crash)**

Run (after ~30s): `ssh spark-147c "tail -n 40 /home/georgi/sweep-candidates-20260702.log"`

Expected: log shows the harness starting the first model (preflight check / restart command), not a Python traceback. If it crashes immediately, diagnose via `superpowers:systematic-debugging` before re-launching — do not just retry blind.

---

### Task 5: Monitor to completion and write the results report

**Files:**
- Create: `docs/reports/2026-07-02-model-category-candidates-sweep-results.md`

**Interfaces:**
- Consumes: `test/finetune_sweep/results/*.json` (or whatever the harness's per-run result format is — check `run_finetune_sweep.py`'s result-writing code if unfamiliar) plus the nohup log from Task 4.
- Produces: a report with one row/section per candidate: PASS / TRAIN_FAILED / RESTART_FAILED / ADAPTER_NO_OP / NO_MEMORIZATION / etc., with the concrete failure detail string, following the existing report style (see `docs/reports/2026-06-30-moe-expert-lora-serving.md` for precedent).

- [ ] **Step 1: Poll until the background run finishes**

Each of the 10 models can take anywhere from ~5 min (small dense) to 45+ min (32B phase-scaled, or a fresh multi-GB HF download). Poll every 15-20 minutes rather than tightly:

Run: `ssh spark-147c "tail -n 20 /home/georgi/sweep-candidates-20260702.log; ps -p <PID> || echo DONE"`

Expected: eventually `ps -p <PID>` reports the process gone (`DONE`). The log contains three per-batch summary blocks (smoke=1 model, novel=4, rest=5); a `DONE` with only the smoke block present means the chain halted on a smoke failure — inspect that before doing anything else.

- [ ] **Step 2: Pull the results artifacts (three files — one per batch)**

The three invocations each write their own timestamped `finetune.cuda-spark.YYYYMMDD-HHMMSS.json`. Pull all of today's:

```bash
mkdir -p /tmp/candidate_sweep_results
scp "spark-147c:~/projects/scalarlm/test/finetune_sweep/results/finetune.cuda-spark.20260702-*.json" /tmp/candidate_sweep_results/ 2>&1 || \
ssh spark-147c "ls -la ~/projects/scalarlm/test/finetune_sweep/results/ | tail -20"
```

If the date prefix differs (run crossed midnight, or the harness names files differently), `ls` the results dir and pick the three files newer than the launch time. Merge them when building the report table in Step 3 — outcomes across all three batches cover the full 10.

- [ ] **Step 3: Write the report**

Write `docs/reports/2026-07-02-model-category-candidates-sweep-results.md` with: a summary table (model | category | outcome | detail | timing), and per-model notes focused on the four novel-arch cases (GLM-4-9B, Pixtral, InternVL3, Molmo) — whether GLM-4 served or hit a key-normalization no-op, whether Pixtral/InternVL3/Molmo loaded at all — since the six PASS-precedent models are confirmation, not new signal.

- [ ] **Step 4: Update the source report with a pointer**

Add a one-line note at the bottom of `docs/reports/2026-07-01-model-categories.md`'s "Potential Sweep Candidates" section pointing to the new results report, so a future reader doesn't re-propose the same candidates as "not yet tried."

- [ ] **Step 5: Commit**

```bash
git add docs/reports/2026-07-02-model-category-candidates-sweep-results.md docs/reports/2026-07-01-model-categories.md
git commit -m "$(cat <<'EOF'
docs(sweep): results for the 10 model-category-report candidates

EOF
)"
```
