# Handoff — fine-tune sweep (2026-06-23)

Continues `docs/handoffs/handoff-scalarlm-finetune-sweep-spark.md` (2026-06-19).
Read that first for environment/access basics; this doc only covers what changed
since and what's open.

## State in one line

The cuda-spark LoRA fine-tune sweep is at **18/21 models PASS**; the 3 remaining
are all root-caused (not mysteries). Branch `georgi/finetune-sweep` is pushed to
origin at **`833a662`** (includes a merge of `origin/main`).

## Don't re-derive — read these first

- **Per-model verdicts, the four+ bugs, and the loss-curve diagnostic**:
  `docs/reports/2026-06-22-finetune-sweep-session-summary.md` (authoritative; has
  the full status table + legend + open items).
- **Commits since `cc04565`** (the sweep work + the main merge): `git log --oneline cc04565..833a662`.
- **Memory** (`~/.claude/projects/-home-georgi-projects-scalarlm/memory/`): start with
  `sweep-nomem-loss-curve-diagnostic.md`, `sweep-serve-loads-all-adapters-cross-arch.md`,
  `qwen3-moe-lora-dropout-paramwrapper.md`, `vllm-fork-hybrid-adapter-layer.md`,
  `nvidia-target-remote-host.md`.
- **vLLM fork** is a *separate git repo* under `vllm/` (branch `georgi/finetune-sweep`,
  tip `41db36934`). Its commits are NOT in the scalarlm push and were not merged with
  upstream this session.

## What changed this session (high level)

1. Root-caused + GPU-validated fixes for the NO_MEM / crash failures (memorization
   budget, cross-arch adapter contamination, MoE attention-only + rank inference,
   multimodal loader/4D-mask). Details + commit hashes in the session-summary report.
2. Per-model `train_args` overrides in `test/finetune_sweep/finetune-sweep.yaml`,
   all validated: **bf16** for every ≥7B model that mode-collapsed in fp32
   (Qwen2.5-14B, Mistral, phi-4, Qwen3-8B, rnj-1); **900 steps** for Qwen2-VL
   (budget-starved, not collapsed).
3. Merged `origin/main` (`833a662`). One conflict, in
   `ml/cray_megatron/megatron/training_loop.py` — integrated main's flex_attention
   `BlockMask` path with this branch's `doc_mask_decision()` multimodal-skip
   (SKIP_MULTIMODAL wins; see the merged comments in that file for the precedence
   rationale). `test_doc_mask_decision.py` + `test_resolve_target_modules.py` pass.

## Open items (priority order)

1. **MoE expert-LoRA serving** — the one genuinely deep item. `qwen3-moe` serves
   attention-only but can't memorize; needs the fork loader to replicate vLLM's
   PEFT→fused-MoE conversion (`FusedMoEWithLoRA.set_lora` wants a per-expert tensor
   list, the `.pt` exports stacked 2-D). See `qwen3-moe-lora-dropout-paramwrapper.md`
   and the report's open-items #1.
2. **gemma-4-dense NO_MEM** — capacity is ruled out (tiny-random-llama memorizes),
   so it's the `Gemma4ForConditionalGeneration` adapter key mapping. Lead doc:
   `docs/reports/2026-06-18-gemma4-dense-adapter-noop-diagnostic.md`.
3. **masint/tiny-random-qwen2-vl RESTART_FAILED** — vLLM base-model load crash-loop
   (pre-training, synthetic fixture; the real Qwen2-VL-7B passes). Low priority;
   likely droppable from the suite. To diagnose, re-run it solo with container-log
   capture (disrupts the live serve).
4. **Not pushed elsewhere**: the vLLM fork commits are local-only on the Spark's and
   this machine's fork checkouts; decide if/when to PR the fork. The fork has NOT
   been merged with upstream vLLM main (a much larger op — was explicitly deferred).
5. Doc nit: `adapter_format.py` `normalize_lora_key` docstring is stale (describes
   old "leave `model.layers.` as-is" behavior).

## Operational gotchas (still true)

- Runner executes **on the Spark** (`spark-147c`) and reads the **Spark's** manifest
  copy. Editing the manifest locally requires `scp` to the Spark before a run.
  `ml/`, `infra/cray_infra`, `vllm/vllm/tokenformer`, `test/` are live-mounted into
  the container; training code (`ml/`) re-imports per fresh slurm job, but the
  long-lived vLLM engine needs `docker restart scalarlm-cray-spark-1` to re-import.
- Sweep runs were driven by small local orchestrator scripts in `/tmp/*_orch.sh`
  that SSH-launch the runner and poll a PID, re-invoking on exit. Reusable pattern.
- Results-table trailing numbers are **timing** (restart_s/train_s/serve_s), NOT
  loss — read the real loss curve from a job's `slurm-1.out`.
- `HF_TOKEN` lives at `~/.hf_env` on the Spark — do not print it. SSH long-idle
  drops: use `-o ServerAliveInterval=20`. `jobs/` is root-owned; auto-mode blocks
  unauthorized destructive cleanup (scancel/status edits).

## Suggested skills

- `superpowers:systematic-debugging` — for items #1–#3 (especially the loss-curve /
  multi-component evidence-gathering discipline that cracked the NO_MEM cases).
- `superpowers:test-driven-development` — the resolve/doc_mask/lora_from_pt fixes all
  shipped with unit tests; keep that pattern for the MoE serving fix.
- `superpowers:verification-before-completion` — every fix here was GPU-validated on
  the Spark before being called done; hold that bar.
