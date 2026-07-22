# Handoff — ScalarLM fine-tune sweep on the DGX Spark

Date: 2026-06-19 · Branch: `georgi/finetune-sweep` (working tree clean) ·
Repo: `/home/georgi/projects/scalarlm`

## Where to start next work day

The `cuda-spark` fine-tune sweep is **working end-to-end** and a full 22-model run
is done. The serving/infra problems are solved; what remains is a short, concrete
list of **model-coverage fixes**. Pick up from the "Open items" in the findings
report (don't re-derive — it's all written down):

- **Findings report (READ THIS FIRST):**
  `docs/reports/2026-06-19-finetune-sweep-cuda-spark-findings.md` (commit `cc04565`).
  Full 22-model scoreboard, serving-config rationale, runner fixes, and the
  root-caused failure taxonomy.

### Immediate next actions (in priority order)
1. **Llama 3.2 quick win.** The user's HF token (`gessha`, on the Spark in
   `~/.hf_env`) now has access to `meta-llama/Llama-3.2-1B-Instruct` and
   `-3.2-3B-Instruct` (`Llama-3.1-8B` is still 403 — needs its own license
   accepted). A targeted run of the two 3.2 models should PASS (dense Llama arch).
   Run command pattern is in the findings report / below.
2. **MoE `target_modules` bug** (`qwen3-moe`): PEFT got `target_modules` as the
   char-set of `"all-linear"` → "not found". Fix in the training LoRA path
   (`ml/adapters/merge_lora_and_push.py:158` builds the config; `ml/adapters/create_lora_model.py:22`
   calls `get_peft_model`). Pass explicit module names or pin/patch PEFT.
3. **Multimodal training unsupported** (`Qwen2-VL-7B`, `gemma-3-4b-it`):
   `ml/cray_megatron/models/load_model.py` uses `AutoModelForCausalLM`, which
   rejects multimodal configs (qwen2-vl fails at load; gemma-3 at forward,
   `IndexError` dim 3). Either add multimodal support (right AutoModel class +
   language-tower-only LoRA) or gate multimodal models out of the sweep.
4. **qwen2-vl serve crash** (tiny-random only): fork bug in
   `vllm/vllm/model_executor/models/qwen2_vl.py` — tied `lm_head` not handled when
   `tie_word_embeddings` is absent (defaults True). Fix sketch in the findings
   report §"Failure taxonomy 1". Real Qwen2-VL-7B serves fine, so low priority.
5. **NO_MEMORIZATION** (7 real instruct models): adapters serve but don't exactly
   memorize in 60 steps (several got very close). Bump `max_steps`/LR — it's a
   training-budget gap, not arch/serving.

## Operational facts the next agent needs

- **Spark access is DIRECT:** `ssh georgi@spark-147c`. (The 3090 / blackwell-maxq-0
  remotes are NOT — see memory `nvidia-target-remote-host`.)
- **SSH to the Spark drops on long-idle / heavy commands** (exit 255). Use
  `-o ServerAliveInterval=20`. `docker exec` into a wedged container can hang the
  session — prefer reading host-mounted files or a throwaway `docker run --rm -v ...`.
- **Run the sweep** on the Spark (host-level, not in-container). The runner reads
  the manifest+code at start, so a `git pull` mid-run doesn't affect the live run.
  Launch pattern (token + generous timeouts), detached:
  ```bash
  ssh georgi@spark-147c 'bash -lc "set -a; source ~/.hf_env; set +a; cd ~/projects/scalarlm; \
    nohup python3 test/finetune_sweep/run_finetune_sweep.py --target cuda-spark \
    [--no-preflight] [--models <id> <id>] --restart-timeout 5400 --train-timeout 2400 \
    > /tmp/sweep_X.log 2>&1 &"'
  ```
- **HF_TOKEN** is persisted at `~/.hf_env` on the Spark (`export HF_TOKEN=...`,
  mode 600). It reaches the container via the compose env passthrough. Do NOT print
  its value. Token works for Mistral + Gemma; Llama 3.2 yes, Llama 3.1-8B no.
- **Training job logs now persist** to host `~/projects/scalarlm/jobs/<hash>/slurm-1.out`
  (bind mount added in `9e522b7`) — read via `docker run --rm -v $PWD/jobs:/j alpine ...`
  if host perms block direct read. This is how the TRAIN_FAILED tracebacks were captured.
- **Repo HTTPS clone auth works on the Spark; SSH (git@github.com) does not.**
  `./vllm` is the live-mounted fork (branch `georgi/finetune-sweep`, `ed66668c3`).
- **Cleanup note:** a Spark container may be left wedged (`docker exec` hangs) after
  a 32B crash-loop; the next run's `--force-recreate` replaces it.

## Key results / state (don't re-discover)

- Last full run report: `test/finetune_sweep/results/finetune.cuda-spark.20260619-173911.md`
  (7 PASS, 7 NO_MEMORIZATION, 3 gated-SKIP Llama, 2 RESTART_FAILED [tiny-qwen2-vl,
  32B], 3 TRAIN_FAILED [now root-caused]).
- **LoRA serving is broad** — empirically Qwen2/Qwen3/Llama/Gemma3/Mistral/Phi3 all
  serve (Qwen3-8B PASSed). The 3-class `scalarlm_*` registry
  (`infra/cray_infra/adapters/model/models.py`) is **tokenformer-only**, not a LoRA
  limit. This corrects `docs/self-served-llms-scratchpad.md` (untracked).
- **32B dropped** from the manifest — can't fit co-located serving on 128GiB
  (needs phase-scaling). Manifest now 21 models.
- The session's runner fixes are all committed (`88e05e5`..`cc04565`) and validated.

## Suggested skills for the next session

- **superpowers:systematic-debugging** — for items 2-4 (training-path bugs); the
  failure modes are already isolated, so go straight to minimal repro → fix.
- **superpowers:test-driven-development** — the sweep runner has good unit coverage
  (`test/unit/test_finetune_sweep_*.py`, run via
  `PYTHONPATH=infra uv run --with pytest --with torch --with pyyaml python -m pytest`);
  add tests for any `target_modules` / loader changes.
- **superpowers:verification-before-completion** — verify fixes with an actual
  targeted Spark run, not just unit tests.

## Memories worth consulting
`spark-baseline-generate-timeout` (RESOLVED), `nvidia-target-remote-host`
(spark-147c direct SSH + blockers), `lora-serving-noop-causal-lm`,
`vllm-fork-hybrid-adapter-layer`, `cuda-docker-compose-env-passthrough`.
