# Custom-code VLMs vs transformers 5.x: layer analysis + two infra fixes

**Date:** 2026-07-03
**Branch:** `georgi/finetune-sweep`
**Scope:** finetune sweep on spark-147c (DGX Spark, GB10, transformers 5.12.1)
**Verdict:** GLM-4-9B-Chat, InternVL3-8B, Molmo-7B-D-0924 accepted as **blocked** on
vendor-code-vs-transformers-5.x drift. Two general infra bugs found and fixed along
the way (both affected *all* models, not just the VLMs).

---

## 1. Two infra fixes (high value — fixed real bugs for every model)

### 1a. `ml/` snapshot freeze — sweep re-runs executed stale code (commit `8cc8ceb`)

`upload_training_data.py` copies `./ml` into `jobs/<hash>/ml` **only** `if not
os.path.exists(ml_directory)`, and the job hash is a deterministic sha256 of
`train_args`. So re-running a model with an unchanged config reused the frozen
snapshot and ran STALE code — deployed source fixes never executed.

This masked the GLM-4 `seq_length` fix for a full day: every re-run failed with a
byte-identical `Cannot find max_position_embeddings in ChatGLMConfig` from a Jul-2
snapshot. Tell: `jobs/<hash>/ml/.../load_language_model_dataset.py` had a stale
mtime and `grep -c seq_length` == 0.

**Fix:** `refresh_model_job_dirs()` deletes every prior job dir for the model
(matched by `llm_name`, via in-container `docker compose exec`) through
`POST /v1/megatron/delete/{hash}` right before submit, forcing the copytree to
re-run against current bind-mounted source. Per-model; compose-only (k8s snapshots
per-namespace). +7 unit tests. **Validated:** after the fix the refresh fired
(GLM `ml/` mtime → today, `seq_length` present) and GLM cleared the
`max_position_embeddings` blocker.

### 1b. transformers/safetensors skew crash-looped the api server (commit `0b5a4e7`)

The megatron layer installs `requirements-megatron.txt` with `--no-deps` (so it
can't clobber the vLLM torch/safetensors pins). vLLM pins `transformers <5`;
megatron forces `transformers>=5.5.0`, so the resolved transformers was
cache-**frozen** at 5.12.1 (last line compatible with the shipped safetensors
0.7.0).

Adding `timm` to `requirements-megatron.txt` busted the Docker layer cache → the
rebuild re-pulled `transformers>=5.5.0` = **5.13.0**, which hard-requires
safetensors>=0.8.0. `--no-deps` left safetensors at 0.7.0, so `import transformers`
raised and the **api server crash-looped (48 restarts)** — blocking every model.
Tell: container "Up 1 second" on every check, `docker inspect -f
{{.RestartCount}}` climbing, `ImportError: safetensors>=0.8.0 is required ... found
0.7.0` in `docker logs`.

**Fix:** cap `transformers>=5.5.0,<5.13`, making the working version explicit
instead of cache-dependent. Rebuilt (only tail layers — vLLM cached). **Validated:**
transformers 5.12.1, `import transformers`/`import timm` OK, api server healthy
(`{"api":"up"}`), no crash-loop.

**Gotcha recorded:** the live compose container (`docker exec`) is the version
source of truth. A stale `scalarlm-cray-spark:latest` tag repeatedly lied (wrong
safetensors, missing torchvision). timm 1.0.27 and torchvision 0.25.0a0 ship in the
NGC base — InternVL's "requires timm" needs **no** torchvision rebuild.

---

## 2. The three custom-code VLMs — each advances one layer per shim

Every targeted fix worked and moved the model to its next distinct
transformers-4.x-vs-5.x incompatibility. The models are all loaded via
`trust_remote_code`; their vendored modeling/config code predates transformers 5.x.

### GLM-4-9B-Chat (zai-org) — fused ChatGLM arch, 5 layers deep
1. custom code → `trust_remote_code` (JobConfig field)
2. `ChatGLMConfig.max_length` missing → backfill from `seq_length`
3. `all_tied_weights_keys` missing → class-level default shim
4. `max_position_embeddings` missing → `get_max_position_embeddings` reads `seq_length`
5. **`ChatGLMConfig has no attribute 'use_cache'`** — `modeling_chatglm.py:1002`
   reads `self.config.use_cache` in the forward; transformers 5.x dropped the base
   default. (NOT fixed.)

**Clean path (not pursued):** transformers 5.x **natively** supports newer GLM
archs (`Glm4vConfig`, `Glm46VConfig`, `GlmOcrConfig`) with no custom code. Swapping
to a natively-supported GLM is the right move rather than forking ChatGLM's 4.x code.

### InternVL3-8B (OpenGVLab) — 4 layers deep
1. custom code → `trust_remote_code`
2. `AutoModelForImageTextToText` rejects `InternVLChatConfig` → honor `auto_map`
   (use the causal class the repo declares)
3. `requires ... timm` → timm present in NGC base (import works once the box is healthy)
4. **`InternVLChatModel has no attribute 'generation_config'`** — model loads;
   `add_eos_token` in the dataset path reads `model.generation_config`. (NOT fixed.)

### Molmo-7B-D-0924 (allenai) — 4 layers deep, deepest failure
1. custom code → `trust_remote_code`
2. `all_tied_weights_keys` missing → class-level default shim
3. `tie_weights() got unexpected kwarg 'missing_keys'` → finalizer compat shim
   (wrap `_finalize_model_loading`, drop 5.x-only kwargs for old-signature overrides)
4. **tensor broadcast mismatch** `[1,28,480,480]` vs `[1,1,28,480,480]` — model
   loads AND runs; fails inside its vision-language **forward compute**. This is a
   real modeling bug under this transformers/torch, not a one-line attribute shim.
   (NOT fixed.)

---

## 3. Decision

Per systematic-debugging Phase 4.5 (3+ fixes, each revealing a new problem
elsewhere = wrong approach, not a failed hypothesis): **stop the per-model vendor
shimming.** Making these three train would mean maintaining forks of each vendor's
modeling code — Molmo especially, whose failure is in forward compute. The general
infra fixes (1a, 1b) and the compat shims already landed are kept; they fixed real,
broadly-applicable bugs.

**Box left clean:** sweep exited, no container running, zero non-terminal jobs.
