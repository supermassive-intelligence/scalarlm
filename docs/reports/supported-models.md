# Supported models — LoRA fine-tune sweep

**Status summary as of 2026-07-08.** Consolidates verdicts that were scattered
across the sweep manifest and ~8 report/handoff docs into one list.

## What "supported" means here

A model is tested by the finetune sweep
(`test/finetune_sweep/finetune-sweep.yaml`): it LoRA-fine-tunes the model to
memorize a single input→output pair (a fixed hex golden string), then serves the
adapter and checks the served model reproduces that string. A full **PASS**
therefore means the model **trains, serves the `.pt` adapter, AND memorizes**
end-to-end on a real GPU target (predominantly `cuda-spark`, the DGX Spark GB10).

Verdict legend: **PASS** (green, all three) · **NO_MEMORIZATION** (trains +
serves, adapter applied but can't reproduce the string) · **TRAIN_FAILED** /
**RESTART_FAILED** (dies before the memorization check) · **BLOCKED** (a
characterized wall — VRAM or vendor-code-vs-transformers drift) · **STAGED**
(configured, arch-precedent says PASS, not yet run on GPU).

### Source keys

The **Source** column on each table below cites where that model's verdict comes
from, using these keys:

| Key | Document |
|---|---|
| **YAML** | `test/finetune_sweep/finetune-sweep.yaml` — inline per-model RESULT / tuning verdict |
| **SS22** | `docs/reports/2026-06-22-finetune-sweep-session-summary.md` — the historical PASS table |
| **MoE30** | `docs/reports/2026-06-30-moe-expert-lora-serving.md` — expert-LoRA converter |
| **P30B** | `docs/reports/2026-07-01-30b-moe-phased-serving-four-fixes.md` — 30B MoE PASS + Qwen1.5-MoE gap |
| **VLM3** | `docs/reports/2026-07-03-vlm-vendor-code-transformers5-blockers.md` — custom-code VLM blockers |
| **GRAN** | `docs/reports/2026-07-04-granite-hybrid-lora-pass.md` — granite hybrid PASS |
| **G4D** | `docs/reports/2026-06-18-gemma4-dense-adapter-noop-diagnostic.md` — gemma-4-dense no-op |
| **MEM** | memory index (`~/.claude/.../memory/MEMORY.md`) — e.g. archdiv sweep, VRAM ceiling notes |
| **RUN0706** | `test/finetune_sweep/results/finetune.cuda-spark.20260706-095423.md` — fix-confirmation run (Phi-4-mini, Qwen2.5-VL, pixtral) |
| **RUN0706b** | 2026-07-06 separate-expert MoE run (Phi-mini-MoE NO_MEM, OLMoE inconclusive); cancelled mid-serve, no results file — verdicts from log + job slurm diagnostics |
| **STAGED07** | 2026-07-07 staged-model run — the 6 formerly-staged Qwen3 models; `test/finetune_sweep/results/finetune.cuda-spark.20260707-113157.md` + lr-1e-3 fix-up `…-122953.md` |
| **VAST0708** | 2026-07-08 cuda-vast H200 bare-metal run — the VRAM-blocked tier validated end-to-end (Qwen3-32B, Qwen3.5-35B-A3B, Mixtral-8x7B-Instruct); runbook `docs/runbooks/finetune-sweep-vast-prebuilt-image-baremetal.md` |
| **VAST0708b** | 2026-07-08 cuda-vast H200 follow-up serial queue — dense `Qwen2.5-32B-Instruct` (PASS) + `Qwen2.5-72B-Instruct` (PASS, **TP=2 serving**); `google/gemma-3-27b-it` (PASS — the serve failure was a **stale-fork deployment gap**, already fixed in fork main). Same runbook |

---

## ✅ Fully supported (PASS end-to-end)

### Dense causal LM
| Model | Arch | Notes | Source |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | Qwen2 | CPU-runnable baseline | SS22 |
| `Qwen/Qwen2.5-1.5B-Instruct` | Qwen2 | last-4 Spark batch | SS22 |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2 | | SS22 |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2 | | SS22 |
| `Qwen/Qwen2.5-14B-Instruct` | Qwen2 | bf16, 450-step budget | SS22 |
| `Qwen/Qwen2.5-32B-Instruct` | Qwen2 (dense, 32B) | **PASS 2026-07-08 on cuda-vast H200** (loss 3.8e-8); **lr 1e-3**, bf16, 450-step, phased, single-GPU (~65 GiB). The real dense-32B Qwen (the README's "Qwen2-32B" doesn't exist — Qwen2 skips 32B) | VAST0708b |
| `Qwen/Qwen2.5-72B-Instruct` | Qwen2 (dense, 72B) | **PASS 2026-07-08 on cuda-vast H200** (loss 3.8e-8); **lr 5e-4**, bf16, 450-step. **Largest model validated end-to-end.** Trains single-GPU via `device_map={"":dev}` at ~142 GiB (right at the 143 GiB H200 ceiling; grad-checkpointed) but **serves only with `tensor_parallel_size: 2`** (135 GiB weights exceed one card at 0.85 util). lr 5e-4 holds through the warmup boundary; 1e-3 would overshoot (72B active) | VAST0708b |
| `Qwen/Qwen2-7B-Instruct` | Qwen2 | 450-step budget (slower-converging) | SS22 |
| `meta-llama/Llama-3.2-1B-Instruct` | Llama | gated (HF_TOKEN) | SS22 |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama | gated | SS22 |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama | gated | SS22 |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | **bf16** — fp32 mode-collapsed at lr 3e-3 | SS22, YAML |
| `microsoft/phi-4` | Phi3 (14B) | bf16, 450-step | SS22, YAML |
| `Qwen/Qwen3-8B` | Qwen3 | bf16 | SS22 |
| `Qwen/Qwen3-1.7B` | Qwen3 | PASS 2026-07-07 (train 306s); bf16, tolerates default lr 3e-3 | STAGED07 |
| `Qwen/Qwen3-4B` | Qwen3 | PASS 2026-07-07 (train 522s); **lr 1e-3** — default 3e-3 mode-collapsed (served `6666…`) | STAGED07 |
| `Qwen/Qwen3-14B` | Qwen3 | PASS 2026-07-07 (train 1662s); bf16, **lr 1e-3**, 450-step | STAGED07 |
| `Qwen/Qwen3-32B` | Qwen3 (dense, 32B) | **PASS 2026-07-08 on cuda-vast H200** (train 625s, loss 3.8e-8); **lr 1e-3**, bf16, 450-step, phased. The earlier GB10 TRAIN_TIMEOUT was unified-memory-marginal, not a hard block — a dedicated 140GiB H200 trains it clean | VAST0708 |
| `google/gemma-3-4b-it` | Gemma3 | gated; earlier "serve no-op" was cross-arch contamination, since fixed | SS22 |
| `google/gemma-3-27b-it` | Gemma3 (multimodal, 27B) | **PASS 2026-07-09 on cuda-vast H200** (loss 3.7e-8, exact hash); **lr 1e-3**, bf16, 450-step, single-GPU (~54 GiB). Serve failure was a **stale-fork deployment gap, already fixed in fork main** (`c13f401ed`, #31–#34): `hybrid_adapter_manager._detect_model_layers_prefix` on the running box (`a5c304b5b`, #29) picked the *vision tower's* `layers.` prefix on this multimodal base, rewriting every decoder LoRA key onto `vision_tower.*` → `add_lora` dropped the adapter → served hash 404'd. Main already collects all `layers.` candidates and prefers the `model.layers.`-ending (decoder) prefix, skipping vision — **no PR needed, bump the deployed image past `c13f401ed`**. Also exposed a serve-harness race (fixed in the test harness: readiness poll + monotonic `max_tokens` to dodge the work-queue request-dedup) | VAST0708b |
| `google/gemma-4-E2B-it` | Gemma4 (mm, ~2.3B eff / 5.1B) | **PASS 2026-07-21 on cuda-spark** (train 707s, loss 0.0000, exact golden hash); bf16, **lr 3e-3** (default tolerated — brief step-32 overshoot to 1.6 self-recovered to 0.0 by step 100), 300-step. **First real Gemma-4 validated end-to-end** — proves the `Gemma4ForConditionalGeneration` LoRA path (train→serve→memorize). Ungated. restart 53s / serve 8.7s | CAP0721 |
| `google/gemma-4-E4B-it` | Gemma4 (mm, ~4.5B eff / 8B) | **PASS 2026-07-21 on cuda-spark** (train 666s / 300 steps, exact golden hash ` aaaf6f8ae738dfc6577e63dda6daf9cc`); bf16, **lr 1e-3** — clean monotone descent (loss ~1e-5 by step 12, ~5e-7 asymptote, **no overshoot** unlike E2B's 3e-3). Confirms the Gemma-4 LoRA path scales to the 8B dense-mm tier. Ungated. serve 9.0s | CAP0721 |
| `google/gemma-4-12B-it` | Gemma4 (mm, 11.95B dense) | **PASS 2026-07-21 on cuda-spark** (train 2045s / 450 steps, exact golden hash ` aaaf6f8ae738dfc6577e63dda6daf9cc`); bf16, **lr 1e-3**, 30-step warmup — clean descent (loss ~5e-6 by step 47, no overshoot). Largest dense Gemma-4 validated on a single GB10 so far; weights 23 GiB, serve EngineCore ~103 GiB. Ungated. serve 10.4s | CAP0721 |
| `google/gemma-4-26B-A4B-it` | Gemma4 **MoE** (25.2B / 3.8B active, 8-of-128 exp) | **PASS 2026-07-22 on cuda-spark** (train 2694.6s / 150 steps, exact golden hash ` aaaf6f8ae738dfc6577e63dda6daf9cc` vs degenerate baseline "is is is"); bf16, **lr 1e-3**, **`lora_dropout: 0`** (PEFT ParamWrapper constraint on grouped `Gemma4TextExperts`), serve `--gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager`. **First Gemma-4 MoE validated end-to-end.** Serve required a fork fix: the expert-key re-normalization collapsed every layer onto `layers.moe.experts` because Gemma4's trained `.pt` has experts directly under the layer index (no container) while vLLM wraps them in a `Gemma4MoE` (`moe`) — fixed by insert-not-replace in `_renormalize_lora_sd_for_model` (`hybrid_adapter_manager.py`), see `2026-07-21-gemma4-moe-serve-lora-collision-fix.md`. serve 12.3s | CAP0721, COLL0722 |
| `google/gemma-4-31B-it` | Gemma4 (mm, 30.7B dense) | **PASS 2026-07-22 on cuda-spark** (train 1809.5s / 150 steps, exact golden hash ` aaaf6f8ae738dfc6577e63dda6daf9cc` vs degenerate baseline "is is is"); bf16, **lr 1e-3**, serve 14.3s / restart 55.8s. **Largest Gemma-4 (and largest dense) validated end-to-end on a single GB10** — the ~62 GiB bf16 activation was VRAM-marginal on the 128 GiB unified pool but held (no OOM/TIMEOUT; the memory note's cuda-vast fallback was not needed). Completes the Gemma-4 ladder (E2B→E4B→12B→26B-A4B MoE→31B dense all PASS). Ungated. | CAP0721 |
| `EssentialAI/rnj-1-instruct` | Gemma3 (8B) | bf16 | SS22 |
| `allenai/OLMo-2-1124-7B-Instruct` | Olmo2 | PASS 2026-07-04 (restart 68s / train 1035s / serve 10s) | YAML, MEM |
| `microsoft/Phi-4-mini-instruct` | Phi3 (~3.8B) | PASS 2026-07-06 (train 537s); **lr 1e-3** fix confirmed (default 3e-3 oscillated → NO_MEM) | RUN0706, YAML |

### Multimodal (VLM)
| Model | Arch | Notes | Source |
|---|---|---|---|
| `Qwen/Qwen2-VL-7B-Instruct` | Qwen2-VL | fp32, 900-step (budget-starved at 450); the first validated VLM PASS | SS22 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL | PASS 2026-07-06 (train 3647s); 900-step, `train_timeout: 4800` | RUN0706, YAML |
| `mistral-community/pixtral-12b` | Llava (12B) | PASS 2026-07-06 (train 2417s); **bf16 + lr 5e-4** fixes the end-of-warmup divergence | RUN0706, YAML |

### MoE (routed-expert LoRA)
| Model | Arch | Notes | Source |
|---|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | Qwen3MoE (30B/3B active) | PASS 2026-07-01; first real large-MoE, phase-scaled, expert-LoRA converter | P30B, YAML |
| `ibm-granite/granite-4.0-h-tiny` | GraniteMoeHybrid (~7B/1B) | PASS 2026-07-04; first hybrid Mamba-2 + MoE. Dense `shared_mlp` adapter carries memorization — no expert converter needed. Needed a 3-part fix (resolver + preflight + LR 1e-3/90-step) | GRAN, YAML |
| `Qwen/Qwen3.5-35B-A3B` | **Qwen3_5MoeForConditionalGeneration** (multimodal MoE, 35B/3B active) | **PASS 2026-07-08 on cuda-vast H200** (train ~1870s, loss 1.5e-8); bf16, **lr 1e-3**, `lora_dropout:0`, 450-step, phased. End-to-end proof of the resolver fix (`dd803fe`) reaching serving — grouped experts adapted via `target_parameters`, dense `shared_expert` + attention targeted, SSM/router/vision excluded. Default lr 3e-3 near-misses (warmup overshoot); **lr 1e-3** memorizes exactly. The GB10 was blocked at training model-load VRAM only | VAST0708 |
| `Qwen/Qwen3.6-35B-A3B` | Qwen3_5MoeForConditionalGeneration | Covered by Qwen3.5-35B-A3B (identical arch/size) — resolver-supported, same lr-1e-3 recipe. Not re-run | VAST0708 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Mixtral (47B/12.9B active) | **PASS 2026-07-08 on cuda-vast H200** (train 1370s, loss 3.9e-8); bf16, **lr 5e-4**, `lora_dropout:0`, 450-step, phased. **Loads as GROUPED experts in transformers 5.12** (`MixtralExperts` batched container) → the existing grouped-expert LoRA path trains + serves it with NO separate-expert converter and NO fork change. More lr-sensitive than the Qwens (12.9B active): lr 1e-3 overshoots hard after warmup (loss → 34); **lr 5e-4** holds the basin. HF-gated (needs token). GB10 VRAM block was hardware-specific | VAST0708 |

### Tiny-random fixtures (smoke tests)
| Model | Arch | Notes | Source |
|---|---|---|---|
| `masint/tiny-random-llama` | Llama | CPU | SS22 |
| `google/gemma-3-270m` | Gemma3 | base (no chat template) | SS22 |
| `google/gemma-3-270m-it` | Gemma3 | | SS22 |
| `yujiepan/qwen3-moe-tiny-random` | Qwen3MoE | validates the expert-LoRA converter (grouped 2D FusedMoE path) | MoE30 |

---

## 🔧 In progress (configured, run, not yet fully green)

| Model | Arch | State | Source |
|---|---|---|---|
| `microsoft/Phi-mini-MoE-instruct` | PhiMoE (7.6B/2.4B) | **NO_MEMORIZATION** (2026-07-06 run). Trains + serves cleanly (no `set_lora` crash), but loss floored at **~1.56** and **zero expert LoRA params** were saved → experts weren't adapted. Root cause: PhiMoE loads as **grouped** experts in transformers 5.x (no per-expert `nn.Linear`), so step A's separate-expert detection correctly didn't fire — and unlike Qwen3MoE, PEFT doesn't auto-adapt this grouped module. Attention-only can't memorize here | RUN0706b |
| `allenai/OLMoE-1B-7B-0924-Instruct` | Olmoe (7B/1B active, 64 exp) | **INCONCLUSIVE** (2026-07-06, run cancelled). **Memorized in training** (loss **1e-4**) but **attention-only** — zero expert LoRA params saved (grouped-in-5.x, same as PhiMoE). Serve-check then **deadlocked** in a generate poll loop (work queue "already processed→skipping" ×135, 0 tok/s, no crash/OOM); cancelled to free the GPU. Does not validate the converter (experts never adapted) | RUN0706b |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | Qwen2MoE (14B/2.7B) | Trains + serves + adapter applied, but served string is **scrambled** → NO_MEMORIZATION. Cause: its **shared expert** is mis-mapped by the converter (validated only on routed grouped experts). Converter gap, deferred | P30B |

> **Separate-expert converter (plan steps A+B) has no remaining test model on
> transformers 5.x.** Every "separate-expert" candidate actually loads as a **grouped**
> batched-parameter container in transformers 5.12 — PhiMoE, OLMoE, *and* Mixtral
> (`MixtralExperts`), none with per-expert `nn.Linear`. Mixtral — the canonical
> separate-expert model, formerly VRAM-blocked on Spark — was **validated end-to-end on
> cuda-vast H200 2026-07-08** and trained + served via the **grouped** path (experts
> adapted through `target_parameters` `gate_up_proj`/`down_proj`, exact memorization),
> with no separate-expert converter and no fork change. So step A's separate-expert
> detection has nothing to fire on under this transformers version. The grouped experts
> DO get LoRA once `LoraConfig.target_parameters` names the expert projections
> ; the earlier PhiMoE/OLMoE "zero
> expert LoRA" was that config gap, since resolved.
> See `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`.

### Staged — all run 2026-07-07 (see `finetune.cuda-spark.20260707-113157.md` +
### the lr-1e-3 fix-up `…-122953.md`)
The six formerly-staged models were run on cuda-spark. Outcomes:
- **Qwen3-1.7B / 4B / 14B → PASS** (now in *Fully supported › Dense causal LM*).
  4B and 14B needed **lr 1e-3**: at the default 3e-3 they mode-collapsed (loss reached
  the basin during warmup, then flatlined at ~2.70 at full lr → served degenerate
  `6666…`). 1.7B is small enough to tolerate 3e-3. Both yamls now pin lr 1e-3 for
  4B/14B/32B.
- **Qwen3-32B → TRAIN_TIMEOUT** (VRAM-marginal) and **Qwen3.5 / 3.6-35B-A3B →
  PRECHECK_NO_OP** (new multimodal-MoE arch) on the GB10. **All three since PASS on
  cuda-vast H200 (2026-07-08)** — now in *Fully supported* above. Both were
  hardware-specific GB10 blocks, not code gaps.

The `STAGED07` source key = this 2026-07-07 staged-model run (result files above).

---

## ❌ Unsupported / blocked

| Model | Arch | Blocker | Source |
|---|---|---|---|
| `tiny-random/gemma-4-dense` | Gemma4 (mm fixture) | NO_MEMORIZATION — **fixture capacity wall, NOT an adapter bug** (corrected 2026-07-21). The G4D adapter-key theory is fixed: the adapter trains, serves, and applies (served output shifts from baseline). It can't memorize because `hidden_size=8` + `vocab=262144` can't separate a 32-token target in 8 dims (frozen-head loss floors at ln(vocab)≈12.17; unfreezing the tied head still asymptotes far from 0). Contrast `tiny-random-llama` (hidden 256, vocab 32k) which PASSes. Serve-only smoke test; validate real Gemma-4 memorization on `gemma-4-E2B-it`. See `2026-07-21-gemma4-dense-memorize-capacity-wall.md` | G4D, **CAP0721** |
| `masint/tiny-random-qwen2-vl` | Qwen2-VL (mm fixture) | RESTART_FAILED — vLLM crash-loops loading the malformed synthetic base config. Low impact (real Qwen2-VL-7B PASSes) | SS22 |
| `zai-org/GLM-4-9B-Chat` | ChatGLM (custom code) | BLOCKED — vendor ChatGLM code vs transformers 5.x, 5 layers deep (ends at `ChatGLMConfig has no attribute 'use_cache'`). Clean path = swap to a natively-supported GLM arch | VLM3 |
| `OpenGVLab/InternVL3-8B` | InternVL (custom code, mm) | BLOCKED — vendor code vs transformers 5.x, 4 layers deep (`InternVLChatModel has no attribute 'generation_config'`) | VLM3 |
| `allenai/Molmo-7B-D-0924` | Molmo (custom code, mm) | BLOCKED — deepest: tensor-broadcast mismatch inside the vision-language **forward compute** (a real modeling bug under this transformers/torch, not a one-line shim) | VLM3 |

### Documented-but-not-tested (README aspirational)
The README "supported models" table lists production-aspirational models not wired
into any sweep or runtime config. Now tested on cuda-vast H200 (2026-07-08): the
README's **`Qwen/Qwen2-32B-Instruct` does not exist** (Qwen2 skips 32B) — the real
dense-32B is `Qwen/Qwen2.5-32B-Instruct`, which **PASSes** (now in *Fully supported*);
`google/gemma-3-27b-it` was tested → **PASS** (its serve-side 404 was a stale-fork
deployment gap, already fixed in fork main; now in *Fully supported*).
Still untested: `Qwen/Qwen3.5-122B-A10B`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`nvidia/Nemotron-3-Super-120B`. These 120B-class models (plus GLM-4.5-Air ~106B and
Llama-4-Scout 109B, ~210 GiB bf16) exceed a single 143 GiB H200 and need **multi-GPU
tensor-parallel for training** — the sweep's `device_map={"":dev}` load is single-GPU
only, so they're tooling-blocked, not just VRAM-blocked (a 72B is the practical single-GPU
train ceiling; serving a 72B already needs TP=2).

---

## Cross-cutting findings

- **Size ceiling on a single Spark:** fp32 co-located ~14B is comfortable, ~32B is
  borderline/OOM. Large models (30B+ MoE, dense 32B) run only via **phase-scaling**
  (train, tear down, then serve — peak GPU = 1 model). Models over the 128GiB
  unified pool even phase-scaled (Mixtral 47B, dense 32-35B, 70B+ dense) need
  `cuda-vast` (rented H200/B200) or k8s multi-GPU — a dedicated 140GiB H200 trains
  Mixtral-8x7B and the 32-35B tier clean (all PASS 2026-07-08).
- **Single-H200 train ceiling ≈ 72B dense; serving a 72B needs TP=2.** The trainer
  loads the base with `device_map={"":dev}` (one GPU, no model-parallel sharding), so a
  72B bf16 (~135 GiB) is the practical single-card training limit — it fits at ~142 GiB
  of 143 with grad-checkpointing (`Qwen2.5-72B-Instruct` PASS 2026-07-08). But 135 GiB
  of weights exceed one card at `gpu_memory_utilization 0.85`, so **serving the 72B
  requires `tensor_parallel_size: 2`** across both GPUs (validated). 120B+ models need
  multi-GPU TP for *training* too, which the current single-GPU load path can't do.
- **dtype:** every ≥7B model that mode-collapsed in fp32 at lr 3e-3 was fixed by
  **bf16** (Mistral, phi-4, Qwen3-8B, rnj-1, Qwen2.5-14B). Multimodal models
  descend cleanly in fp32 but are **budget-starved** — they need 900 steps, not
  the 300/450 default.
- **MoE memorization requires the experts.** Attention-only (or attention+dense)
  LoRA serves but can't memorize on pure-sparse MoEs — the expert-LoRA converter is
  the real unlock. Exception: hybrids with an always-on dense path (granite's
  `shared_mlp`) memorize through that without a converter.
- **Converter coverage:** grouped-expert MoE is validated across Qwen3MoE (2D gated
  FusedMoE) **and Mixtral** (`MixtralExperts`) — on transformers 5.12 the formerly
  "separate-expert" arches (Mixtral/PhiMoE/OLMoE) all load grouped, so their experts
  adapt via `LoraConfig.target_parameters` (no per-expert converter). **Shared-expert**
  (Qwen2MoE/deepseek) remains an open converter gap.

## 🎯 Architectural Gap Analysis (July 2026)

The following architectural directions are currently missing or under-represented in the sweep:

- **Linear-Complexity / Recurrents:** Missing block-transformer hybrids (e.g., **Jamba-2**) and pure RNN-style bases (**RWKV-7**), which are critical for verifying long-context LoRA and non-transformer support.
- **Advanced MoE Patterns:** While grouped/routed experts are PASS, **Shared-Expert MoE** (e.g., DeepSeek-V3) remains an open converter gap. **Multi-head Latent Attention (MLA)** also needs validation to verify KV-cache reduction efficiency under LoRA.
- **Native Multimodal:** Current VLMs are "Encoder $\rightarrow$ Projector $\rightarrow$ LLM". Native "Omni" architectures (integrated modalities) are missing and may exhibit different layer-prefix behaviors.
- **Samba/Sleek:** High-efficiency small-model architectures utilizing mixed linear/non-linear layers.

## Targets

`cpu` · `cuda-docker` (3090) · `cuda-spark` (DGX Spark GB10, primary) ·
`cuda-vast` (rented H200/B200, for VRAM-blocked models) · `cuda-k8s` (blackwell cluster).
