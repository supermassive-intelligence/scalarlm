# Supported models — LoRA fine-tune sweep

**Status summary as of 2026-07-06.** Consolidates verdicts that were scattered
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
| `Qwen/Qwen2-7B-Instruct` | Qwen2 | 450-step budget (slower-converging) | SS22 |
| `meta-llama/Llama-3.2-1B-Instruct` | Llama | gated (HF_TOKEN) | SS22 |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama | gated | SS22 |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama | gated | SS22 |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | **bf16** — fp32 mode-collapsed at lr 3e-3 | SS22, YAML |
| `microsoft/phi-4` | Phi3 (14B) | bf16, 450-step | SS22, YAML |
| `Qwen/Qwen3-8B` | Qwen3 | bf16 | SS22 |
| `google/gemma-3-4b-it` | Gemma3 | gated; earlier "serve no-op" was cross-arch contamination, since fixed | SS22 |
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
| `microsoft/Phi-mini-MoE-instruct` | PhiMoE (7.6B/2.4B) | TRAIN_FAILED (router returned a tuple, wrapped as LoRA target). **Router-exclusion fix landed** (2ecf306); **separate-expert training targets landed** (plan step A). Re-run now trains experts but still serves without them (NO_MEM) until the **separate-expert serve converter** (plan step B) lands | YAML, MEM |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | Qwen2MoE (14B/2.7B) | Trains + serves + adapter applied, but served string is **scrambled** → NO_MEMORIZATION. Cause: its **shared expert** is mis-mapped by the converter (validated only on routed grouped experts). Converter gap, deferred | P30B |

### Staged (configured, arch-precedent PASS, not yet run on GPU)
| Model | Arch | Rationale | Source |
|---|---|---|---|
| `Qwen/Qwen3-1.7B` | Qwen3 | Qwen3-8B PASSes → expect PASS | YAML |
| `Qwen/Qwen3-4B` | Qwen3 | ″ | YAML |
| `Qwen/Qwen3-14B` | Qwen3 | ″ (bf16, 450-step) | YAML |
| `Qwen/Qwen3-32B` | Qwen3 (dense) | phase-scaled; **dense-32B activation memory is high** — TRAIN_FAILED/OOM is expected-territory, not a bug | YAML |
| `Qwen/Qwen3.5-35B-A3B` | Qwen3MoE | staged; needs phase-scaling + the ~70GiB download | YAML |
| `Qwen/Qwen3.6-35B-A3B` | Qwen3MoE | ″ | YAML |

---

## ❌ Unsupported / blocked

| Model | Arch | Blocker | Source |
|---|---|---|---|
| `tiny-random/gemma-4-dense` | Gemma4 (mm fixture) | NO_MEMORIZATION — capacity ruled out (tiny-random-llama memorizes); it's the `Gemma4ForConditionalGeneration` adapter key mapping | SS22, G4D |
| `masint/tiny-random-qwen2-vl` | Qwen2-VL (mm fixture) | RESTART_FAILED — vLLM crash-loops loading the malformed synthetic base config. Low impact (real Qwen2-VL-7B PASSes) | SS22 |
| `zai-org/GLM-4-9B-Chat` | ChatGLM (custom code) | BLOCKED — vendor ChatGLM code vs transformers 5.x, 5 layers deep (ends at `ChatGLMConfig has no attribute 'use_cache'`). Clean path = swap to a natively-supported GLM arch | VLM3 |
| `OpenGVLab/InternVL3-8B` | InternVL (custom code, mm) | BLOCKED — vendor code vs transformers 5.x, 4 layers deep (`InternVLChatModel has no attribute 'generation_config'`) | VLM3 |
| `allenai/Molmo-7B-D-0924` | Molmo (custom code, mm) | BLOCKED — deepest: tensor-broadcast mismatch inside the vision-language **forward compute** (a real modeling bug under this transformers/torch, not a one-line shim) | VLM3 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Mixtral (47B/12.9B) | BLOCKED — VRAM. Dies in phase-1 training at model-load on the single Spark's 128GiB unified pool (co-located API worker went down with it, hung the box). **Disabled** in the manifest; needs multi-GPU / a bigger card (`cuda-vast` H200/B200). Never reached the separate-expert converter test | YAML, MEM |

### Documented-but-not-tested (README aspirational)
The README "supported models" table lists production-aspirational models not wired
into any sweep or runtime config: `google/gemma-3-27b-it`, `Qwen/Qwen2-32B-Instruct`,
`Qwen/Qwen3.5-122B-A10B`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`nvidia/Nemotron-3-Super-120B`. These are untested by the sweep. GLM-4.5-Air (~106B)
and Llama-4-Scout (109B) were considered as MoE candidates but tagged k8s/multi-GPU
only (~210GiB bf16 exceeds a single Spark even phase-scaled).

---

## Cross-cutting findings

- **Size ceiling on a single Spark:** fp32 co-located ~14B is comfortable, ~32B is
  borderline/OOM. Large models (30B+ MoE, dense 32B) run only via **phase-scaling**
  (train, tear down, then serve — peak GPU = 1 model). Models over the 128GiB
  unified pool even phase-scaled (Mixtral 47B, 70B+ dense) need `cuda-vast`
  (rented H200/B200) or k8s multi-GPU.
- **dtype:** every ≥7B model that mode-collapsed in fp32 at lr 3e-3 was fixed by
  **bf16** (Mistral, phi-4, Qwen3-8B, rnj-1, Qwen2.5-14B). Multimodal models
  descend cleanly in fp32 but are **budget-starved** — they need 900 steps, not
  the 300/450 default.
- **MoE memorization requires the experts.** Attention-only (or attention+dense)
  LoRA serves but can't memorize on pure-sparse MoEs — the expert-LoRA converter is
  the real unlock. Exception: hybrids with an always-on dense path (granite's
  `shared_mlp`) memorize through that without a converter.
- **Converter coverage:** grouped-expert Qwen3MoE (2D gated FusedMoE) is validated;
  **separate-expert** (Mixtral/PhiMoE) and **shared-expert** (Qwen2MoE/deepseek)
  arches are open converter gaps.

## Targets

`cpu` · `cuda-docker` (3090) · `cuda-spark` (DGX Spark GB10, primary) ·
`cuda-vast` (rented H200/B200, for VRAM-blocked models) · `cuda-k8s` (blackwell cluster).
