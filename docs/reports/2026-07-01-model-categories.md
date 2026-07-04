# Model Categories in `cray-megatron`

**Date:** 2026-07-01

The training harness (`ml/cray_megatron/`) dispatches model-loading, LoRA
target-module resolution, and training-step construction by architecture
category. Four categories exist today. Membership below is read directly
from `test/finetune_sweep/finetune-sweep.yaml` (the finetune sweep's model
list) plus the arch-class comments already recorded there; it will drift as
models are added/removed from that file.

## Sweep verdict status (updated 2026-07-04)

Markers below record the sweep's end-to-end verdict (restart → train tiny
LoRA → serve → check memorization) where one has been observed. Legend:

- ✅ **PASS** — trained, served, and reproduced the golden output (memorized).
- ⛔ **BLOCKED** — hit a hard wall; reason noted inline and in the summary below.
- *(unmarked)* — in the sweep list but no separate terminal verdict recorded
  here yet (e.g. staged-but-not-run large entries, or size-ladder fill-ins
  expected to PASS by arch precedent).

### Blocked models and reasons

| Model | Category | Verdict | Root cause |
|---|---|---|---|
| `zai-org/GLM-4-9B-Chat` | Causal LM (custom ChatGLM) | ⛔ TRAIN_FAILED | vendor `modeling_chatglm.py` vs transformers 5.x — 5 layers deep, terminal wall is `ChatGLMConfig` missing `use_cache` in forward. Clean path: swap to a natively-supported GLM (Glm4v/Glm46V/GlmOcr), don't fork ChatGLM. See `2026-07-03-vlm-vendor-code-transformers5-blockers.md`. |
| `OpenGVLab/InternVL3-8B` | Multimodal (custom) | ⛔ TRAIN_FAILED | vendor code vs transformers 5.x — 4 layers deep; loads then `InternVLChatModel` has no `generation_config` (read by `add_eos_token`). |
| `allenai/Molmo-7B-D-0924` | Multimodal (custom) | ⛔ TRAIN_FAILED | vendor code vs transformers 5.x — deepest; loads AND runs, fails in vision-language **forward compute** (tensor broadcast `[1,28,480,480]` vs `[1,1,28,480,480]`) — a real modeling bug, not a one-line shim. |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE (separate experts) | ⛔ VRAM | 47B/~94GiB bf16 dies in phase-1 **training model-load** (~37% of shards → SIGTERM/SIGKILL) and takes the co-resident API worker down on a single GB10 (128GiB unified). Same TRAIN-load wall the 30B cleared, exceeded at 47B. Needs a multi-GPU / larger-headroom target. Its intended converter test (SEPARATE-expert path) was never reached — use `allenai/OLMoE-1B-7B` on the Spark instead. |

The three custom-code models above were accepted as blocked under
systematic-debugging Phase 4.5 (each shim advances one layer and reveals the
next — the wrong approach, not a failed hypothesis); per-model vendor-code
shimming was stopped by decision on 2026-07-03.

Two MoE report candidates below were **excluded before running** (not blocked
by a sweep verdict) purely on size: `zai-org/GLM-4.5-Air` (~106B) and
`meta-llama/Llama-4-Scout-17B-16E` (109B, also gated) are ~210GiB in bf16 —
over the GB10's 128GiB unified pool even phase-scaled — so they're k8s /
multi-GPU targets only.

## 1. Causal LM

Dispatched via `AutoModelForCausalLM`. The default/majority path — no
special predicate needed in `load_model.py`.

- `Qwen/Qwen2.5-0.5B` (Qwen2ForCausalLM) — ✅ PASS
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct` (bf16 override)
- `Qwen/Qwen2-7B-Instruct`
- `Qwen/Qwen3-8B` (Qwen3ForCausalLM) — ✅ PASS (bf16)
- `masint/tiny-random-llama` — ✅ PASS
- `meta-llama/Llama-3.2-1B-Instruct` — ✅ PASS
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-270m-it`
- `google/gemma-3-270m` (base, no chat template)
- `google/gemma-3-4b-it` (Gemma3ForCausalLM) — ✅ PASS (serve no-op resolved)
- `tiny-random/gemma-4-dense` (Gemma4 arch, dense/non-vision checkpoint)
- `EssentialAI/rnj-1-instruct` (Gemma3ForCausalLM, 8B) — ✅ PASS (bf16)
- `mistralai/Mistral-7B-Instruct-v0.3` (MistralForCausalLM) — ✅ PASS (bf16; fp32 mode-collapsed)
- `microsoft/phi-4` (Phi3ForCausalLM, 14B) — ✅ PASS (bf16)

## 2. Multimodal / Conditional Generation

Dispatched via `AutoModelForImageTextToText`, gated by `is_multimodal(model_config)`
(`ml/cray_megatron/megatron/doc_mask.py`). Requires the defensive
`mm_token_type_ids` fix in `training_loop.py`.

- `masint/tiny-random-qwen2-vl`
- `Qwen/Qwen2-VL-7B-Instruct` — ✅ PASS (900-step budget)

## 3. MoE

Orthogonal to the above — a dense or multimodal model's `AutoModel*` class is
unchanged; what differs is `resolve_target_modules.py`'s `_moe_servable_linear_paths`,
which excludes routed-expert and router paths from LoRA so the resulting
`.pt` adapter stays compatible with vLLM's `FusedMoEWithLoRA` serving path.

- `yujiepan/qwen3-moe-tiny-random` (Qwen3MoeForCausalLM) — ✅ PASS (validates expert-LoRA converter)
- `Qwen/Qwen3-30B-A3B-Instruct-2507` (Qwen3MoeForCausalLM) — ✅ PASS (phase-scaled; on-device device_map load fix)
- `Qwen/Qwen3.5-35B-A3B` (Qwen3MoeForCausalLM, newer lineage)
- `Qwen/Qwen3.6-35B-A3B` (Qwen3MoeForCausalLM, newest lineage)
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` (Qwen2MoeForCausalLM; shared + routed experts)

## 4. Diffusion (encoder-decoder, block-diffusion) — new, training-only

Dispatched via `DiffusionGemmaForBlockDiffusion.from_pretrained()` directly —
not registered under either `AutoModelForCausalLM` or
`AutoModelForImageTextToText` — gated by a new `is_diffusion(model_config)`
predicate (`doc_mask.py`), checked before the multimodal fork. See
`docs/superpowers/specs/2026-07-01-diffusiongemma-design.md`, ADR 0007
(decoder-only LoRA scope), and ADR 0008 (vLLM serving deferred — this
category is training-only, excluded from the finetune sweep's closed loop).

- `google/diffusiongemma-26B-A4B-it` — not yet added to
  `test/finetune_sweep/finetune-sweep.yaml`; planned once the training path
  lands.

---

## Potential Sweep Candidates (not yet in the sweep)

Researched via web search (HuggingFace/vendor trending lists, 2026-07-01) and
filtered against the sweep's existing size ceiling (fp32 default: ~14B
comfortable, 32B borderline/OOM co-located, 72B omitted; bf16 override halves
footprint — the sweep already uses this for 14B+ dense and 30-35B MoE
entries) and gating status where known. Sizes/licenses should be re-verified
on the HF model card before adding — this list is a starting point, not a
final decision.

### 1. Causal LM

- `microsoft/Phi-4-mini-instruct` — ~3.8B, MIT, ungated. Small/fast; good
  complement to the existing `phi-4` (14B) entry. **✅ PASS** (added to sweep;
  bf16 + lr 1e-3 — the sweep-default 3e-3 oscillated and served an
  out-of-basin adapter → NO_MEMORIZATION).
- `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B` —
  Apache-2.0, ungated. The sweep only has `Qwen3-8B`; these fill out the
  dense Qwen3 size ladder (32B is the borderline-fp32 case, same as
  `Qwen2.5-14B-Instruct`'s bf16 treatment would likely be needed).
- `zai-org/GLM-4-9B-Chat` (or current GLM-4.x dense chat checkpoint) —
  MIT-family license; a non-Qwen/Llama/Gemma/Mistral/Phi family, broadens
  arch coverage. **⛔ BLOCKED — TRAIN_FAILED** (added to sweep; custom
  ChatGLM code vs transformers 5.x, terminal at missing `use_cache`). Clean
  path is a natively-supported GLM, not a ChatGLM fork.
- `allenai/OLMo-2-1124-7B-Instruct` (`Olmo2ForCausalLM`) — 7B dense,
  Apache-2.0, ungated. **Fully-open** (data + code + weights), reordered-norm
  arch distinct from Llama/Qwen; a clean-room dense family with zero current
  coverage. vLLM lists `Olmo2ForCausalLM` as ✅LoRA. *(added 2026-07-04;
  not yet run.)*

### 2. Multimodal / Conditional Generation

Sweep currently only exercises Qwen2-VL. Diversifying the multimodal
category by vendor/arch is higher-value than adding more Qwen2-VL sizes:

- `mistralai/Pixtral-12B-2409` — Apache-2.0, ungated. Mistral's first VLM;
  distinct arch family from Qwen2-VL. **✅ PASS** (added as the HF-native
  `mistral-community/pixtral-12b`; bf16 + lr 5e-4 — the default 3e-3 diverged
  at end-of-warmup and served the diverged adapter).
- `OpenGVLab/InternVL3-8B` (or a smaller InternVL3 checkpoint) — MIT-licensed,
  strong open-weight VLM; worth checking exact chat-template/processor
  support in the installed `transformers` version before adding.
  **⛔ BLOCKED — TRAIN_FAILED** (custom code vs transformers 5.x; loads then
  missing `generation_config`).
- `allenai/Molmo-7B-D-0924` — Apache-2.0, ungated, AI2-released; distinct
  vision-tower design from Qwen2-VL/Pixtral. **⛔ BLOCKED — TRAIN_FAILED**
  (deepest; fails in forward-compute tensor broadcast, a real modeling bug).
- `Qwen/Qwen2.5-VL-7B-Instruct` — same family as the existing `Qwen2-VL-7B`
  entry but newer generation; lower-priority than the above three for arch
  diversity, but a natural drop-in given Qwen2-VL already works.
  **✅ PASS** (added to sweep; 900-step budget, `train_timeout: 4800`).

### 3. MoE

- `mistralai/Mixtral-8x7B-Instruct-v0.1` — 47B total / 12.9B active,
  Apache-2.0, ungated. Classic MoE arch distinct from Qwen{2,3}Moe.
  **⛔ BLOCKED — VRAM** (validated 2026-07-03 on cuda-spark / GB10 128GiB
  unified). The "fits comfortably" estimate below was wrong: at ~94GiB bf16
  it dies in phase-1 **training model-load** (~37% of shards → SIGTERM/SIGKILL)
  and takes the co-resident API worker down — the same TRAIN-load wall the
  30B cleared, exceeded at 47B. Its intended converter test (Mixtral uses
  SEPARATE per-expert `w1/w2/w3` — a different PEFT export than Qwen3Moe's
  grouped experts) was never reached; router/expert **target resolution** is
  already handled generically by name (`block_sparse_moe.gate` /
  `.experts.*`). Needs a multi-GPU / larger-headroom target; to test the
  separate-expert converter on the Spark use `allenai/OLMoE-1B-7B` instead.
- `zai-org/GLM-4.5-Air` — ~106B total / 12B active. Likely exceeds the
  current co-located size wall even in bf16 (~24GiB active-only estimate is
  misleading; total weight footprint at load is what matters) — would need
  phase-scaling (k8s target only, per the existing 30-35B entries' staging
  note) rather than cuda-spark co-located.
- `meta-llama/Llama-4-Scout-17B-16E-Instruct` — 109B total / 17B active,
  gated (needs HF_TOKEN + license acceptance, same pattern as Llama-3.x
  entries). Also likely needs phase-scaling given total footprint.
- `ibm-granite/granite-4.0-h-tiny` (`GraniteMoeHybridForCausalLM`) —
  ~7B total / ~1B active, Apache-2.0, ungated. **New architectural category
  for the sweep: hybrid Mamba-2 / attention + MoE** (GQA + Mamba2 layers +
  routed MoE with shared experts, SwiGLU, RMSNorm, tied embeddings). Highest
  arch-diversity value here — it stresses `resolve_target_modules` on
  SSM/Mamba projection layers (which nothing else in the sweep has) *and* the
  MoE expert-LoRA path in one model, at a size that fits co-located. vLLM
  lists `GraniteMoeHybridForCausalLM` as ✅LoRA, but the **fork-registry check
  is mandatory** — the pinned vLLM-0.19 fork may not carry the hybrid arch.
  *(added 2026-07-04; not yet run.)*
- `microsoft/Phi-mini-MoE-instruct` (`PhiMoEForCausalLM`) — 7.6B total /
  2.4B active, MIT, ungated, 4k context. Mixtral-style **SEPARATE** experts
  (per-expert `w1/w2/w3`), 16 experts top-2. **Directly closes the converter
  question Mixtral couldn't reach** (Mixtral died at VRAM before the
  separate-expert converter ran) — this is small enough to actually exercise
  that path on the Spark. vLLM lists `PhiMoEForCausalLM` as ✅LoRA. Fork
  ParamWrapper likely needs `lora_dropout: 0` (same knob as qwen3-moe-tiny /
  Qwen1.5-MoE). *(added 2026-07-04; not yet run.)*

Larger frontier MoE releases (DeepSeek-V3.2, GLM-5.2, Kimi K2.6, Qwen3-235B-A22B,
Qwen3-Coder-480B-A35B) are hundreds of billions of total params — well
outside this repo's single-GPU (Spark/3090/Blackwell) sweep footprint even
phase-scaled; not realistic candidates without a multi-GPU/sharded training
path.

### 4. Diffusion

- `GSAI-ML/LLaDA-8B-Instruct` (and `LLaDA-8B-Base`) — open HF checkpoints,
  the most accessible non-Gemma dLLM. **Different architecture than
  DiffusionGemma**: LLaDA is a flat (non encoder-decoder) masked-diffusion
  model using an actual `[MASK]` token, not uniform-state corruption — the
  `is_diffusion()`/loader/corruption design in
  `docs/superpowers/specs/2026-07-01-diffusiongemma-design.md` is
  DiffusionGemma-specific and would NOT directly generalize to LLaDA without
  re-verifying its forward signature and corruption scheme against its own
  `transformers` source, the same way this session did for DiffusionGemma.
- `Dream-org/Dream-v0-Instruct-7B` / `Dream-org/Dream-Coder-v0-Instruct-7B` —
  open discrete-diffusion checkpoints; same caveat as LLaDA (verify actual
  architecture before assuming DiffusionGemma's design transfers).
- Mercury (Inception Labs) — **not a candidate**: no public downloadable
  weights as of this research (API-only).

Sources:
- [Best Open-Source LLM Models in 2026](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [State of Open Source on Hugging Face: Spring 2026](https://huggingface.co/blog/huggingface/state-of-os-hf-spring-2026)
- [Best Open-Weight Vision-Language Models 2026](https://presenc.ai/research/best-open-weight-vision-language-models-2026)
- [Multimodal AI: Open-Source Vision Language Models 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [7 Top Mixture of Experts AI Models for Developers in 2026](https://www.labellerr.com/blog/top-open-source-moe-llms/)
- [The Open-Source LLM Revolution 2026](https://www.alphamatch.ai/blog/open-source-llm-comparison-blog-2026)
- [Deploy Diffusion Language Models on GPU Cloud (2026)](https://www.spheron.network/blog/deploy-diffusion-language-models-dllm-gpu-cloud-2026/)
- [Awesome-DLMs survey repo](https://github.com/VILA-Lab/Awesome-DLMs)
