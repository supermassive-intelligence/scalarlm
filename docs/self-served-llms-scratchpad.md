# Self-Served LLMs — Scratchpad

> Working notes on popular open-weight LLMs for self-hosting, organized by
> deployment target. Compiled June 2026. Benchmarks/version numbers move fast —
> treat as a snapshot, re-verify before committing to a model.

## TL;DR

Open-weight models are now within single-digit percentage points of frontier
closed models on everyday work, at ~4–10× lower cost. The dominant families:
**Qwen, DeepSeek, Llama, GLM, Kimi, Mistral, Gemma, Phi**.

For ScalarLM fine-tuning work specifically: **Qwen** and **Mistral** have the
most mature fine-tuning tooling and ship permissive **Apache 2.0** licenses.

> **Cross-platform note:** the categories below are "smallest comfortable
> target," not exclusive buckets. Most models run on _several_ targets — the
> same weights that fit a 1-GPU box also run on a homelab rig, a DGX Spark, or
> (quantized) on MLX/Apple. See the **Cross-platform matrix** below for the full
> span per family. Rule of thumb: a model that fits a smaller target also runs
> on every larger one (CPU = floor, datacenter = ceiling).

---

## By deployment category

### CPU-only / low-resource
Small, heavily-quantizable models that run without a GPU (slow but functional).
| Model | Params | License | Notes |
|-------|--------|---------|-------|
| Phi-4 | 14B | MIT | Smallest capable footprint, ~8GB; runs quantized on CPU |
| Mistral Small 4 | 24B | Apache 2.0 | 256K context; GGUF quant runs on CPU with patience |
| Gemma 3 (smaller variants) | 4B–12B | Gemma license | Good quality-per-byte for CPU inference |

### Single consumer GPU (1× 16–24GB, e.g. 3090/4090)
| Model | Params | VRAM (approx) | License | Notes |
|-------|--------|---------------|---------|-------|
| Gemma 3 27B | 27B | ~16GB | Gemma license | Easiest single-GPU self-host |
| Phi-4 | 14B | ~8GB | MIT | Fits comfortably, room for context |
| Mistral Small 4 | 24B | fits w/ quant | Apache 2.0 | Single consumer GPU with quantization |
| Qwen 3 (14B–32B tier) | 14–32B | 16–24GB | Apache 2.0 | Strong reasoning at this size |

> Maps to our 3090 box — see [[nvidia-target-remote-host]].

### DGX Spark / single high-memory node (128GB unified, GB10)
Unified-memory boxes change the math — large MoE models fit where they wouldn't
on discrete consumer VRAM.
| Model | Params | License | Notes |
|-------|--------|---------|-------|
| GLM-5 / GLM-4.7 | large MoE | MIT | Cost-efficient general use, tops open rankings |
| DeepSeek V4 | MoE | MIT | Best perf-per-inference-cost for self-hosted |
| Qwen 3 235B-A22B | 235B MoE (22B active) | Apache 2.0 | Best all-round reasoning + coding |
| Kimi K2.6 | large | open | Strongest open coding/agentic, 256K context |

### MLX / Apple Silicon (M-series, unified memory)
MLX-format quantized models; viable range scales with RAM (16GB → 192GB).
| Model | Params | Notes |
|-------|--------|-------|
| Qwen 3 (4B–32B MLX) | 4–32B | Best-supported MLX conversions, active community |
| Gemma 3 (4B–27B MLX) | 4–27B | Good quality, broad MLX coverage |
| Mistral Small 4 MLX | 24B | Fits 32GB+ Macs |
| DeepSeek / Llama 4 (quantized) | large | Need 64–192GB unified for usable quants |

### Homelab / multi-GPU rigs
Mixing 2–8 consumer or prosumer GPUs; aim for the strongest model that fits.
| Model | Params | License | Notes |
|-------|--------|---------|-------|
| Llama 4 Scout | MoE | Llama license | Ultra-long context (~10M tokens), huge fine-tune community |
| Qwen 3 235B-A22B | 235B MoE | Apache 2.0 | Frontier-competitive, splits well across GPUs |
| DeepSeek V4 | MoE | MIT | Best cost/perf for self-host |
| GLM-5 | MoE | MIT | Cost-efficient |

### Datacenter / production serving (8×H100/B200+)
Full-precision or light-quant frontier open weights.
| Model | License | Notes |
|-------|---------|-------|
| DeepSeek V4 Pro | MIT | Code/math leader |
| Qwen 3.7 Max | Apache 2.0 | Broad reasoning, multilingual |
| Llama 4 Maverick/Scout | Llama license | Tool calling, community fine-tunes |
| Mistral Large 3 | Mistral license | European languages, compliance |

---

## Cross-platform matrix

Which targets each family can realistically run on (✅ native/comfortable,
🟡 only with quantization or trimmed context, — impractical). Larger models in a
family shift the row right.

| Family | CPU | 1-GPU | DGX Spark | MLX/Apple | Homelab (multi-GPU) | Datacenter |
|--------|:---:|:-----:|:---------:|:---------:|:-------------------:|:----------:|
| Phi-4 14B | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gemma 3 (4–27B) | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mistral Small 4 24B | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ |
| Qwen 3 (14–32B dense) | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen 3 235B-A22B (MoE) | — | — | 🟡 | 🟡 | ✅ | ✅ |
| Llama 4 Scout (MoE) | — | — | 🟡 | 🟡 | ✅ | ✅ |
| DeepSeek V4 (MoE) | — | — | 🟡 | 🟡 | ✅ | ✅ |
| GLM-5 (MoE) | — | — | 🟡 | 🟡 | ✅ | ✅ |
| Kimi K2.6 | — | — | 🟡 | — | 🟡 | ✅ |
| Mistral Large 3 | — | — | 🟡 | 🟡 | ✅ | ✅ |

---

## ScalarLM support — what we can actually test

Checked against this repo on `georgi/finetune-sweep` (2026-06-12).

**vLLM fork:** `supermassive-intelligence/vllm-fork`, branch `main`, based on
**vLLM v0.19.0** (the `scalarlm-on-v0.19.0` lineage referenced in `Dockerfile`).
The local `./vllm` dir is empty — the CUDA/vLLM stack builds and runs on the
remote boxes (3090, blackwell-maxq-0), not this machine.

Two very different code paths:

- **Serving** (`infra/cray_infra/adapters/model/models.py`) registers ScalarLM
  adapters for a **fixed allowlist of 3 architecture classes** only:
  `LlamaForCausalLM`, `GemmaForCausalLM` (the original Gemma **v1** class), and
  `Qwen2ForCausalLM`. Anything else has no tokenformer/serving adapter and is
  not servable as-is. ⚠️ The class names are exact: **Gemma 2/3** use
  `Gemma2/Gemma3ForCausalLM` and **Qwen3** uses `Qwen3ForCausalLM`/MoE classes —
  none of those are registered, so "Gemma 3" / "Qwen 3" are **not** servable
  without adding adapters. Also note the known LoRA-serving no-op for causal LMs
  on the fork (`normalize_lora_key` strips the `model.` prefix) — adapters serve
  base output until that fork fix lands.
- **Finetuning** (`ml/cray_megatron/models/load_model.py`) is
  **architecture-agnostic**: it loads via HF `AutoModelForCausalLM.from_pretrained`
  and attaches LoRA/tokenformer adapters through PEFT. Any HF causal LM that
  `transformers` supports can be fine-tuned. SDPA is the default attention path
  (flash-attn was removed; the code notes Qwen3-family flash-attn bugs), with an
  automatic fallback to `eager` for configs that reject SDPA.

| Model (family) | Serve in ScalarLM? | Finetune in ScalarLM? | Notes |
|----------------|:------------------:|:---------------------:|-------|
| Llama (incl. Llama 4 dense parts) | ✅ `scalarlm_llama` | ✅ HF AutoModel | First-class; LoRA-serving no-op caveat applies |
| Qwen **2** (Qwen2 / Qwen2-VL) | ✅ `scalarlm_qwen2` | ✅ | The Qwen class that is actually registered |
| Qwen **3** / Qwen3-MoE | ❌ not registered | ✅ (SDPA; eager fallback) | Finetune works; serving needs a `Qwen3` adapter |
| Gemma **v1** | ✅ `scalarlm_gemma` | ✅ | The registered Gemma class |
| Gemma **2 / 3** | ❌ not registered | ✅ | Popular Gemma 3 27B is finetune-only here |
| Mistral (Small 4 / Large 3) | ❌ no adapter | ✅ | Finetune via HF; serving needs a `Mistral` adapter |
| Phi-4 | ❌ no adapter | ✅ | Finetune-only without a new serving adapter |
| DeepSeek (V4/R1, MoE) | ❌ no adapter | 🟡 HF support varies | MoE finetuning is heavier; verify transformers support |
| GLM-5 / Kimi K2.6 | ❌ no adapter | 🟡 | Verify HF `AutoModel` support before relying on it |

**Easiest end-to-end (serve **and** finetune today, no fork changes):**
**Llama**, **Qwen2**, and **Gemma v1** — ideally the tiny test variants already
in `infra/cray_infra/util/default_config.py` (`masint/tiny-random-llama`,
`Qwen/Qwen2-7B-Instruct`, `google/gemma-3-270m-it` for finetune,
`tiny-random/gemma-4-dense`, etc.). To serve a newer Gemma 3 / Qwen 3 / Mistral,
a serving adapter for that architecture class has to be added first.

---

## Selection cheatsheet

- **Single consumer GPU:** Gemma 3 27B, Mistral Small 4 (quant), Phi-4 14B
- **Best fine-tuning ecosystem:** Qwen, Mistral (best-documented tooling)
- **Most permissive licenses:** Qwen / Mistral Small (Apache 2.0), DeepSeek / GLM (MIT), Phi-4 (MIT)
- **Agentic coding:** Qwen 3.6 Plus, Kimi K2.6, DeepSeek V4
- **Deep math reasoning:** DeepSeek R1 / V4
- **Ultra-long context:** Llama 4 Scout (~10M tokens), Kimi K2.6 (256K)
- **European languages / compliance:** Mistral Large 3

---

## License quick reference

| License | Models | Implications |
|---------|--------|--------------|
| Apache 2.0 | Qwen 3/3.5, Mistral Small 4 | No usage caps, no royalties, no geo restrictions |
| MIT | DeepSeek, GLM-5, Phi-4 | Fully permissive |
| Llama license | Llama 4 | Some usage/scale restrictions — read before commercial use |
| Gemma license | Gemma 3 | Google terms — review acceptable-use clauses |

---

## Sources

- [Best Open-Source LLMs in 2026 — HuggingFace](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Best Open Source LLM 2026 (ranking + Ollama) — WhatLLM.org](https://whatllm.org/best-open-source-llm)
- [Best Open-Source LLMs for Agentic Coding 2026 — MindStudio](https://www.mindstudio.ai/blog/best-open-source-llms-agentic-coding-2026)
- [Best Open-Source LLM May 2026 (Llama 4 vs Qwen 3.5 vs DeepSeek V4 vs Gemma 4 vs Mistral) — Codersera](https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/)
- [Best Open Source LLMs 2026 (benchmarks, licenses, GPU deployment) — AceCloud](https://acecloud.ai/blog/best-open-source-llms/)
- [Best Open Source LLM January 2026 — WhatLLM.org](https://whatllm.org/blog/best-open-source-models-january-2026)
- [Best Open Source LLM February 2026 — WhatLLM.org](https://whatllm.org/blog/best-open-source-models-february-2026)
- [Best Self-Hosted LLM Leaderboard 2026 — Onyx](https://onyx.app/self-hosted-llm-leaderboard)
- [Open Source LLM Comparison Table 2026 — ComputingForGeeks](https://computingforgeeks.com/open-source-llm-comparison/)

_Last updated: 2026-06-12_
