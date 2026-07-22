# How a Model Is Supported — Inference & Training Internals

Personal reference for adding new models to ScalarLM. Traces the full path from
config → loading → adapters for both inference and training, with source files,
config knobs, and dependencies. Line numbers are accurate as of 2026-06-03.

---

## TL;DR — the two support surfaces are different

| | Inference | Training |
|---|---|---|
| Who loads the model | **vLLM** (`vllm-fork` build) | **HuggingFace `transformers`** (`AutoModelForCausalLM`) |
| Support gate | Architecture must exist in vLLM's model registry | Any HF-loadable causal LM (`transformers>=5.5.0`) |
| Where the model name comes from | `config["model"]` (cray-config.yaml / `SCALARLM_MODEL`) | `job_config["llm_name"]` (per-job train_args) |
| ScalarLM's extra layer | LoRA/tokenformer adapters hot-loaded via vLLM `/v1/load_lora_adapter` | tokenformer/LoRA surgery on the HF module tree |
| Hardest constraint when adding a model | vLLM must support the arch | tokenformer surgeon must find `mlp` layers + `hidden_size` |

**Training is far more permissive than inference.** A model can almost always be
trained (HF loads it) but will only *serve* if the vLLM fork knows its architecture.

---

## 1. Inference

### 1.1 Config flow (where the served model is chosen)

```
default_config.py (Config.model)          # built-in default
        │  overridden by
        ▼
/app/cray/cray-config.yaml                # rendered from Helm values → configmap
        │  overridden by
        ▼
SCALARLM_MODEL env var                     # per-process override
```

- **`infra/cray_infra/util/default_config.py:8`** — `model: str = "tiny-random/gemma-4-dense"` (+ commented alternatives).
- **`infra/cray_infra/util/get_config.py`** — loads YAML at `SCALARLM_CONFIG_PATH` (default `/app/cray/cray-config.yaml`), then applies `SCALARLM_<KEY>` env overrides with type coercion. Every `Config` field is overridable as `SCALARLM_<UPPERCASE_FIELD>`.
- **Helm** — `deployment/helm/scalarlm/values*.yaml` `model:` → templated into the configmap (`_helpers.tpl:87`, `*_configmap.yaml`) → becomes `cray-config.yaml` in the pod.

### 1.2 Server startup (how vLLM is launched)

- **`infra/cray_infra/one_server/create_vllm.py`** — the in-process vLLM OpenAI server.
  - `build_vllm_cli_args(config)` builds the arg list, then `args.model = config["model"]` (`create_vllm.py:88`).
  - Runs vLLM's own `build_async_engine_client` / `init_app_state` — **the model is loaded by vLLM, not by ScalarLM.**
  - SM<8.0 GPUs: forces `VLLM_ATTENTION_BACKEND=FLASHMLA` and `dtype=float32`. CPU: `dtype=float32`.
  - Extra raw vLLM flags via `SCALARLM_VLLM_ARGS` env (override/dedup).
- **`infra/cray_infra/one_server/vllm_cli_args.py`** — `build_vllm_cli_args(config)`. Base flags:
  `--dtype`, `--max-model-len=auto`, `--gpu-memory-utilization`, `--tensor-parallel-size`,
  `--enable-auto-tool-choice`, `--tool-call-parser=hermes`, `--trust-remote-code`,
  conditional `--enable-lora` (gated on `config["enable_lora"]`),
  conditional `--limit-mm-per-prompt` (multimodal).

> `--trust-remote-code` is always passed, so models with custom HF architecture
> code can load *if* vLLM can execute their modeling code. The hard limit is still
> the vLLM fork's architecture coverage.

### 1.3 Inference-relevant config knobs (`default_config.py`)

| Field | Default | Purpose when adding a model |
|---|---|---|
| `model` | `tiny-random/gemma-4-dense` | The HF id vLLM serves |
| `dtype` | `auto` | `auto`=native; force `float32`/`bfloat16` for hw quirks |
| `gpu_memory_utilization` | `0.40` | Raise for big models / dedicated GPUs |
| `tensor_parallel_size` | `1` | **>1 required for multi-GPU / large MoE** |
| `max_model_length` | `256` | Served context cap (note `--max-model-len=auto` overrides at CLI) |
| `enable_lora` | `True` | Must stay true to hot-load tokenformer/LoRA adapters |
| `limit_mm_per_prompt` | `'{"image":2}'` | Multimodal input caps |
| `tokenformer_r` | `32` | Tokenformer low-rank dim (must match training) |
| `tokenformer_num_heads` | `4` | Tokenformer heads (must match training) |
| `tokenformer_cache_capacity` | `2` | Adapter LRU cache size |

### 1.4 ScalarLM's adapter layer at inference

Two distinct mechanisms — **know which is live:**

1. **ACTIVE — tokenformer/LoRA served as vLLM LoRA adapters.**
   Trained adapters are loaded into the already-loaded base model via vLLM's native
   endpoint. See `infra/cray_infra/one_server/create_generate_worker.py:268-297`
   (`/v1/load_lora_adapter`, `LoadLoRAAdapterRequest`).
   - Discovery: **`infra/cray_infra/training/vllm_model_manager.py:44`** `find_model()`
     scans `training_job_directory` for `*.pt` checkpoints and auto-registers them.
   - Runtime adapter code: `infra/cray_infra/adapters/model/tokenformer.py`
     (`TokenformerModel.from_local_checkpoint`, `ModelStateManager`),
     `infra/cray_infra/adapters/vllm/attention_adapter.py`.

2. **DORMANT — `ScalarLMModelRegistry`.**
   `infra/cray_infra/adapters/vllm/registry.py` + `adapters/model/models.py` wrap
   `LlamaForCausalLM` / `GemmaForCausalLM` / `Qwen2ForCausalLM` with a tokenformer
   adapter class. **This path is currently disabled** — `register_scalarlm_models`
   is commented out in `adapters/__init__.py:74-75,115-117`. Don't rely on it; it
   only adapts those 3 base classes anyway.

### 1.5 Inference dependencies

- **`vllm-fork`** — built in `Dockerfile` (the `vllm` stage, `pip install -e .`).
  Repo: `https://github.com/supermassive-intelligence/vllm-fork.git`, branch `main`.
  **This is the real inference-support boundary** — add-arch support lives here, in
  `vllm/model_executor/models/`.
- **`infra/requirements-vllm.txt`** — `fastapi-utils`, `typing-inspect`.
- `torch` (from base image), `transformers` (tokenizers / chat templates).

---

## 2. Training

### 2.1 Job config flow (where the trained model is chosen)

- **`infra/cray_infra/util/default_job_config.py`** — `JobConfig`.
  - `llm_name: str = "meta-llama/Llama-3.2-1B-Instruct"` — **the model to train.**
  - `adapter_type: "tokenformer"` (or `"lora"` / `"none"`).
  - `distribution_strategy: "fsdp"` (or `"ddp"` / `"no_distribution"`).
  - `dtype: "auto"`, `attn_implementation: "auto"` (→ sdpa, fallback eager),
    `gradient_checkpointing`, `max_steps`, `learning_rate`, `lora_config`, etc.
  - Set per job via train_args submitted through the SDK (not the global config).

### 2.2 Load path (how the model is materialized)

Entry: `ml/cray_megatron/main.py` → training harness → **`ml/cray_megatron/models/load_model.py`**:

1. `load_model_config()` — `AutoConfig.from_pretrained(llm_name)` + `AutoTokenizer.from_pretrained(llm_name)` (`load_model.py:41-56`).
2. `materialize_model()`:
   - `download_model(name)` → `ml/cray_megatron/huggingface/download_model.py` → `huggingface_hub.snapshot_download` (main rank only).
   - `AutoModelForCausalLM.from_pretrained(..., attn_implementation=sdpa|eager)` (`load_model.py:75-102`). **This is the training support gate — any HF causal LM loads here.** sdpa→eager fallback on `ValueError/ImportError/RuntimeError`.
   - optional gradient checkpointing (`use_cache=False`, `enable_input_require_grads`).
   - `add_adapters_to_model(...)` (see 2.3).
   - dtype conversion (per-job `dtype` wins over global; `auto`=native).
   - distribution strategy wrap (FSDP/DDP) + move to device.

### 2.3 Adapter surgery (ScalarLM's training value-add)

- **`ml/adapters/add_adapters_to_model.py`** — dispatch on `job_config["adapter_type"]`:
  - `tokenformer` → `ml/adapters/create_tokenformer_model.py`
  - `lora` → `ml/adapters/create_lora_model.py` (PEFT)
  - `none` → plain model
- **Tokenformer surgeon — `ml/tokenformer/tokenformer_surgeon.py`** (the compatibility-critical file):
  - `insert_adapter_modules()` walks `model.named_modules()` and wraps every layer
    whose **last path component contains `mlp`** (`_is_adapter_layer`, line 143-147)
    with a `TokenformerAdapter`.
  - **Excludes** vision/audio towers via `_NON_LANGUAGE_PATH_COMPONENTS`
    (`vision_tower`, `audio_tower`, `embed_vision`, `embed_audio`, `multi_modal_projector`) — so multimodal models train only the language MLPs.
  - **hidden_size resolution** (line 163-172): `model.config.text_config.hidden_size`
    (multimodal) → else `model.config.hidden_size` → else `model.model_config.hidden_size`.
    A model whose config doesn't expose hidden_size one of these ways will log an error and skip.
  - `create_tokenformer_model.py` then freezes the base and unfreezes only:
    `tokenformer*`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `norm`, `rotary_emb`,
    `embed_tokens`, `input_layernorm`, `post_attention_layernorm`, plus `lm_head` if
    the model has <100M params.
  - Tokenformer math: `infra/cray_infra/adapters/model/tokenformer.py`
    `TokenformerAdapter` (uses `tokenformer_r`, `tokenformer_num_heads` from config).

### 2.4 Output → handoff to inference

Training writes `.pt` checkpoints into `training_job_directory` (`/app/cray/jobs/<model>`).
Inference's `vllm_model_manager.find_model()` discovers those `.pt` files and serves
them as LoRA adapters on top of the base `config["model"]`.

### 2.5 Training dependencies

- **`infra/requirements-megatron.txt`** (GPU): `torchao>=0.16.1`, `accelerate`,
  `transformers>=5.5.0`, `huggingface_hub>=0.37`, `peft`.
- **`infra/requirements-megatron-cpu.txt`** (CPU/embedding): adds `sentence-transformers`.
- `gpu_aware_mpi` (built from `infra/cray_infra/training/gpu_aware_mpi` in the Dockerfile).
- **`transformers>=5.5.0` is the practical arch boundary for training** — a brand-new
  architecture needs a transformers version that ships its modeling code (or
  `trust_remote_code` + the model repo's custom code).

---

## 3. Adding a new model — checklists

### 3.1 To serve it (inference)

1. **Confirm vLLM-fork supports the architecture.** Check
   `vllm/model_executor/models/` in the fork for the arch class. If absent, the
   model will not load — you must add/port a vLLM model class (or use a build that has it).
2. Set the model name: Helm `values/*.yaml` `model:` (prod) **or** `SCALARLM_MODEL`
   env **or** `default_config.py:8` (dev default).
3. Size the runtime: bump `gpu_memory_utilization`, set `tensor_parallel_size>1` for
   multi-GPU / large MoE, set `dtype` if the hardware needs it.
4. Multimodal: set `limit_mm_per_prompt`.
5. If you want to hot-load adapters for it, keep `enable_lora: true`.
6. Custom-arch models: rely on `--trust-remote-code` (already passed) — still needs
   vLLM to be able to run the modeling code.

### 3.2 To train it

1. Set `llm_name` in train_args (per-job).
2. Confirm `AutoModelForCausalLM.from_pretrained(llm_name)` works under the pinned
   `transformers>=5.5.0` (bump the requirement or use `trust_remote_code` for new arch).
3. **Tokenformer compatibility** (if `adapter_type=tokenformer`):
   - Model config must expose `hidden_size` (or `text_config.hidden_size`).
   - Model must have submodules whose last name component contains `mlp`.
   - Non-standard blocks (Mamba, MoE with differently-named experts) may not get
     wrapped — verify `_is_adapter_layer` matches the layers you expect, or use
     `adapter_type=lora`/`none`.
4. Pick `distribution_strategy` (fsdp for large, ddp/no_distribution for small).
5. Set `dtype` per-job if the hardware needs it (e.g. `float32` on Apple-Silicon CPU).
6. Multimodal: vision/audio towers are auto-excluded from tokenformer — only the LM trains.

### 3.3 Files you'll most likely touch

| Goal | File(s) |
|---|---|
| Change served model default | `infra/cray_infra/util/default_config.py:8` |
| Per-deployment served model | `deployment/helm/scalarlm/values/*.yaml` |
| vLLM launch flags | `infra/cray_infra/one_server/vllm_cli_args.py` |
| Add inference arch support | `vllm-fork` repo (`vllm/model_executor/models/`) |
| Change trained model default | `infra/cray_infra/util/default_job_config.py` (or train_args) |
| Tokenformer layer matching | `ml/tokenformer/tokenformer_surgeon.py` |
| Tokenformer trainable params | `ml/adapters/create_tokenformer_model.py` |
| Training deps / transformers pin | `infra/requirements-megatron*.txt` |

---

## 4. Gotchas

- **Inference ≠ training support.** You can train a model the server can't serve
  (HF loads it, vLLM doesn't know the arch). Always check both surfaces.
- **The `ScalarLMModelRegistry` is dormant** (commented out in `adapters/__init__.py`).
  It is *not* what enforces or provides model support today.
- **tokenformer_r / tokenformer_num_heads must match** between the training config
  and the serving config, or the loaded adapter weights won't align.
- **`--max-model-len=auto`** at the CLI overrides the `max_model_length` config field.
- The README "supported models" table is aspirational/deployment-facing and is *not*
  enforced anywhere in code (`docs/test-plan.md:37` admits it's untested). See
  `docs/personal/models-audit.md`.
