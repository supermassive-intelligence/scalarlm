from pydantic import BaseModel

from typing import Optional, Union


class LoraConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Union[str, list] = "all-linear"  # or list of module names


class DiffusionConfig(BaseModel):
    # DiffusionGemma canvas denoising knobs (see ADR 0007/0008 and
    # docs/superpowers/specs/2026-07-01-diffusiongemma-design.md). Only consumed
    # when the model is DiffusionGemma (auto-detected via is_diffusion()); ignored
    # otherwise. Nested block like lora_config; MUST be declared here or
    # get_job_config()'s Pydantic model silently DROPS it from train_args (the same
    # footgun that hid trust_remote_code).
    canvas_length: int = 256   # decoder block size = the loader's pad/truncate target
    eps: float = 0.001         # minimum corruption level for t ~ U(eps, 1)
    # Per-step probability of the two-pass self-conditioning scheme (Analog-Bits):
    # feed the model's own no-grad prediction back as the self-conditioning signal
    # to match the iterative serve decode. 0 disables it (single-pass v1 training).
    self_conditioning_prob: float = 0.5

class JobConfig(BaseModel):

    job_directory: str
    training_data_path: str
    dataset_hash: str

    #llm_name: str = "masint/tiny-random-llama"
    llm_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    # Training
    max_steps: int = 100
    learning_rate: float = 3e-3
    batch_size: int = 1
    gradient_clip_value: float = 1.0
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False

    # Linear warmup before the LinearLR decay. 0 disables warmup
    # (scheduler is the bare LinearLR(start=1.0 → end=0) over max_steps).
    # When >0, the scheduler becomes SequentialLR(warmup → decay): LR
    # ramps from learning_rate/1000 up to learning_rate over warmup_steps,
    # then decays linearly to 0 over the remaining (max_steps - warmup_steps).
    # Recommended: 1-5% of max_steps for LoRA fine-tunes on large models;
    # the cold optimizer state plus full learning_rate on step 0 is a
    # known source of early-training NaN bursts.
    warmup_steps: int = 0

    # 0 disables; otherwise log torch.cuda.memory_allocated/reserved
    # every N steps. Used to distinguish a real leak (allocated
    # grows) from caching-allocator fragmentation (reserved grows,
    # allocated flat).
    cuda_memory_log_interval: int = 100

    # HF attn_implementation passed to from_pretrained at training time.
    # "auto" (the default) resolves to "sdpa" — flash-attention is no
    # longer supported. Can be forced to "sdpa" or "eager"; eager is the
    # universal fallback for configs that reject sdpa.
    attn_implementation: str = "auto"

    max_token_block_size: int = 16777216 # 16 mega tokens

    training_mode: str = "language_model"  # or "embedding"

    # Distribution strategy
    distribution_strategy: str = "fsdp"

    # Checkpointing
    steps_per_checkpoint: int = 100
    max_checkpoints_to_keep: int = 3

    gpus: int = 1
    nodes: int = 1

    # Adapters
    adapter_type: str = "tokenformer"
    lora_config: Optional[LoraConfig] = LoraConfig()

    # DiffusionGemma-only canvas denoising config (nested, like lora_config).
    # None for every non-diffusion model; is_diffusion() gates whether it is read.
    diffusion: Optional[DiffusionConfig] = None

    # 4 hours in seconds
    timeout: int = 4 * 60 * 60

    training_history_length: int = 1024

    # Override the global cray-config.yaml `dtype` for this job only.
    # "auto" defers to the global config (which itself defaults to
    # "auto" = the model's native dtype). Other values: "float32",
    # "float16", "bfloat16". Per-job control matters on CPU-on-Apple-
    # Silicon where bf16 matmuls SIGILL under Apple's hypervisor while
    # fp32 paths run fine — operators can pass `dtype: float32` in
    # train_args without changing the running deployment's config.
    dtype: str = "auto"

    # Opt-in per job: allow AutoConfig/AutoTokenizer to execute a repo's
    # custom modeling code (HF `trust_remote_code`). Required to TRAIN
    # models that ship custom code (InternVL3, Molmo, GLM-4/ChatGLM, ...);
    # without it load_model raises "contains custom code which must be
    # executed ... pass trust_remote_code=True" and TRAIN_FAILEDs. MUST be
    # declared here: get_job_config() funnels the raw train_args through
    # this model, and Pydantic DROPS any field not declared — so a
    # `trust_remote_code: true` in train_args silently vanishes without
    # this line. Defaults False so arbitrary remote code only runs when the
    # job explicitly opts in. (vLLM serve passes trust_remote_code on its
    # own, which is why such models serve but wouldn't train.)
    trust_remote_code: bool = False

