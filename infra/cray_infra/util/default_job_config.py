from pydantic import BaseModel

from typing import Optional, Union


class LoraConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Union[str, list] = "all-linear"  # or list of module names

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

