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

    # 0 disables; otherwise log torch.cuda.memory_allocated/reserved
    # every N steps. Used to distinguish a real leak (allocated
    # grows) from caching-allocator fragmentation (reserved grows,
    # allocated flat).
    cuda_memory_log_interval: int = 100

    # "auto" runs pick_attention_backend; can be forced to one of
    # "flash_attention_2" / "flash_attention_3" / "sdpa" / "eager"
    # when a model breaks auto-detection (e.g. Gemma-4 has per-layer
    # head_dim variation that config inspection underreports).
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

