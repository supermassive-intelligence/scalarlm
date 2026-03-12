from pydantic import BaseModel

from typing import Optional


class LoraConfig(BaseModel):
    r: int = 32
    target_modules: str = "all-linear"
    use_rslora: bool = True
    modules_to_save: list = ["lm_head"]

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

