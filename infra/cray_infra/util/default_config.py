import os
from pydantic import BaseModel
from typing import Optional

# Determine the root directory based on SCALARLM_NATIVE_EXECUTION environment variable
# Docker: /app/cray (container mount point)
# Native: Calculate from this file's location in the repo
if os.environ.get("SCALARLM_NATIVE_EXECUTION") == "true":
    # Native execution - calculate paths from this file's location
    # The repo structure is: {repo_root}/infra/cray_infra/util/default_config.py
    _current_file = os.path.abspath(__file__)
    _cray_infra_dir = os.path.dirname(os.path.dirname(_current_file))  # .../infra/cray_infra
    _INFRA_DIR = os.path.dirname(_cray_infra_dir)  # .../infra
    _ROOT = os.path.dirname(_INFRA_DIR)  # .../scalarlm (repo root)
else:
    # Docker environment - use standard Docker paths
    _ROOT = "/app/cray"
    _INFRA_DIR = "/app/cray/infra"

class Config(BaseModel):
    # Base directories - auto-detected based on environment
    base_dir: str = _ROOT  # Root directory for all ScalarLM data
    infra_dir: str = _INFRA_DIR  # Infrastructure code directory (for autoreload)
    api_url: str = "http://localhost:8000"

    # MLX Model:
    # model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    model: str = "tiny-random/gemma-4-dense"  # Fast for testing
    #model: str = "google/gemma-3-270m-it"
    #model: str = "yujiepan/qwen3-moe-tiny-random"
    #model: str = "masint/tiny-random-llama"
    #model: str = "masint/tiny-random-qwen2-vl"
    #model: str = "Snowflake/Arctic-Text2SQL-R1-7B"
    #model: str = "Qwen/Qwen2-7B-Instruct"
    #model: str = "Qwen/Qwen2-VL-7B-Instruct"

    native_execution: bool = False  # Running natively without Docker (e.g., MLX on Apple Silicon)
    inference_only: bool = False  # Skip training background tasks


    # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    train_job_entrypoint: str = os.path.join(_ROOT, "scripts", "train_job_entrypoint.sh")
    training_job_directory: str = os.path.join(_ROOT, "jobs")

    log_directory: str = os.path.join(_ROOT, "nfs", "logs")

    max_gpus_per_node: int = 1
    max_train_time: int = 24 * 60 * 60
    extra_training_seconds: int = 300  # 5 minutes buffer before slurm kills the job
    tensor_parallel_size: int = 1

    slurm_wait_time: int = 30 # seconds
    node_info_time_limit: int = 3600 # seconds

    megatron_refresh_period: int = 30 # seconds

    vllm_api_url: str = "http://localhost:8001"

    # vLLM Engine Configuration
    generate_batch_size: int = 1024

    response_timeout: int = 60 # seconds
    inference_work_queue_timeout: int = 30 # seconds
    inference_work_queue_idle_time: int = 5 # seconds
    inference_work_queue_ack_timeout: int = 300 # seconds

    inference_work_queue_path: str = os.path.join(_ROOT, "inference_work_queue.sqlite")
    upload_base_path: str = os.path.join(_ROOT, "inference_requests")

    gpu_memory_utilization: float = 0.40
    max_model_length: int = 256
    default_max_output_tokens: int = 128
    dtype: str = "auto"
    limit_mm_per_prompt:str = '{"image":2}'

    max_log_length: int = 100

    server_list: str = "all"

    tokenformer_r: int = 32
    tokenformer_num_heads: int = 4

    tokenformer_cache_capacity: int = 2

    hf_token: str = ""

    hf_encrypted_token: bytes = b"gAAAAABpyvSQu2QUlUfp-YavLwueXqCU0j2Lhe9Lddij4B-qV3JngfcH4uCtjVGXlWAyM2o91nZXhsS3B3q3zKNiLxnxhFpJd0ddbwWPysez2OpZX4jTFOA9-xjQVk454A_qk6pdJxMv"
    encryption_key: bytes = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

    # MLX-specific configuration (Apple Silicon)
    mlx_backend: bool = False  # Auto-detected based on platform, can be explicitly set
    mlx_dtype: str = "float32"  # Options: float32, float16, bfloat16
    mlx_quantization: str = "4bit"  # Options: 4bit, 8bit, or None
    unified_memory_fraction: float = 0.8  # Fraction of unified memory to use
    jobs_dir: str = os.path.join(_ROOT, "jobs")  # Directory containing training job checkpoints

