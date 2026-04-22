from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    api_url: str = "http://localhost:8000"

    model: str = "tiny-random/gemma-4-dense"
    #model: str = "google/gemma-3-270m-it"
    #model: str = "yujiepan/qwen3-moe-tiny-random"
    #model: str = "masint/tiny-random-llama"
    #model: str = "masint/tiny-random-qwen2-vl"
    #model: str = "Snowflake/Arctic-Text2SQL-R1-7B"
    #model: str = "Qwen/Qwen2-7B-Instruct"
    #model: str = "Qwen/Qwen2-VL-7B-Instruct"


    # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    train_job_entrypoint: str = "/app/cray/scripts/train_job_entrypoint.sh"
    training_job_directory: str = "/app/cray/jobs"

    log_directory: str = "/app/cray/nfs/logs"

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

    # Bound on concurrent calls the OpenAI-compatible proxy forwards to vLLM.
    # A semaphore keeps at most `openai_queue_concurrency` requests in flight
    # at once; extras wait. Overflow beyond `openai_queue_max_depth` gets
    # 503 + Retry-After rather than queued indefinitely.
    openai_queue_concurrency: int = 16
    openai_queue_max_depth: int = 256
    openai_queue_retry_after_seconds: int = 1

    # When true, exposes /v1/bench/nop — a no-op endpoint used by the
    # benchmark harness to isolate FastAPI/uvicorn ceiling from the
    # inference layer. Left off in production; flip via config or env.
    bench_endpoints_enabled: bool = False

    # Phase 6. When true (default), the openai proxy calls vLLM's
    # OpenAIServingCompletion / OpenAIServingChat Python APIs in-process
    # via the vllm_registry, skipping the localhost-HTTP hop to vLLM's
    # own FastAPI server on vllm_api_url. Fall back to the HTTP proxy by
    # setting this to false — useful when the scalarlm server talks to
    # a vLLM in a separate process / host.
    openai_inprocess_enabled: bool = True

    # Phase 7. Concurrency bound for the Batch API runner — how many
    # input-JSONL lines we dispatch in parallel via asyncio.gather. The
    # pre-Phase-7 behaviour was sequential per-line awaits (effective
    # concurrency 1); raising this lifts the Batch API from the flat
    # 1.9–3.2 p/s curve toward the array-completions throughput. Bounded
    # so a 1000-line batch can't saturate the proxy's openai_queue_concurrency
    # by itself — set this no larger than openai_queue_concurrency.
    batch_runner_concurrency: int = 16

    inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"
    upload_base_path: str = "/app/cray/inference_requests"

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

