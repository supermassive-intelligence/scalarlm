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
    publish_job_entrypoint: str = "/app/cray/scripts/publish_job_entrypoint.sh"
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

    # Hard cap on requests submitted to vLLM but not yet finished.
    # The generate worker refuses to pull more work from the SQLiteAckQueue
    # until the in-flight count drops below this threshold. vLLM's own
    # scheduler doesn't apply admission backpressure, so without this cap
    # the worker's get_batch_size heuristic over-counts available capacity
    # and floods vLLM's waiting queue. Defaults to generate_batch_size so
    # existing deployments keep the same effective ceiling.
    max_inflight_requests: int = 1024

    response_timeout: int = 60 # seconds
    inference_work_queue_timeout: int = 30 # seconds
    inference_work_queue_idle_time: int = 5 # seconds
    # The restart watchdog nacks any request that's been unacked longer
    # than this back into the pending queue. Was 300s, which was shorter
    # than p99 for long generations on saturated GPUs — the watchdog
    # then re-submitted to vLLM while the original was still running,
    # compounding load. Raised to 30 min; operators with fast inference
    # can lower it in cray-config.yaml.
    inference_work_queue_ack_timeout: int = 1800 # seconds

    inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"
    upload_base_path: str = "/app/cray/inference_requests"

    gpu_memory_utilization: float = 0.40
    max_model_length: int = 256
    default_max_output_tokens: int = 128
    dtype: str = "auto"
    limit_mm_per_prompt:str = '{"image":2}'

    # Whether to pass --enable-lora to vLLM on startup. When true (default),
    # every layer in the model is wrapped in a LoRA-aware shim so adapters
    # can be hot-loaded. That wrapping has cost on every forward pass even
    # with no adapter loaded, and on some vLLM-fork builds the MoE LoRA
    # wrapper has outright bugs (e.g. the TorchAllocator.set interface drift
    # on scalarlm-on-v0.19.0 HEAD). Deployments that know they won't use
    # LoRA/tokenformer adapters should set SCALARLM_ENABLE_LORA=false to
    # skip the wrapping entirely. Dynamic adapter loading via
    # /v1/load_lora_adapter won't be available in that mode — fine for
    # pure-inference pods, not fine for training-eval pods.
    enable_lora: bool = True

    max_log_length: int = 100

    server_list: str = "all"

    tokenformer_r: int = 32
    tokenformer_num_heads: int = 4

    tokenformer_cache_capacity: int = 2

    hf_token: str = ""

    hf_encrypted_token: bytes = b"gAAAAABpyvSQu2QUlUfp-YavLwueXqCU0j2Lhe9Lddij4B-qV3JngfcH4uCtjVGXlWAyM2o91nZXhsS3B3q3zKNiLxnxhFpJd0ddbwWPysez2OpZX4jTFOA9-xjQVk454A_qk6pdJxMv"
    encryption_key: bytes = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

    # ------------------------------------------------------------------
    # OpenAI chat-completions queue (docs/openai-chat-completions-queue.md)
    # ------------------------------------------------------------------

    # Mirrors vLLM's --max-num-seqs. Chat admission threshold is a
    # multiple of this. Kept in scalarlm config so admission decisions
    # don't round-trip to vLLM at request time.
    max_num_seqs: int = 256

    # Admission high-water mark = chat_admit_factor × max_num_seqs.
    # Beyond this, requests get HTTP 429 + Retry-After. See §5.
    chat_admit_factor: int = 4

    # Coalescer (§6) — packs admitted requests into one queue row to
    # amortize SQLite write cost. packing_factor is the primary
    # throughput knob (§6.3): start here, raise if 429 rate climbs.
    chat_coalescer_packing_factor: int = 10
    chat_coalescer_window_ms: int = 50
    chat_coalescer_bypass_threshold: int = 10

    # Whitespace heartbeat interval (§9). 4 s is below httpx's default
    # 5 s read timeout, leaving margin for slow clocks / event-loop
    # jitter without the caller seeing an idle gap.
    chat_heartbeat_interval_seconds: float = 4.0

    # Optional cap on per-request total time. Unset by default — the
    # queue's existing ack-timeout (inference-queue.md §5.2) provides
    # a backstop for stuck batches.
    chat_max_request_seconds: Optional[float] = None

    # WaitEstimator (§5). Used for the Retry-After hint on 429.
    chat_wait_estimator_default_seconds: float = 5.0
    chat_wait_estimator_padding: float = 1.5
    chat_wait_estimator_sample_size: int = 32

    # Apparent-buffering heuristic (§13.2). Flags requests whose
    # duration falls within `match_threshold_seconds` of a known
    # proxy idle timeout — strong signal that an upstream proxy is
    # buffering responses despite our heartbeats.
    chat_buffering_check_proxy_timeout_seconds: float = 60.0
    chat_buffering_match_threshold_seconds: float = 0.5

