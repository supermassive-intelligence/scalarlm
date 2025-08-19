# Set vLLM environment variables BEFORE any vLLM imports
import os
import torch

# Set device target before vLLM imports for proper device inference
if not torch.cuda.is_available():
    print("No CUDA available, forcing CPU platform")
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Enable debug logging as suggested by error
    # Set additional vLLM CPU environment variables
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_MODELSCOPE"] = "False"
    # Remove CUDA_VISIBLE_DEVICES for CPU mode to avoid device conflicts
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
else:
    print(f"CUDA available with {torch.cuda.device_count()} GPU(s), using GPU platform")

from cray_infra.util.get_config import get_config
from cray_infra.huggingface.get_hf_token import get_hf_token

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

import uvicorn
import logging

logger = logging.getLogger(__name__)

async def create_vllm(port):
    
    print(f"DEBUG: BEFORE CONFIG - Environment variables:")
    print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    config = get_config()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = [
        f"--dtype={config['dtype']}",
        f"--max-model-len={config['max_model_length']}",
        f"--max-num-batched-tokens={config['max_model_length']}",
        f"--max-seq-len-to-capture={config['max_model_length']}",
        f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
        f"--max-log-len={config['max_log_length']}",
        f"--swap-space=0",
        "--enable-lora"
    ]

    # Handle multimodal limits (restored from original)
    if config['limit_mm_per_prompt'] is not None:
        args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")

        
    # CPU backend only supports V1 scheduler
    if not torch.cuda.is_available():
        os.environ["VLLM_USE_V1"] = "1"
        logger.info("Setting VLLM_USE_V1=1 for CPU backend")
        # V1 doesn't support --disable-async-output-proc
    else:
        # Only add this for GPU mode
        args.append("--disable-async-output-proc")

    # Device is automatically detected by platform detection now
    # No need to explicitly set --device argument

    print(f"DEBUG: About to parse args: {args}")
    print(f"DEBUG: Environment variables:")
    print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    args = parser.parse_args(args=args)

    args.port = port
    args.model = config["model"]

    logger.info(f"Running vLLM with args: {args}")

    await run_server(args)
