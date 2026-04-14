"""
Create and configure vllm-mlx server for Apple Silicon.

This module runs vllm-mlx as a subprocess (simpler than in-process integration).
"""

from cray_infra.util.get_config import get_config
from cray_infra.huggingface.get_hf_token import get_hf_token

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


async def create_vllm_mlx(server_status, port: int):
    """
    Create and start vllm-mlx server as a subprocess.

    Args:
        server_status: Server status object for communication
        port: Port to run the server on (typically 8001)
    """
    logger.info(f"Starting vllm-mlx server on port {port}")

    config = get_config()

    # Set environment
    env = os.environ.copy()
    env["HF_TOKEN"] = get_hf_token()
    env["SCALARLM_JOBS_DIR"] = config.get("jobs_dir", "/app/cray/jobs")

    model_name = config["model"]
    max_tokens = config.get("max_model_length", 2048)

    logger.info(f"Starting vllm-mlx subprocess: model={model_name}, port={port}")

    # Run vllm-mlx server as subprocess
    cmd = [
        "python", "-m", "vllm_mlx.server",
        "--model", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-tokens", str(max_tokens),
    ]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    logger.info(f"✅ vllm-mlx server started (PID: {process.pid})")

    # Stream output in background (don't block FastAPI startup)
    async def stream_output():
        try:
            # Read output line by line without blocking
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                )
                if not line:
                    break
                logger.info(f"[vllm-mlx] {line.rstrip()}")
        except Exception as e:
            logger.error(f"Error reading vllm-mlx output: {e}")
        finally:
            process.wait()
            logger.info(f"vllm-mlx server stopped (exit code: {process.returncode})")

    # Start background task to stream output
    asyncio.create_task(stream_output())

    # Wait briefly for server to start, then return
    await asyncio.sleep(2)
    logger.info("vllm-mlx server startup initiated, continuing with FastAPI startup...")
