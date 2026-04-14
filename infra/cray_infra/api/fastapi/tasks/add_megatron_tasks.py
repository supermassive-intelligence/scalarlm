from cray_infra.util.get_config import get_config

from cray_infra.training.restart_megatron_jobs import restart_megatron_jobs
from cray_infra.training.register_megatron_models import register_megatron_models
from cray_infra.training.register_megatron_workers import register_megatron_workers
from cray_infra.generate.clear_acked_requests_from_queue import clear_acked_requests_from_queue

from cray_infra.api.fastapi.setup_frontend import setup_frontend

from fastapi_utils.tasks import repeat_every

from contextlib import asynccontextmanager

import traceback
import sys
import logging
import shutil

logger = logging.getLogger(__name__)


def is_slurm_available() -> bool:
    """Check if SLURM is available on the system."""
    return shutil.which("squeue") is not None


@asynccontextmanager
async def add_megatron_tasks(app):
    config = get_config()

    megatron_refresh_period = config["megatron_refresh_period"]

    # Check if SLURM is available - skip training tasks if not
    slurm_available = is_slurm_available()
    if not slurm_available:
        logger.info("SLURM not available - skipping training-related background tasks")
        logger.info("Running in inference-only mode (MLX or non-SLURM environment)")

    @repeat_every(seconds=megatron_refresh_period)
    async def run_megatron_tasks():
        try:
            # Only run SLURM-dependent tasks if SLURM is available
            if slurm_available:
                await register_megatron_models()
                await restart_megatron_jobs()
                await register_megatron_workers()

            # Always run queue cleanup and frontend setup
            await clear_acked_requests_from_queue()
            await setup_frontend()
        except Exception as e:
            print_exception()
            raise e

    await run_megatron_tasks()

    yield


def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    messages = traceback.format_exception(exc_type, exc_value, exc_traceback)

    logger.error("".join(messages))
