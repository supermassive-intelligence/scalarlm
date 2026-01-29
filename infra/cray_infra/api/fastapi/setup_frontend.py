from cray_infra.one_server.wait_for_vllm import get_vllm_health

import asyncio
import logging

logger = logging.getLogger(__name__)
frontend_task = None
frontend_process = None


async def monitor_frontend_process():
    """Monitor the frontend process and restart it if it fails."""
    global frontend_process

    frontend_entrypoint_path = "/app/ui/entrypoint.sh"

    while True:
        try:
            logger.info("Starting frontend process...")
            frontend_process = await asyncio.create_subprocess_shell(
                f"bash {frontend_entrypoint_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for the process to complete
            stdout, stderr = await frontend_process.communicate()

            if frontend_process.returncode == 0:
                logger.info(f"Frontend process exited successfully:\n{stdout.decode()}")
                break  # Don't restart on successful exit
            else:
                logger.error(
                    f"Frontend process failed with return code {frontend_process.returncode}:\n{stderr.decode()}"
                )
                logger.info("Attempting to restart frontend...")
                await asyncio.sleep(2)  # Brief delay before restart

        except Exception as e:
            logger.exception(f"An error occurred while running the frontend: {e}")
            logger.info("Attempting to restart frontend...")
            await asyncio.sleep(2)  # Brief delay before restart
        finally:
            frontend_process = None


async def setup_frontend():
    """Start the frontend process if it's not already running."""
    global frontend_task

    if frontend_task is not None and not frontend_task.done():
        logger.info("Frontend is already running. Skipping setup.")
        return

    it await get_vllm_health() != 200:
        logger.info("vLLM is not healthy. Waiting before starting frontend...")
        return

    logger.info("Starting frontend setup...")
    frontend_task = asyncio.create_task(monitor_frontend_process())
