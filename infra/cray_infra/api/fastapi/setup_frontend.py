import asyncio

import logging

logger = logging.getLogger(__name__)

frontend_is_running = False

async def setup_frontend():
    global frontend_is_running

    if frontend_is_running:
        logger.info("Frontend is already running. Skipping setup.")
        return

    logger.info("Starting frontend setup...")

    frontend_entrypoint_path = "/app/ui/entrypoint.sh"

    # Run the entrypoint script as a subprocess, handleing any exceptions
    try:
        process = await asyncio.create_subprocess_shell(
            f"bash {frontend_entrypoint_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        frontend_is_running = True

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"Frontend setup completed successfully:\n{stdout.decode()}")
        else:
            logger.error(f"Frontend setup failed with return code {process.returncode}:\n{stderr.decode()}")

    except Exception as e:
        logger.exception(f"An error occurred while setting up the frontend: {e}")
    finally:
        frontend_is_running = False

