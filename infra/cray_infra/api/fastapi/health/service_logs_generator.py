from cray_infra.util.get_config import get_config

import aiofiles
import os
import json

import logging

logger = logging.getLogger(__name__)


def service_logs_generator(service_name: str, starting_line_number: int):
    service_log_file = get_service_log_file(service_name)

    logger.info(f"Using log file: {service_log_file}")

    async def generate():
        line_number = 0
        async with aiofiles.open(service_log_file, mode="r") as f:
            async for line in f:
                if line_number < starting_line_number:
                    line_number += 1
                    continue

                yield json.dumps({"line": line.rstrip(), "line_number": line_number}) + "\n"
                line_number += 1

    return generate()


def get_service_log_file(service_name: str) -> str:
    config = get_config()
    log_base_dir = config["log_directory"]

    service_log_file = os.path.join(log_base_dir, f"{service_name}.log")

    if not os.path.isfile(service_log_file):
        raise FileNotFoundError(f"Log file for service '{service_name}' not found.")

    return service_log_file
