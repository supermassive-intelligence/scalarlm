from cray_infra.one_server.create_api import create_api
from cray_infra.one_server.create_vllm import create_vllm
from cray_infra.one_server.create_megatron import create_megatron
from cray_infra.one_server.create_embedding_worker import create_embedding_worker
from cray_infra.one_server.create_embedding_service import create_embedding_service

import asyncio
import logging

logger = logging.getLogger(__name__)


async def start_cray_server(server_list: list):

    running_status = ServerStatus()

    logger.debug(f"Starting servers: {server_list}")

    started_any_server = False

    if ("api" in server_list) or ("all" in server_list):
        logger.debug("Starting API server")
        api_task = asyncio.create_task(
            create_api(port=8000, running_status=running_status)
        )
        running_status.tasks.append(api_task)
        started_any_server = True

    if ("vllm" in server_list) or ("all" in server_list):
        logger.debug("Starting VLLM server")
        vllm_task = asyncio.create_task(
            create_vllm(port=8001)
        )
        running_status.tasks.append(vllm_task)
        started_any_server = True
        
        # Start the separate embedding service
        logger.debug("Starting Embedding Service")
        embedding_service_process = await create_embedding_service()
        if embedding_service_process:
            logger.info("✓ Embedding service started successfully")
        else:
            logger.warning("⚠ Embedding service failed to start - embeddings will not work")
        
        # Also start the embedding worker when vLLM is started
        logger.debug("Starting Embedding Worker")
        worker_task = asyncio.create_task(
            create_embedding_worker(running_status=running_status)
        )
        running_status.tasks.append(worker_task)
        logger.info("✓ Embedding worker started to process queue requests")

    if "megatron" in server_list:
        logger.debug("Megatron server doesn't need python")
        megatron_task = asyncio.create_task(
            create_megatron(running_status=running_status)
        )
        running_status.tasks.append(megatron_task)
        started_any_server = True

    if not started_any_server:
        logger.error(
            "No valid server type provided. Please specify 'api', 'vllm', 'megatron', or 'all'."
        )

    return running_status


class ServerStatus:
    def __init__(self):
        self.servers = []
        self.tasks = []

    async def shutdown(self):
        for task in self.tasks:
            logger.debug(f"Task {task} is cancelled")
            task.cancel()

        for server in self.servers:
            logger.debug(f"Server {server} is cancelled")
            await server.shutdown()
