from infra.cray_infra.one_server.start_cray_server import start_cray_server
from infra.cray_infra.util.get_config import get_config
from infra.cray_infra.one_server.wait_for_vllm import get_vllm_health, wait_for_vllm

import masint
import aiohttp
import unittest
import pytest
import asyncio

import logging
import os

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

TINY_BASE_MODEL = "masint/tiny-random-llama"
TEST_QUESTION = "What is 14 times 14?"
TEST_ANSWER = "14 times 14 is 14 squared."
TEST_PROMPT = f"<|start_header_id|>user<|end_header_id|> {TEST_QUESTION} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"


class TestFineTuning(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        logger.info("Starting server")
        self.app = await start_cray_server(server_list=["api", "vllm"])
        logger.debug(f"Server started: {self.app}")

    async def test_fine_tuning(self):

        # 0. VLLM Health up
        await wait_for_vllm()
        health_status = await get_vllm_health()
        self.assertEqual(health_status, 200)

        llm = masint.AsyncSupermassiveIntelligence()

        # 1. call generate on base model
        base_model_generate_response = await llm.generate(prompts=[TEST_PROMPT])
        logger.debug(
            f"Base model on prompt {TEST_PROMPT} returned {base_model_generate_response}"
        )

        # 2. train a base model with small dataset
        training_response = await llm.train(
            create_training_set(), train_args={"max_steps": 1}
        )

        job_hash = os.path.basename(training_response["job_directory"])
        logger.debug(f"Created a training job: {job_hash}")

        training_status = training_response["status"]
        tuned_model_name = training_response["model_name"]

        # 3. Wait till training is complete
        for _ in range(10):
            training_response = await llm.get_training_job(job_hash)
            training_status = training_response["status"]

            if training_status == "COMPLETED":
                break

            await asyncio.sleep(1)
            logger.info(f"Training job {job_hash} has status {training_status}")

        # 4. Generate response on tuned model
        tuned_model_generate_response = await llm.generate(
            prompt=TEST_PROMPT, model_name=tuned_model_name
        )
        logger.debug(
            f"Trained model on prompt {TEST_PROMPT} returned {tuned_model_generate_response}"
        )

        # 5. Compare and make sure based model and trained model have different responses
        assert base_model_generate_response != tuned_model_generate_response

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()


def create_training_set():
    dataset = []

    count = 10

    for i in range(count):
        dataset.append(
            {
                "input": f"What is {i} times {i}",
                "output": "The answer is " + str(i) + " squared.",
            }
        )

    return dataset
