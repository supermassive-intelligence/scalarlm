from cray_infra.one_server.start_cray_server import start_cray_server

import masint
import os
import unittest
import asyncio

import logging

logger = logging.getLogger(__name__)


class TestMegatronTrainer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        logger.info("Starting server")

        self.app = await start_cray_server(server_list=["api"])

        logger.debug(f"Server started: {self.app}")

    async def test_upload_dataset(self):
        logger.debug("Testing upload ability of train endpoint")

        llm = masint.AsyncSupermassiveIntelligence()

        dataset = get_dataset()

        status = await llm.train(dataset, train_args={"max_steps": 1})

    async def test_get_training_job_info(self):
        logger.debug("Testing the get training job info endpoint")
        llm = masint.AsyncSupermassiveIntelligence()

        logger.debug("Creating a training job")
        dataset = get_dataset()

        training_response = await llm.train(dataset, train_args={"max_steps": 1})
        job_hash = os.path.basename(training_response["job_directory"])
        training_status = training_response.get("status")

        for _ in range(10):
            training_response = await llm.get_training_job(job_hash)
            print(f"\n\n****\n\n {training_response} *****\n\n")
            job_status = training_response.get("job_status")
            job_config = training_response.get("job_config")
            training_status = job_status.get("status")

            if training_status == "COMPLETED":
                break

            await asyncio.sleep(1)
            logger.info(f"Training job {job_hash} has status {training_status}")

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()


def get_dataset():
    dataset = []

    count = 10000

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}", "output": "The answer is " + str(i + i)}
        )

    return dataset
