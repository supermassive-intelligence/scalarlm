from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse,
)

from cray_infra.training.launch_training_job import launch_training_job
from cray_infra.training.upload_training_data import upload_training_data
from cray_infra.training.training_logs_generator import training_logs_generator
from cray_infra.training.training_logs_generator import get_latest_model

from cray_infra.util.get_config import get_config

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse

import yaml
import os
import traceback
import json

import logging

logger = logging.getLogger(__name__)

megatron_router = APIRouter(prefix="/megatron")


@megatron_router.post("/train")
async def train(request: Request):
    logger.info(f"Training request received: {request}")
    training_data_path, params = await upload_training_data(request)

    try:
        job_config = params

        logger.info(f"Training args: {job_config}")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid request body",
        )

    job_status = await launch_training_job(job_config)

    return TrainResponse(job_status=job_status, job_config=job_config)


@megatron_router.get("/train/{job_hash}")
async def get_training_job_info(job_hash: str):
    try:
        if job_hash == "latest":
            job_hash = get_latest_model()

        job_directory_path = get_job_directory_for_hash(job_hash)
        status_filepath = os.path.join(job_directory_path, "status.json")
        # Get job status
        try:
            with open(status_filepath, "r") as file:
                job_status = json.loads(file.readline().strip())
        except FileNotFoundError:
            logger.error("File not found")
        except json.JSONDecodeError:
            logger.error("Invalid JSON in first line")

        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job was not found at {job_directory_path}",
            )
        # Get job config
        config_filepath = os.path.join(job_directory_path, "config.yaml")
        try:
            with open(config_filepath, "r") as file:
                job_config = yaml.safe_load(file)

        except FileNotFoundError:
            logger.error(f"{config_filepath} file not found")
        except json.JSONDecodeError:
            logger.error("Invalid YAML")

        if not job_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job config was not found at {job_directory_path}",
            )

        return TrainResponse(job_status=job_status, job_config=job_config)
    except Exception as e:
        logger.exception(
            f"Error retrieving training job {job_hash} "
            "Exception: {type(e).__name__}, Message: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve training job information: {str(e)}",
        )


def get_job_directory_for_hash(hash_id: str):
    config = get_config()
    return os.path.join(config["training_job_directory"], hash_id)


@megatron_router.get("/train/logs/{model_name}")
async def get_training_logs(model_name: str, starting_line_number: int = 0):
    try:
        return StreamingResponse(
            content=training_logs_generator(model_name, starting_line_number),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(e)
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)
