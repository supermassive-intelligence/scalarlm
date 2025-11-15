from cray_infra.api.fastapi.health.check_health import check_health
from cray_infra.api.fastapi.health.service_logs_generator import service_logs_generator

from fastapi import APIRouter

from fastapi.responses import JSONResponse

from starlette.responses import StreamingResponse

import traceback

import logging

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health")


@health_router.get("")
async def health():
    return await check_health()


@health_router.get("/keepalive")
async def health():
    return {"status": "ok"}


@health_router.get("/logs/{service_name}")
async def get_service_logs(service_name: str, starting_line_number: int = 0):
    try:
        return StreamingResponse(
            content=service_logs_generator(service_name, starting_line_number),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(e)
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@health_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in health_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
