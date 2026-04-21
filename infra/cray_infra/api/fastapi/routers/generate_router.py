from cray_infra.api.fastapi.generate.deprecation import log_path_a_deprecation
from cray_infra.api.fastapi.generate.get_work import get_work
from cray_infra.api.fastapi.generate.get_adaptors import get_adaptors
from cray_infra.api.fastapi.generate.generate import generate
from cray_infra.api.fastapi.generate.finish_work import finish_work
from cray_infra.api.fastapi.generate.get_results import get_results
from cray_infra.api.fastapi.generate.metrics import metrics
from cray_infra.api.fastapi.generate.upload import upload
from cray_infra.api.fastapi.generate.download import download
from cray_infra.api.fastapi.generate.clear_queue import clear_queue

from cray_infra.api.fastapi.routers.request_types.generate_request import (
    GenerateRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_work_request import GetWorkRequest
from cray_infra.api.fastapi.routers.request_types.finish_work_request import (
    FinishWorkRequests,
)
from cray_infra.api.fastapi.routers.request_types.get_results_request import (
    GetResultsRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_adaptors_request import (
    GetAdaptorsRequest,
)
from cray_infra.api.fastapi.routers.request_types.download_request import DownloadRequest

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

generate_router = APIRouter(prefix="/generate")


# ``log_path_a_deprecation`` is attached to client-facing endpoints only
# (generate, get_results, upload, download). Worker-facing endpoints
# (get_work, finish_work, get_adaptors, clear_queue, metrics) stay as-is
# because they're ScalarLM-internal and outlive the Path A surface.
_DEPRECATION = [Depends(log_path_a_deprecation)]


@generate_router.post("", dependencies=_DEPRECATION)
async def generate_endpoint(request: GenerateRequest):
    return await generate(request)

@generate_router.post("/get_results", dependencies=_DEPRECATION)
async def get_results_endpoint(request: GetResultsRequest):
    return await get_results(request)

@generate_router.post("/get_work")
async def get_work_endpoint(request: GetWorkRequest):
    return await get_work(request)

@generate_router.post("/finish_work")
async def finish_work_endpoint(requests: FinishWorkRequests):
    return await finish_work(requests)

@generate_router.post("/upload", dependencies=_DEPRECATION)
async def upload_endpoint(request: Request):
    return await upload(request)

@generate_router.post("/download", dependencies=_DEPRECATION)
async def download_endpoint(request: DownloadRequest):
    return await download(request)

@generate_router.post("/clear_queue")
async def clear_queue_endpoint():
    return await clear_queue()


@generate_router.post("/get_adaptors")
async def get_adaptors_endpoint(request: GetAdaptorsRequest):
    return await get_adaptors(request)


@generate_router.get("/metrics")
async def metrics_endpoint():
    return await metrics()



@generate_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in generate_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
