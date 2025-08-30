from cray_infra.training.squeue import squeue
from cray_infra.training.get_gpu_count import get_gpu_count
from cray_infra.training.get_node_count import get_node_count

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import logging

logger = logging.getLogger(__name__)

slurm_router = APIRouter(prefix="/slurm")


@slurm_router.get("/status")
async def slurm_status():
    """Get SLURM cluster status including queue information and resource counts."""
    try:
        squeue_info = await squeue()
        gpu_count = get_gpu_count()
        node_count = get_node_count()
        
        return {
            "queue": squeue_info.dict(),
            "resources": {
                "gpu_count": gpu_count,
                "node_count": node_count
            },
            "status": "active"
        }
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e)
            },
            status_code=500
        )


@slurm_router.get("/squeue")
async def get_squeue():
    """Get SLURM queue information."""
    return await squeue()


@slurm_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in slurm_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")