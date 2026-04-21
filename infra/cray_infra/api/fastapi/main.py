from cray_infra.api.fastapi.middleware.request_id import (
    request_id_and_log_middleware,
)
from cray_infra.api.fastapi.routers.openai_batches.router import (
    openai_batches_router,
)
from cray_infra.api.fastapi.routers.openai_v1_router import (
    openai_v1_router,
)
from cray_infra.api.fastapi.routers.megatron_router import (
    megatron_router,
)
from cray_infra.api.fastapi.routers.health_router import (
    health_router,
)
from cray_infra.api.fastapi.routers.generate_router import (
    generate_router,
)
from cray_infra.api.fastapi.routers.slurm_router import (
    slurm_router,
)

from cray_infra.api.fastapi.tasks.add_megatron_tasks import (
    add_megatron_tasks,
)
from cray_infra.api.fastapi.routers.add_chat_proxy import (
    add_chat_proxy,
)
from cray_infra.api.fastapi.setup_ui import add_ui

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
import os

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=add_megatron_tasks)

# Initialize observability (OpenTelemetry tracing)
try:
    from cray_infra.observability.init_tracing import init_tracing
    tracer = init_tracing(app, service_name="scalarlm-api")
    logger.info("Observability: OpenTelemetry tracing initialized")
except Exception as e:
    logger.warning(f"Observability: Failed to initialize tracing: {e}")
    tracer = None

app.include_router(openai_v1_router, prefix="/v1")
app.include_router(openai_batches_router, prefix="/v1")
app.include_router(megatron_router, prefix="/v1")
app.include_router(health_router, prefix="/v1")
app.include_router(generate_router, prefix="/v1")
app.include_router(slurm_router)

# Add Prometheus metrics endpoint
@app.get("/v1/metrics")
async def get_prometheus_metrics():
    """Expose Prometheus metrics endpoint."""
    try:
        from cray_infra.observability.prometheus_metrics import metrics_endpoint
        return metrics_endpoint()
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("Metrics unavailable", status_code=500)

add_chat_proxy(app)
add_ui(app)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(request_id_and_log_middleware)
