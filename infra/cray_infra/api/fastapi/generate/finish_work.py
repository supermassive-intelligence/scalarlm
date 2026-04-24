from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.api.work_queue.get_unfinished_result import get_unfinished_result
from cray_infra.api.work_queue.update_and_ack import update_and_ack

from cray_infra.api.fastapi.routers.request_types.finish_work_request import (
    FinishWorkRequests,
)
from cray_infra.generate.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)


async def finish_work(requests: FinishWorkRequests):
    inference_work_queue = await get_inference_work_queue()

    for request in requests.requests:
        logger.debug(f"Finishing work for request {request.request_id}")

        result = await get_unfinished_result(request_id=request.request_id)

        # The group may have been wiped out from in-memory state:
        # uvicorn auto-reload between submit and finish_work, or the
        # group was already finalized/cleared by a sibling request's
        # finish_work call. Drop this completion cleanly — if the
        # client was still waiting, their poll will either succeed
        # against the already-written response file or time out. We
        # must return 200 here so the worker doesn't retry and
        # amplify load.
        if result is None:
            logger.warning(
                "finish_work: no in-memory group state for %s "
                "(likely process restart or already-finalized group); "
                "dropping result.",
                request.request_id,
            )
            continue

        if request.response is not None:
            result["response"] = request.response

        if request.error is not None:
            result["error"] = request.error

        await update_and_ack(inference_work_queue, request_id=request.request_id, item=result)

        metrics = get_metrics()

        metrics.record_completed_request(
            token_count=request.token_count, flop_count=request.flop_count
        )
