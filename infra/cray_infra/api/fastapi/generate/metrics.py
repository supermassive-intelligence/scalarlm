from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.generate.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)


async def metrics():
    """
    Get the current metrics.

    `queue_depth` is pulled from the persistent SQLiteAckQueue at read time
    rather than the in-memory running counter in Metrics. The counter is
    still used internally to gate throughput math ("time while non-empty"
    via Metrics.epoch_time), but it's a running estimate that drifts up
    every time a request is submitted but never finished (error paths
    that skip finish_work, uvicorn reloads, crashed inference workers).
    The queue length is ground truth.
    """
    real_depth = None
    try:
        queue = await get_inference_work_queue()
        real_depth = len(queue)
    except ValueError as e:
        # persistqueue raises ValueError via __len__ when its internal
        # counter has drifted negative (get_work_item.py now ack's the
        # duplicates that caused this). Report 0 rather than the drifted
        # in-memory counter so operators don't chase a ghost.
        logger.debug("Queue len() drifted negative, reporting 0: %s", e)
        real_depth = 0
    except Exception as e:
        logger.debug("Falling back to in-memory queue_depth: %s", e)
    if real_depth is not None and real_depth < 0:
        real_depth = 0
    return get_metrics().get_all_metrics(queue_depth_override=real_depth)
