from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.generate.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)


async def metrics():
    """
    Get the current metrics.

    queue_depth is reported as len(SQLiteAckQueue) + streaming_inflight:
      - SDK path (`/v1/generate`) is queued through SQLite, so its
        ground truth is `len(queue)`.
      - OpenAI-streaming path (`/v1/chat/completions`) proxies straight
        to vLLM and never touches SQLite; we count those via the
        in-memory `streaming_inflight` counter (try/finally-balanced
        so it doesn't drift).

    The in-memory `Metrics.queue_depth` running counter is still used
    internally to gate throughput math (epoch_time) and to feed
    instantaneous rates, but it's no longer the surface number — it
    drifts up on any error path that skips finish_work.
    """
    sdk_depth = None
    try:
        queue = await get_inference_work_queue()
        sdk_depth = len(queue)
    except ValueError as e:
        # persistqueue raises ValueError via __len__ when its internal
        # counter has drifted negative (get_work_item.py now ack's the
        # duplicates that caused this). Report 0 rather than the drifted
        # in-memory counter so operators don't chase a ghost.
        logger.debug("Queue len() drifted negative, reporting 0: %s", e)
        sdk_depth = 0
    except Exception as e:
        logger.debug("Falling back to in-memory queue_depth: %s", e)
    if sdk_depth is not None and sdk_depth < 0:
        sdk_depth = 0
    return get_metrics().get_all_metrics(sdk_queue_depth=sdk_depth)
