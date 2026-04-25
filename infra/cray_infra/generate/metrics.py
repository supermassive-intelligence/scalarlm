
import time
from collections import deque

generate_metrics = None

# Rolling-window length for instantaneous rate metrics. token/s,
# request/s, and flop/s are reported as "completions in the last
# RATE_WINDOW_SECONDS / RATE_WINDOW_SECONDS" rather than a cumulative
# average over the lifetime of the process. The latter barely changed
# between polls and rendered a flat sparkline.
RATE_WINDOW_SECONDS = 60.0


class Metrics:
    def __init__(self):
        self.queue_depth = 0
        # Concurrent OpenAI-streaming requests in flight. Driven by
        # _wrap_with_metrics in openai_v1_router.py; that path doesn't
        # touch the SQLiteAckQueue, so it has no other ground truth.
        # try/finally ensures every increment matches a decrement, so
        # this counter doesn't drift the way Metrics.queue_depth can.
        self.streaming_inflight = 0
        self.epoch_time = None

        self.total_completed_requests = 0
        self.total_completed_tokens = 0
        self.total_completed_flops = 0

        self.total_completed_response_time = 0.0

        # Per-completion window: (timestamp, tokens, flops). Pruned
        # to the last RATE_WINDOW_SECONDS in get_all_metrics.
        self._rate_window: deque = deque()

    def record_completed_request(self, token_count: int, flop_count: int):
        """
        Record a completed request.
        """
        # Always feed the rate window — it doesn't depend on the
        # legacy queue_depth bookkeeping.
        now = time.time()
        self._rate_window.append(
            (now, int(token_count or 0), int(flop_count or 0))
        )

        if self.queue_depth <= 0:
            self.queue_depth = 0
            return

        self.total_completed_requests += 1

        if flop_count is not None:
            self.total_completed_flops += flop_count

        if token_count is not None:
            self.total_completed_tokens += token_count

        self.queue_depth -= 1

        time_since_epoch = now - self.epoch_time

        self.total_completed_response_time += time_since_epoch

        self.epoch_time = now

    def record_new_request(self):
        """
        Record a new request.
        """
        if self.queue_depth == 0:
            self.epoch_time = time.time()

        self.queue_depth += 1

    def record_streaming_start(self):
        """OpenAI-streaming path admission; increments streaming_inflight."""
        self.streaming_inflight += 1

    def record_streaming_end(self):
        """OpenAI-streaming path completion; decrements streaming_inflight."""
        if self.streaming_inflight > 0:
            self.streaming_inflight -= 1

    def get_all_metrics(self, sdk_queue_depth=None):
        """
        Get the current metrics.

        `sdk_queue_depth` is the live SQLiteAckQueue length passed in by
        the route handler — the ground truth for the SDK path. We add
        `streaming_inflight` to it so the reported queue_depth reflects
        BOTH submission paths (SDK via SQLite, OpenAI-streaming via the
        in-memory counter). When `sdk_queue_depth` is None we fall back
        to the legacy `queue_depth` counter, which is fine for callers
        that don't have access to the queue.
        """
        if sdk_queue_depth is not None:
            queue_depth = sdk_queue_depth + self.streaming_inflight
        else:
            queue_depth = self.queue_depth + self.streaming_inflight

        token_rate, request_rate, flop_rate = self._windowed_rates()

        return {
            "queue_depth": queue_depth,
            "requests": self.total_completed_requests,
            "tokens": self.total_completed_tokens,
            "total_time": self.total_completed_response_time,
            "token/s": token_rate,
            "request/s": request_rate,
            "flop/s": flop_rate,
        }

    def _windowed_rates(self):
        """
        Compute (tokens, requests, flops) per second over the last
        RATE_WINDOW_SECONDS. Prunes the window in place. Returns zeros
        when the window is empty (no completions yet).
        """
        now = time.time()
        cutoff = now - RATE_WINDOW_SECONDS
        while self._rate_window and self._rate_window[0][0] < cutoff:
            self._rate_window.popleft()

        if not self._rate_window:
            return 0.0, 0.0, 0.0

        # Span is the elapsed seconds across the window. Use the full
        # configured window length once we have at least one full
        # window's worth of samples; otherwise use elapsed-since-first
        # so a fresh process doesn't divide by ~0.
        oldest_ts = self._rate_window[0][0]
        span = max(min(RATE_WINDOW_SECONDS, now - oldest_ts), 0.001)

        total_tokens = 0
        total_flops = 0
        for _, tokens, flops in self._rate_window:
            total_tokens += tokens
            total_flops += flops

        return (
            total_tokens / span,
            len(self._rate_window) / span,
            total_flops / span,
        )


def get_metrics() -> Metrics:
    """
    Get the metrics object.
    """
    global generate_metrics
    if generate_metrics is None:
        generate_metrics = Metrics()
    return generate_metrics
