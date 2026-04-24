"""
Unit tests for cray_infra.generate.metrics.

Metrics.queue_depth is a running counter that drifts up whenever a
request is submitted but never finished (error paths that skip
finish_work, uvicorn reloads, worker crashes). The `/v1/generate/metrics`
route handler passes the SQLiteAckQueue's actual `len()` as an override
to avoid surfacing that drift to the UI; these tests pin the override
semantics.
"""

from cray_infra.generate.metrics import Metrics


def test_get_all_metrics_uses_in_memory_counter_when_no_override():
    m = Metrics()
    m.record_new_request()
    m.record_new_request()
    m.record_new_request()
    result = m.get_all_metrics()
    assert result["queue_depth"] == 3


def test_override_replaces_in_memory_counter():
    m = Metrics()
    m.record_new_request()
    m.record_new_request()
    m.record_new_request()
    # SQLiteAckQueue reports 0 ground truth. The counter drifted up to 3
    # due to error paths; the override wins.
    result = m.get_all_metrics(queue_depth_override=0)
    assert result["queue_depth"] == 0


def test_override_zero_is_respected_not_treated_as_none():
    # Guard against the "override is falsy → fall back" bug.
    m = Metrics()
    m.record_new_request()
    result = m.get_all_metrics(queue_depth_override=0)
    assert result["queue_depth"] == 0


def test_override_none_falls_back_to_counter():
    m = Metrics()
    m.record_new_request()
    m.record_new_request()
    result = m.get_all_metrics(queue_depth_override=None)
    assert result["queue_depth"] == 2


def test_override_does_not_disturb_throughput_math():
    m = Metrics()
    m.record_new_request()
    # Finish one request with a token count to exercise the per-request math.
    m.record_completed_request(token_count=10, flop_count=100)
    result = m.get_all_metrics(queue_depth_override=999)
    # Throughput fields come from the in-memory totals, unaffected by
    # the override.
    assert result["queue_depth"] == 999
    assert result["requests"] == 1
    assert result["tokens"] == 10
