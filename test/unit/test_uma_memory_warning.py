"""Unit tests for the unified-memory gpu_memory_utilization warning.

Integrated GPUs such as GB10 / DGX Spark and Jetson share system RAM with
the CPU. The warning must fire above the configured threshold on integrated
GPUs and stay silent on discrete ones. The threshold is a heuristic, not a
guarantee of sufficient host headroom.
"""

from __future__ import annotations

import pytest

from cray_infra.one_server.vllm_cli_args import (
    UMA_SAFE_GPU_MEMORY_UTILIZATION,
    uma_memory_warning,
)


@pytest.mark.parametrize("utilization", [0.85, 0.92, 1.0])
def test_warns_above_the_safe_band_on_integrated_gpu(utilization):
    warning = uma_memory_warning(utilization, is_integrated_gpu=True)
    assert warning is not None
    assert str(utilization) in warning


@pytest.mark.parametrize("utilization", [0.40, 0.80])
def test_silent_at_or_below_the_safe_band(utilization):
    assert uma_memory_warning(utilization, is_integrated_gpu=True) is None


@pytest.mark.parametrize("utilization", [0.40, 0.92, 1.0])
def test_silent_on_discrete_gpu_where_vram_is_separate(utilization):
    assert uma_memory_warning(utilization, is_integrated_gpu=False) is None


def test_shipped_default_is_inside_the_safe_band():
    """The default must not trip its own warning on a unified-memory host."""
    from cray_infra.util.default_config import Config

    default = Config().gpu_memory_utilization
    assert default <= UMA_SAFE_GPU_MEMORY_UTILIZATION
    assert uma_memory_warning(default, is_integrated_gpu=True) is None
