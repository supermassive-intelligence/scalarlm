"""Unit tests for the sm120 diffusion attention tiling patch.

Runs the patch against a minimal stand-in for vllm-fork's
triton_unified_attention.py carrying the two exact anchors, and checks the
result is what the patch claims: tuned only on capability family 120, and
idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import patch_sm120_diffusion_attention_tiling  # noqa: E402

SOURCE = '''\
def launch(head_size, max_seqlen_q, num_queries_per_kv, current_platform):
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    launch_num_warps = None
    launch_num_stages = None
    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and current_platform.is_device_capability_family(100)
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    # Ideally we would launch with kernel with:
    TILE_SIZE_PREFILL = 32
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128
    return BLOCK_M, BLOCK_Q, TILE_SIZE_PREFILL, launch_num_warps, launch_num_stages
'''


class _Platform:
    def __init__(self, family):
        self.family = family

    def is_device_capability_family(self, family):
        return family == self.family


def _patched(tmp_path):
    target = tmp_path / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE)
    patch_sm120_diffusion_attention_tiling(tmp_path)
    ns: dict = {}
    exec(target.read_text(), ns)
    return target, ns["launch"]


@pytest.mark.parametrize("head_size", [256, 512])
def test_sm120_canvas_passes_get_the_wide_tiling(tmp_path, head_size):
    _, launch = _patched(tmp_path)
    assert launch(head_size, 256, 2, _Platform(120)) == (64, 32, 64, 8, 1)


def test_sm120_decode_calls_keep_the_defaults(tmp_path):
    _, launch = _patched(tmp_path)
    assert launch(512, 1, 2, _Platform(120)) == (16, 8, 32, None, None)


def test_b200_path_is_unchanged(tmp_path):
    _, launch = _patched(tmp_path)
    assert launch(256, 256, 2, _Platform(100)) == (32, 16, 128, 8, 2)
    assert launch(512, 256, 2, _Platform(100)) == (16, 8, 32, None, None)


def test_applying_twice_is_a_no_op(tmp_path):
    target, _ = _patched(tmp_path)
    once = target.read_text()
    patch_sm120_diffusion_attention_tiling(tmp_path)
    assert target.read_text() == once


def test_drifted_source_fails_loudly(tmp_path):
    target = tmp_path / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE.replace("TILE_SIZE_PREFILL = 128", "TILE_SIZE_PREFILL = 96"))
    with pytest.raises(AssertionError):
        patch_sm120_diffusion_attention_tiling(tmp_path)
