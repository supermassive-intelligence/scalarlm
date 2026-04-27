"""
Unit tests for cray_megatron.models.load_model.pick_attention_backend.

Auto-detects the fastest attention backend HF transformers can serve
given the runtime's installed packages. The right behaviour:

  - flash_attention_3 wins when the helper is importable AND returns True
  - flash_attention_2 wins when the helper is importable AND returns True,
    and flash_attention_3 wasn't found
  - sdpa otherwise; warning emitted on CUDA hosts so operators know
    they're missing a free speedup, suppressed on CPU
"""

import sys
import types
from unittest.mock import patch

import pytest

# Module under test imports gpu_aware_mpi at top level on the production
# path; in the unit test harness that should already be importable, but
# guard anyway to avoid a hard fail.
gpu_aware_mpi = pytest.importorskip("gpu_aware_mpi")

from cray_megatron.models.load_model import pick_attention_backend


def _stub_transformers_utils(*, fa2: bool, fa3: bool, has_fa3_helper: bool = True):
    """
    Insert a fake `transformers.utils` into sys.modules whose
    `is_flash_attn_2_available` / `is_flash_attn_3_available` return
    the requested booleans. Returns a context manager.
    """
    fake = types.ModuleType("transformers.utils")
    fake.is_flash_attn_2_available = lambda: fa2
    if has_fa3_helper:
        fake.is_flash_attn_3_available = lambda: fa3
    return patch.dict(sys.modules, {"transformers.utils": fake})


# ---- backend selection ----------------------------------------------------


def test_picks_flash3_when_available():
    with _stub_transformers_utils(fa2=False, fa3=True):
        impl, warning = pick_attention_backend()
    assert impl == "flash_attention_3"
    assert warning is None


def test_picks_flash2_when_only_flash2_available():
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, warning = pick_attention_backend()
    assert impl == "flash_attention_2"
    assert warning is None


def test_prefers_flash3_over_flash2():
    with _stub_transformers_utils(fa2=True, fa3=True):
        impl, _ = pick_attention_backend()
    assert impl == "flash_attention_3"


def test_falls_back_to_sdpa_with_warning_on_cuda():
    with _stub_transformers_utils(fa2=False, fa3=False), \
        patch("torch.cuda.is_available", return_value=True):
        impl, warning = pick_attention_backend()
    assert impl == "sdpa"
    assert warning is not None
    # Warning must mention the install hint so operators know the fix.
    assert "flash-attn" in warning


def test_falls_back_to_sdpa_silently_on_cpu():
    with _stub_transformers_utils(fa2=False, fa3=False), \
        patch("torch.cuda.is_available", return_value=False):
        impl, warning = pick_attention_backend()
    assert impl == "sdpa"
    assert warning is None


def test_handles_missing_fa3_helper_on_old_transformers():
    # Pre-fa3 transformers releases didn't ship is_flash_attn_3_available;
    # the picker has to ImportError-tolerate that and fall through to fa2.
    with _stub_transformers_utils(fa2=True, fa3=False, has_fa3_helper=False):
        impl, _ = pick_attention_backend()
    assert impl == "flash_attention_2"


def test_handles_torch_cuda_check_raising():
    # Some early-process states have torch.cuda not yet probed; we
    # should treat that as "no CUDA" and emit no warning.
    with _stub_transformers_utils(fa2=False, fa3=False), \
        patch("torch.cuda.is_available", side_effect=RuntimeError("no driver")):
        impl, warning = pick_attention_backend()
    assert impl == "sdpa"
    assert warning is None
