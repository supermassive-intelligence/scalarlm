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


# ---- head_dim gate --------------------------------------------------------


class _Cfg:
    """Tiny stand-in for an HF config object — only needs attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_head_dim_over_256_forces_sdpa_even_when_flash_available():
    # Gemma-4 case: flash-attn 2 wheel is installed but the model has
    # head_dim > 256 which flash-attn's CUDA kernel rejects.
    cfg = _Cfg(head_dim=288, hidden_size=8192, num_attention_heads=32)
    with _stub_transformers_utils(fa2=True, fa3=True):
        impl, warning = pick_attention_backend(cfg)
    assert impl == "sdpa"
    assert warning is not None
    assert "head_dim" in warning
    assert "288" in warning


def test_head_dim_gate_reads_gemma4_global_head_dim():
    # Gemma-4 31B's text_config has `head_dim=256` (sliding layers,
    # under the cap) AND `global_head_dim=512` (full_attention layers,
    # over the cap). Reading only `head_dim` lets the 512-dim layers
    # through and crashes flash-attn at first forward. The gate has
    # to consider every `*_head_dim` attribute the config carries.
    text = _Cfg(
        head_dim=256,
        global_head_dim=512,
        hidden_size=5376,
        num_attention_heads=32,
    )
    cfg = _Cfg(text_config=text, model_type="gemma4")
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, warning = pick_attention_backend(cfg)
    assert impl == "sdpa"
    assert "512" in (warning or "")


def test_head_dim_gate_walks_nested_text_config():
    # Multimodal HF configs (Gemma4, Llama4) hide the real transformer
    # params inside `text_config`. The picker has to walk that.
    text = _Cfg(head_dim=320)
    cfg = _Cfg(text_config=text, vision_config=_Cfg(head_dim=64))
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, warning = pick_attention_backend(cfg)
    assert impl == "sdpa"
    assert "320" in (warning or "")


def test_head_dim_at_256_still_permits_flash():
    # Boundary: 256 is exactly flash-attn's cap, not over it.
    cfg = _Cfg(head_dim=256)
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, warning = pick_attention_backend(cfg)
    assert impl == "flash_attention_2"
    assert warning is None


def test_head_dim_derived_from_hidden_size_when_attribute_missing():
    # Configs without explicit head_dim still need to be inspected via
    # hidden_size // num_attention_heads.
    cfg = _Cfg(hidden_size=8192, num_attention_heads=16)  # 512
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, _ = pick_attention_backend(cfg)
    assert impl == "sdpa"


def test_no_model_config_does_not_gate_flash():
    # Backwards-compat: callers that don't pass a config get the old
    # behaviour (flash if installed, regardless of model shape).
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, _ = pick_attention_backend()
    assert impl == "flash_attention_2"


def test_unprobeable_config_does_not_gate_flash():
    # If no head_dim and no hidden_size/heads pair are discoverable,
    # we have no signal — better to let flash try and fall through to
    # the materialize_model retry-on-sdpa belt rather than block it.
    cfg = _Cfg()
    with _stub_transformers_utils(fa2=True, fa3=False):
        impl, _ = pick_attention_backend(cfg)
    assert impl == "flash_attention_2"
