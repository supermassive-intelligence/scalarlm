"""Unit tests for the Gemma4 dense fp8 patch (source transformation; no GPU).

The GEMM path itself is tested on hardware in test_dense_fp8_kernels.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import DENSE_FP8_SRC, patch_gemma4_dense_fp8  # noqa: E402

GEMMA4 = '''\
import torch
from torch import nn


class Gemma4MLP(nn.Module):
    pass


class Gemma4Attention(nn.Module):
    pass
'''


@pytest.fixture
def vllm_root(tmp_path: Path) -> Path:
    models = tmp_path / "vllm" / "model_executor" / "models"
    models.mkdir(parents=True)
    (models / "gemma4.py").write_text(GEMMA4)
    return tmp_path


def test_patch_adds_module_and_guarded_import(vllm_root: Path):
    patch_gemma4_dense_fp8(vllm_root)
    models = vllm_root / "vllm" / "model_executor" / "models"
    src = (models / "gemma4.py").read_text()
    assert (models / "scalarlm_dense_fp8.py").read_text() == DENSE_FP8_SRC
    hook = src.index("scalarlm_dense_fp8")
    assert hook < src.index("class Gemma4MLP("), "the import must run before the model classes are defined"
    assert 'environ.get("SCALARLM_DENSE_FP8") == "1"' in src, "the patch must be off unless the env var is set"
    compile(src, "gemma4.py", "exec")
    compile(DENSE_FP8_SRC, "scalarlm_dense_fp8.py", "exec")


def test_patch_is_idempotent(vllm_root: Path):
    patch_gemma4_dense_fp8(vllm_root)
    once = (vllm_root / "vllm" / "model_executor" / "models" / "gemma4.py").read_text()
    patch_gemma4_dense_fp8(vllm_root)
    assert (vllm_root / "vllm" / "model_executor" / "models" / "gemma4.py").read_text() == once


def test_patch_requires_its_anchor(tmp_path: Path):
    models = tmp_path / "vllm" / "model_executor" / "models"
    models.mkdir(parents=True)
    (models / "gemma4.py").write_text("class Something:\n    pass\n")
    with pytest.raises(AssertionError):
        patch_gemma4_dense_fp8(tmp_path)


def test_missing_gemma4_is_skipped(tmp_path: Path):
    patch_gemma4_dense_fp8(tmp_path)  # prints and returns; nothing written
    assert not (tmp_path / "vllm").exists()


def test_module_documents_its_env_switches():
    for name in ("SCALARLM_DENSE_FP8=1", "SCALARLM_DENSE_FP8_LAYERS", "SCALARLM_DENSE_FP8_LOG"):
        assert name in DENSE_FP8_SRC
    assert "qkv_proj,o_proj,gate_up_proj,down_proj" in DENSE_FP8_SRC
