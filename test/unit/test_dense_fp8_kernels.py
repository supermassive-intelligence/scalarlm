"""Hardware tests for the Gemma4 dense fp8 patch (scalarlm_dense_fp8.py).

Requires a CUDA GPU with fp8 tensor cores and the vllm custom ops. The module is loaded
from the patch source so these tests check what the patch installs. What is covered:

  * layer selection and the K % 16 rule are pure functions
  * weight quantization: per-channel scales, codes stay in range, exact for inputs that
    e4m3 represents exactly, relative error within fp8 precision otherwise
  * the fp8 GEMM against the bf16 reference: exact (to bf16 rounding) when the inputs are
    exactly representable, within a few percent relative Frobenius error on Gaussian data
  * the installed method: untouched layers go through the original path, selected layers
    through fp8 with bias and leading dims preserved
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)
pytest.importorskip("vllm")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))
from apply_patches import DENSE_FP8_SRC  # noqa: E402


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    path = tmp_path_factory.mktemp("fp8") / "scalarlm_dense_fp8.py"
    path.write_text(DENSE_FP8_SRC)
    os.environ.setdefault("SCALARLM_DENSE_FP8_LAYERS", "qkv_proj,o_proj,gate_up_proj,down_proj")
    spec = importlib.util.spec_from_file_location("scalarlm_dense_fp8", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # installs the method swap on UnquantizedLinearMethod
    yield mod
    mod.UnquantizedLinearMethod.process_weights_after_loading = mod._orig_process
    mod.UnquantizedLinearMethod.apply = mod._orig_apply


def test_layer_selection(m):
    layers = m.selected_layers("qkv_proj,o_proj:gate_up_proj")
    assert layers == ("qkv_proj", "o_proj", "gate_up_proj")
    assert m.is_selected("model.layers.3.self_attn.qkv_proj", layers)
    assert m.is_selected("model.layers.3.mlp.gate_up_proj", layers)
    assert not m.is_selected("model.layers.3.mlp.down_proj", layers)
    assert not m.is_selected("model.layers.3.mlp.experts", layers)
    assert not m.is_selected("lm_head", layers)
    assert m.selected_layers() == tuple(m.DEFAULT_LAYERS.split(","))


def test_convertible_rule(m):
    ok = torch.zeros(8, 2816, dtype=torch.bfloat16)
    assert m.convertible(ok)
    assert not m.convertible(torch.zeros(8, 2818, dtype=torch.bfloat16))   # K % 16 != 0
    assert not m.convertible(ok.float())
    assert not m.convertible(torch.zeros(8, 16, 16, dtype=torch.bfloat16))


def _pow2_tensor(*shape, device="cuda"):
    """Values +-2^k, k in [-3, 3]: exactly representable in e4m3 and in bf16."""
    k = torch.randint(-3, 4, shape, device=device).float()
    sign = torch.randint(0, 2, shape, device=device).float() * 2 - 1
    return (sign * torch.exp2(k)).to(torch.bfloat16)


def test_weight_quantization(m):
    torch.manual_seed(0)
    w = (torch.randn(256, 512, device="cuda") * 0.02).to(torch.bfloat16)
    w8, scale = m.quantize_weight(w)
    assert w8.dtype == torch.float8_e4m3fn and w8.shape == w.shape
    assert scale.shape == (256, 1) and scale.dtype == torch.float32
    deq = w8.float() * scale
    # per-row absmax maps to +-448, so every row's max is reconstructed exactly (to e4m3 rounding of 448)
    assert torch.allclose(deq.abs().amax(dim=1), w.float().abs().amax(dim=1), rtol=1e-2)
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.06, f"fp8 weight round-trip error {rel:.4f}"
    wp = _pow2_tensor(64, 128)
    w8p, sp = m.quantize_weight(wp)
    assert torch.equal((w8p.float() * sp).to(torch.bfloat16), wp), "powers of two must round-trip exactly"


def test_fp8_linear_exact_on_representable_inputs(m):
    torch.manual_seed(1)
    x = _pow2_tensor(64, 256)
    w = _pow2_tensor(96, 256)
    w8, ws = m.quantize_weight(w)
    out = m.fp8_linear(x, w8, ws)
    ref = torch.nn.functional.linear(x, w)
    # both sides accumulate in fp32 and round once to bf16; summation order may differ
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=0.0), (out - ref).abs().max()


@pytest.mark.parametrize("M,N,K", [(16, 128, 256), (256, 2816, 2112), (2048, 8192, 2816)])
def test_fp8_linear_close_to_bf16(m, M, N, K):
    torch.manual_seed(M + N + K)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    w8, ws = m.quantize_weight(w)
    out = m.fp8_linear(x, w8, ws).float()
    ref = torch.nn.functional.linear(x, w).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.05, f"relative Frobenius error {rel:.4f}"
    assert out.shape == (M, N) and out.dtype == torch.float32  # (.float() above; the op returns bf16)


class _Layer(torch.nn.Module):
    def __init__(self, prefix, n, k, bias=False):
        super().__init__()
        self.prefix = prefix
        self.weight = torch.nn.Parameter((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.randn(n, device="cuda").to(torch.bfloat16), requires_grad=False) if bias else None


def test_installed_method_converts_only_selected_layers(m):
    torch.manual_seed(2)
    method = m.UnquantizedLinearMethod()
    sel = _Layer("model.layers.0.mlp.down_proj", 2816, 2112, bias=True)
    other = _Layer("model.layers.0.mlp.router", 128, 2816)
    method.process_weights_after_loading(sel)
    method.process_weights_after_loading(other)
    assert hasattr(sel, "scalarlm_w8") and not hasattr(other, "scalarlm_w8")
    x = torch.randn(4, 32, 2112, device="cuda", dtype=torch.bfloat16)   # leading dims preserved
    out = method.apply(sel, x, sel.bias)
    ref = torch.nn.functional.linear(x, sel.weight, sel.bias)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05
    xo = torch.randn(8, 2816, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(method.apply(other, xo), torch.nn.functional.linear(xo, other.weight)), "unselected layers must take the original path"


def test_installed_method_leaves_misaligned_weights_alone(m):
    method = m.UnquantizedLinearMethod()
    odd = _Layer("model.layers.1.self_attn.o_proj", 64, 2818)
    method.process_weights_after_loading(odd)
    assert not hasattr(odd, "scalarlm_w8")
