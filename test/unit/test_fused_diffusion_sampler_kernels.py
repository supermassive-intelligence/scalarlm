"""GPU tests for the fused DiffusionGemma sampler kernels (scalarlm_fused_sampler.py).

Skipped without CUDA + Triton. Run inside the serving image on one GPU (~16 GB free, ~4 min):

    python -m pytest test/unit/test_fused_diffusion_sampler_kernels.py -v

The module under test is the source embedded in apply_patches.py, so these tests cover what
the image build writes into vLLM.

What is checked:
  * the in-kernel uniform noise: in [0, 1), never 1.0 (that would give +inf Gumbel noise and
    force a random token), and the same top-level mass as torch.rand in fp32
  * Gumbel-max sampling: sampled-token frequencies match vLLM's original formula
    (`_compiled_sample_step` phase 2) for confident, medium and flat-tail rows
  * independence across rows and across calls, and uniformity over vocab positions
  * deterministic outputs against an fp64 reference: argmax, entropy (including the 0.005
    confidence threshold and the 0.1 entropy bound), bf16 probabilities, masked (-inf) tokens,
    vocab sizes that are not a multiple of the kernel block
  * the in-kernel softcap on raw bf16 logits equals `_softcap_logits` followed by the fp32 path
  * batch invariance: a request's deterministic outputs are bitwise equal alone and in a batch

Note on the spec: fp32 Gumbel noise tops out at 16.64 (u = 1 - 2^-24), so both the original
formula and this one undersample tokens with probability below ~1e-7 relative to exact
softmax. The requirement here is equality with the original, not with exact softmax.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import triton.language as tl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))
from apply_patches import FUSED_DIFFUSION_SAMPLER_SRC  # noqa: E402


def _load_sampler():
    # Triton reads kernel source through inspect, so the module has to live in a real file.
    d = Path(tempfile.mkdtemp(prefix="scalarlm_fused_sampler_"))
    f = d / "scalarlm_fused_sampler.py"
    f.write_text(FUSED_DIFFUSION_SAMPLER_SRC)
    spec = importlib.util.spec_from_file_location("scalarlm_fused_sampler", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load_sampler()
_UNIFORM = fs._uniform24

DEV = "cuda"
V, CL = 262144, 256
Z = 4.5                      # z-score bound for the statistical checks (~7e-6 two-sided)


# ---------------------------------------------------------------- the uniform noise itself
@triton.jit
def _u_stats(seed, V, n1_ptr, n0_ptr, mx_ptr, mn_ptr, sm_ptr, gmx_ptr, top_ptr, BLOCK: tl.constexpr):
    """Same expression as _row_stats: u = _uniform24(seed, row * V + idx)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mx = tl.zeros([BLOCK], tl.float32)
    mn = tl.full([BLOCK], 2.0, tl.float32)
    sm = tl.zeros([BLOCK], tl.float32)
    gmx = tl.full([BLOCK], float("-inf"), tl.float32)
    n1 = tl.zeros([BLOCK], tl.int32)
    n0 = tl.zeros([BLOCK], tl.int32)
    ntop = tl.zeros([BLOCK], tl.int32)
    for start in range(0, V, BLOCK):
        idx = start + offs
        u = _UNIFORM(seed, row.to(tl.int64) * V + idx)
        mx = tl.maximum(mx, u)
        mn = tl.minimum(mn, u)
        sm += u
        n1 += tl.where(u >= 1.0, 1, 0)
        n0 += tl.where(u <= 0.0, 1, 0)
        ntop += tl.where(u >= 0.99999994, 1, 0)      # the top fp32 level, u = 1 - 2^-24
        uc = tl.maximum(u, 1e-20)
        gmx = tl.maximum(gmx, -tl.log(-tl.log(uc)))
    tl.store(n1_ptr + row, tl.sum(n1, 0))
    tl.store(n0_ptr + row, tl.sum(n0, 0))
    tl.store(mx_ptr + row, tl.max(mx, 0))
    tl.store(mn_ptr + row, tl.min(mn, 0))
    tl.store(sm_ptr + row, tl.sum(sm, 0))
    tl.store(gmx_ptr + row, tl.max(gmx, 0))
    tl.store(top_ptr + row, tl.sum(ntop, 0))


def test_uniform_noise_matches_torch_rand_fp32():
    n = 16384                                         # 16384 * 262144 = 4.3e9 draws per seed
    tot = ones = tops = 0
    umax, umin, s, gmax = 0.0, 1.0, 0.0, -1e9
    for seed in (1, 12345, 2**31 - 2):
        i32 = lambda: torch.zeros(n, dtype=torch.int32, device=DEV)   # noqa: E731
        f32 = lambda: torch.zeros(n, device=DEV)                       # noqa: E731
        n1, n0, tp, mx, mn, sm, gm = i32(), i32(), i32(), f32(), f32(), f32(), f32()
        _u_stats[(n,)](seed, V, n1, n0, mx, mn, sm, gm, tp, BLOCK=2048, num_warps=8)
        tot += n * V
        ones += int(n1.sum())
        tops += int(tp.sum())
        umax, umin = max(umax, float(mx.max())), min(umin, float(mn.min()))
        s += float(sm.double().sum())
        gmax = max(gmax, float(gm.max()))
    assert ones == 0 and umax < 1.0, "u == 1.0 would give +inf Gumbel noise"
    assert umin >= 0.0
    assert abs(s / tot - 0.5) < 6 * math.sqrt(1 / 12 / tot) + 1e-6
    assert math.isfinite(gmax) and gmax < 16.7
    expected = tot * 2.0**-24                         # torch.rand: every fp32 level has mass 2^-24
    assert abs(tops - expected) / math.sqrt(expected) < Z, (
        f"top noise level count {tops} vs {expected:.0f}: the far Gumbel tail differs from torch.rand"
    )


# ---------------------------------------------------------------- sampling distribution
KINDS = {   # head probabilities; the rest of the mass is spread over the remaining vocab
    "confident": [0.97, 0.015, 0.008, 0.004],   # a committed position
    "medium": [0.60, 0.25, 0.10],
    "flat": [0.50],                             # half the mass over 262k tokens
}
T = 0.6


def _row(kind):
    head = KINDS[kind]
    p = torch.full((V,), (1.0 - sum(head)) / (V - len(head)), dtype=torch.float64)
    pos = torch.randperm(V, generator=torch.Generator().manual_seed(7))[: len(head)]
    p[pos] = torch.tensor(head, dtype=torch.float64)
    return (torch.log(p) * T).float().to(DEV), pos.to(DEV)   # softmax(logits / T) == p


def _fused(row, canvases, calls):
    x = row[None, :].expand(canvases * CL, V)       # stride 0: every row reads the same logits
    temp = torch.full((canvases,), T, device=DEV)
    out = []
    for _ in range(calls):
        am, gm, _, _ = fs.vocab_stats_and_probs(x, temp, CL, 0, 2048)
        out.append(gm)
    return torch.stack(out), am


def _original(row, n, calls):
    """vLLM's `_compiled_sample_step` phase 2, verbatim."""
    out = []
    for _ in range(calls):
        scaled = row[None, :].expand(n, V).float() / T
        u = torch.rand_like(scaled).clamp(min=1e-20)
        out.append((scaled - torch.log(-torch.log(u))).argmax(dim=-1))
        del scaled, u
    return torch.stack(out)


@pytest.mark.parametrize("kind", list(KINDS))
def test_sampling_matches_the_original_formula(kind):
    row, pos = _row(kind)
    f, am = _fused(row, 256, 16)                    # ~1.05M samples
    o = _original(row, 2048, 512)                   # ~1.05M samples
    assert bool((am == pos[0]).all())
    for c in [int(i) for i in pos] + [-1]:          # each head token, then the tail bucket
        cf = float(((f == c) if c >= 0 else ~torch.isin(f, pos)).float().mean())
        co = float(((o == c) if c >= 0 else ~torch.isin(o, pos)).float().mean())
        pool = (cf * f.numel() + co * o.numel()) / (f.numel() + o.numel())
        se = math.sqrt(max(pool * (1 - pool), 1e-12) * (1 / f.numel() + 1 / o.numel()))
        assert abs(cf - co) < Z * se, f"{kind}: token {c}: fused {cf:.5f} vs original {co:.5f}"


@pytest.mark.parametrize("kind", list(KINDS))
def test_noise_is_independent_across_rows_and_calls(kind):
    row, pos = _row(kind)
    f, _ = _fused(row, 256, 4)
    _, cnt = torch.unique(f, return_counts=True)
    coll = float(((cnt.double() / f.numel()) ** 2).sum())      # P(two independent samples agree)
    n = f.shape[1]
    se = math.sqrt(coll * (1 - coll) / n)
    adjacent = float((f[0, 1:] == f[0, :-1]).float().mean())
    across_calls = float((f[0] == f[1]).float().mean())
    assert abs(adjacent - coll) < Z * se, f"adjacent rows agree {adjacent:.4f}, expected {coll:.4f}"
    assert abs(across_calls - coll) < Z * se, f"consecutive calls agree {across_calls:.4f}, expected {coll:.4f}"
    if kind == "flat":                                          # tail samples uniform over the vocab
        tail = f[f != pos[0]].flatten()
        bins = torch.bincount((tail * 64 // V).long(), minlength=64).double()
        e = tail.numel() / 64
        assert float(((bins - e) / math.sqrt(e)).abs().max()) < Z


# ---------------------------------------------------------------- deterministic outputs
def _logits(n_rows, v=V, dtype=torch.float32, masked=False, seed=0):
    """Raw LM-head-like rows mixing confident and uncertain positions."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(n_rows, v, device=DEV, generator=g) * 2.0
    hot = torch.randint(0, v, (n_rows,), device=DEV, generator=g)
    kind = torch.rand(n_rows, device=DEV, generator=g)
    z[torch.arange(n_rows, device=DEV), hot] += torch.where(kind < 0.4, 30.0, torch.where(kind < 0.7, 16.0, 4.0))
    if masked:
        z[:, torch.randint(0, v, (v // 3,), device=DEV, generator=g)] = float("-inf")
    return z.to(dtype)


def _reference(x32, temp, cl):
    inv_t = (1.0 / temp.double()).repeat_interleave(cl)[:, None]
    am = torch.empty(x32.shape[0], dtype=torch.long, device=DEV)
    ent = torch.empty(x32.shape[0], dtype=torch.float64, device=DEV)
    probs = []
    for i in range(0, x32.shape[0], 256):
        s = x32[i:i + 256].double() * inv_t[i:i + 256]
        lp = s.log_softmax(-1)
        pr = lp.exp()
        am[i:i + 256] = s.argmax(-1)
        ent[i:i + 256] = -(pr * torch.where(pr > 0, lp, torch.zeros_like(lp))).sum(-1)
        probs.append(pr)
    return am, ent, torch.cat(probs)


def _check(x_in, x_ref, softcap, n_req, cl, v):
    temp = torch.linspace(0.4, 0.8, n_req, device=DEV)
    am, gm, ent, pr = fs.vocab_stats_and_probs(x_in, temp, cl, 0, v, softcap=softcap)
    ram, rent, rp = _reference(x_ref, temp, cl)
    top2 = torch.topk(x_ref, 2, dim=-1).values
    tied = (top2[:, 0] - top2[:, 1]) <= 1e-5 * top2[:, 0].abs()        # either index is right
    assert bool(((am == ram) | tied).all()), "argmax differs"
    err = (ent.double() - rent).abs()
    assert float(err.max()) < 2e-4 or float((err / rent.clamp(min=1e-3)).max()) < 1e-3, "entropy"
    for thr in (0.005, 0.1):                                           # confidence threshold, entropy bound
        flips = (ent.double() < thr) != (rent < thr)
        assert bool((~flips | ((rent - thr).abs() < 2e-4)).all()), f"entropy on the wrong side of {thr}"
    perr = (pr.float().double() - rp).abs()
    assert bool((perr <= rp * 2**-8 * 1.02 + 1e-9).all()), "probabilities beyond bf16 rounding"
    masked = torch.isinf(x_ref) & (x_ref < 0)
    if bool(masked.any()):
        rows = torch.arange(len(am), device=DEV)
        assert bool((pr.float()[masked] == 0).all()), "masked token got probability"
        assert not bool(masked[rows, am].any()) and not bool(masked[rows, gm].any()), "masked token chosen"
    return am, ent, pr


def _capped(x):
    return torch.tanh(x.float() / 30.0) * 30.0      # == vLLM's _softcap_logits(x, 30.0)


def test_deterministic_outputs_match_fp64_reference():
    x = _capped(_logits(8 * CL))
    _check(x, x, 0.0, 8, CL, V)


def test_masked_tokens_stay_masked():
    x = _capped(_logits(8 * CL, masked=True, seed=1))
    _check(x, x, 0.0, 8, CL, V)


def test_vocab_size_not_a_multiple_of_the_block():
    x = _capped(_logits(64, v=5003, seed=2))
    _check(x, x, 0.0, 4, 16, 5003)


@pytest.mark.parametrize("masked", [False, True])
def test_in_kernel_softcap_on_raw_bf16_logits(masked):
    raw = _logits(8 * CL, dtype=torch.bfloat16, masked=masked, seed=3)
    _check(raw, _capped(raw), 30.0, 8, CL, V)


def test_softcap_reference_is_vllms():
    dg = pytest.importorskip("vllm.model_executor.models.diffusion_gemma")
    raw = _logits(CL, dtype=torch.bfloat16, seed=4)
    assert torch.equal(dg._softcap_logits(raw, 30.0), _capped(raw))


def test_outputs_do_not_depend_on_the_rest_of_the_batch():
    x = _capped(_logits(8 * CL))
    temp = torch.linspace(0.4, 0.8, 8, device=DEV)
    am, _, ent, pr = fs.vocab_stats_and_probs(x, temp, CL, 0, V)
    r = 3
    sl = slice(r * CL, (r + 1) * CL)
    am1, _, ent1, pr1 = fs.vocab_stats_and_probs(x[sl], temp[r:r + 1], CL, 0, V)
    assert torch.equal(am1, am[sl]) and torch.equal(ent1, ent[sl]) and torch.equal(pr1, pr[sl])
