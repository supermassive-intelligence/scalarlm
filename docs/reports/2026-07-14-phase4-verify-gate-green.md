# Phase-4 VERIFY gate GREEN — v0.25 integration branch builds + passes Phase-0

**Date:** 2026-07-14
**Branch:** `georgi/vllm-0.25-integration` (fork), harness run on cuda-spark (GB10)
**Gate:** rebuild the vLLM-gate image on NGC 26.04 + run the Phase-0 harness
against the integration branch. This is the real VERIFY gate closing the
Phase-4 code-reconcile wave (see `2026-07-14-phase4-carryforward-inventory.md`).

## Result: GREEN

- **Build:** `scalarlm-cray:ngc2604-int` (29 GB) built clean on NGC 26.04-py3
  (torch 2.12.0a0, transformers 5.13.1, `vllm-0.6.6.dev0+g2210c8efc...cu132`).
  371 objects + libtorch_stable + editable install, ~35 min.
- **Phase-0 harness:** `13 passed, 3 skipped` in 60 s — **exactly the
  pre-migration baseline**. Symbol-drift green (every core-vLLM symbol the
  tokenformer/adapter layer imports resolves on v0.25); meta-build +
  two-pass adapter-key normalization green for all four arch families
  (llama, qwen2, qwen3, qwen3-moe) — no shed orphaned a call, no v0.25 tree
  reshape broke the LoRA path. 3 skips = transformers<5.13-gated gemma-mm
  fixtures (Phase-2 territory, expected).

## Three fixes required to clear the gate (land durably)

All three surfaced only at VERIFY — none were visible at Phase-2 (that gate
built the v0.19-era fork, before these v0.25 files/deps existed) or Phase-4
(compile-clean `.py`, no build/runtime).

### 1. `setuptools-rust` build dependency (build recipe)

v0.25's `pyproject.toml` build-system now requires `setuptools-rust`
(`setup.py:21` hard-imports `from setuptools_rust.build import build_rust`);
the Rust parser move (gemma4.rs et al.) brought it in. With
`--no-build-isolation`, build-system requires are skipped, so the image must
provide it. Build died in 3 s: `ModuleNotFoundError: No module named
'setuptools_rust'`.

**Fix** (`Dockerfile.ngc2604-spike:123`):
```
-RUN pip install setuptools-scm
+RUN pip install setuptools-scm setuptools-rust
```
The Rust *frontend* stays optional (`setup.py:1234`
`optional=not should_require_rust_frontend()`; `VLLM_REQUIRE_RUST_FRONTEND`
unset → missing cargo just skips the Rust build). Consistent with ADR 0008
(gemma4 serving deferred — no Rust toolchain needed).

### 2. `TORCH_TARGET_VERSION` 2.11 → 2.12 (integration branch)

`csrc/libtorch_stable/cuda_view.cu` is **new in v0.25** (absent in v0.19 — why
Phase-2 was green). Its two `torch::stable::from_blob` call sites use
**capturing-lambda deleters**, an overload gated
`#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0` in torch's
`csrc/stable/ops.h`. vLLM's `CMakeLists.txt` pins the stable-ABI target at
`TORCH_TARGET_VERSION=0x020B` (2.11) for portability, which compiles that
overload out — leaving only `DeleterFnPtr` overloads a capturing lambda can't
bind to. Build died in 178 s:
`cuda_view.cu(41,73): error: no instance of overloaded function
"torch::stable::from_blob"`.

**Fix** (`vllm/CMakeLists.txt:1094` `_C_stable_libtorch`, `:1319`
`_moe_C_stable_libtorch`):
```
-    TORCH_TARGET_VERSION=0x020B000000000000ULL)
+    TORCH_TARGET_VERSION=0x020C000000000000ULL)
```
Justified: the `.so` only ever runs on this exact NGC 26.04 image (torch
2.12.0a0), so targeting the 2.12 stable ABI is correct. Same alpha-torch-tax
class as the Phase-2 `register_opaque_type(hoist=)` finding (26.03's 2.11a0 →
26.04's 2.12a0).

### 3. `VLLM_TARGET_DEVICE=cpu` for the harness container (harness runner)

The Phase-0 harness is CPU-only, run with no `--gpus`. v0.25's DeviceConfig
resolves an unspecified platform's `device_type` to `''` →
`torch.device('') → RuntimeError: Device string must not be empty` — the 4
meta-build tests fail at `DeviceConfig.__post_init__`. The fork's carried
`platforms/__init__.py:182-193` (SCALARLM FIX) selects `CpuPlatform` **only
when `VLLM_TARGET_DEVICE=cpu` is set** — the harness container must pass it.
The v0.19 image happened to default to CPU; the carried v0.25 override makes
the env var load-bearing.

**Fix** (`test/finetune_sweep/run_phase0_harness.sh`): add
`-e VLLM_TARGET_DEVICE=cpu` to the `docker run` (and use `IMAGE=` +
`NO_BIND=1` against the rebuilt v0.25 image — the header already anticipates
dropping bind-mounts once the v0.25 toolchain lands).

Verified invocation (NO_BIND, as-built image):
```
docker run --rm -e VLLM_TARGET_DEVICE=cpu --entrypoint bash \
  scalarlm-cray:ngc2604-int -lc '
    pip install -q pytest; cd /app/cray/vllm
    python3 -m pytest tests/tokenformer/test_symbol_drift.py \
      tests/tokenformer/test_model_load_regression.py \
      -q --no-header --noconftest -p no:cacheprovider'
# → 13 passed, 3 skipped in 60.76s
```

## Landing status

- Fixes 1 & 2 are applied in the ephemeral spark worktree
  `/home/georgi/vllm025-intbuild` only (its top level is not a git repo; the
  `vllm/` subdir is a detached-HEAD checkout of the integration tip
  `a55fa5fd3`). **Not yet committed** — awaiting go.
- **To land:** commit Fix 2 (`CMakeLists.txt`) onto
  `georgi/vllm-0.25-integration` and push (origin's branch still has 0x020B →
  won't rebuild without it); land Fix 1 into the production Dockerfile; land
  Fix 3 into `run_phase0_harness.sh` on `georgi/finetune-sweep`.

## Next

Phase-5 hardware validation (real weight-load + serve on GB10 / vast), then
Phase-6 PR to `supermassive-intelligence/vllm-fork main` (sign-off Kari/Greg).

## Cross-refs

- Phase-4: `2026-07-14-phase4-carryforward-inventory.md`
- Phase-2: `2026-07-14-phase2-toolchain-spike-findings.md`
- Phase-0: `test/finetune_sweep/run_phase0_harness.sh`, `vllm/tests/tokenformer/`
- Plan: `2026-07-13-vllm-fork-migration-plan.md`
