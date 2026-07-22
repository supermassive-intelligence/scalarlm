# Phase-2 toolchain spike — findings (desk portion)

**Date:** 2026-07-14
**Plan:** `docs/reports/2026-07-13-vllm-fork-migration-plan.md` (Phase 2, the front gate)
**Verdict:** the toolchain risk is **bounded, not "unbounded"** — the two scary parts
(Rust toolchain on the base image; the `torch == 2.11.0` pin) both have clean, already-present
escape hatches. One genuine code risk remains (`csrc/libtorch_stable/*` vs torch 2.11).
The on-GPU **build + serve** validation still has to run on remote/Spark hardware — this box
has no GPU and can't hold the ~20 GB NGC image.

## NGC base image: torch version by tag (from NVIDIA release notes)

| NGC tag | torch | CUDA | Note |
|---|---|---|---|
| `26.01-py3` (**current base**) | 2.10.0a0 | 13.1.1 | matches v0.19 fork's torch 2.10 — why it works today |
| **`26.03-py3`** (**Phase-2 target**) | **2.11.0a0** | 13.2.0 | closest to v0.25's torch 2.11 |
| `26.06-py3` | 2.13.0a0 | 13.3.0 | already past 2.11 |

Python is **3.12** across all three (v0.25 wants `>=3.10,<3.15` ✅). NGC ships **no Rust
toolchain** (no `cargo`/`rustc`).

## The four toolchain questions, answered

1. **torch 2.11.** NGC ships *alpha* builds (`2.11.0a0`), and by PEP 440 `2.11.0a0 < 2.11.0`,
   so v0.25's `[build-system]` pin `torch == 2.11.0` would reject it — **except** the
   ScalarLM build already installs vLLM with **`--no-build-isolation`** (`Dockerfile:68`),
   which uses the container's torch and bypasses the build-requires pin entirely. So bumping
   the base to `26.03-py3` gives torch 2.11.0a0 and the build consumes it as-is. *Verify no
   runtime torch-version assertion trips.*
2. **Rust.** NGC has no `cargo`. Use the **abi3 precompiled-`.so` escape hatch** —
   `VLLM_USE_PRECOMPILED_RUST=1` makes `setup.py` fetch upstream's prebuilt wheel from
   `https://wheels.vllm.ai/{commit}/…` and extract `vllm/_rust_*.abi3.so`
   (`setup.py:490-501`). The Rust extension is `pyo3/abi3-py38` → Python-forward-compatible.
   The fork does **not** modify the Rust parser engine, so consuming the prebuilt `.so` is
   sound. No toolchain needed on the vast/Spark base image.
3. **transformers.** Current image caps `transformers>=5.5.0,<5.13`
   (`infra/requirements-megatron.txt`) — the cap existed because the old image shipped
   safetensors 0.7.0. **safetensors is now 0.8.0** in the built image, so the `<5.13` cap can
   be lifted to satisfy v0.25's `>=5.5.3` floor. *Confirm safetensors ≥0.8.0 on the 26.03
   image.*
4. **CUDA arches.** `26.03` = CUDA 13.2, which covers v0.25's family-specific `10.0f/12.0f`
   Blackwell targets. Re-validate the sm_90 / GB10 sm_120 kernel builds on hardware (the
   existing footguns), but the arch support is present.

## The one genuine "our code" risk

`csrc/libtorch_stable/*` — the fork's C++ ABI shims were built for torch 2.10. These must be
re-verified against torch 2.11's C++ ABI. This is the only toolchain item that is fork code,
not environment; it's the thing most likely to need a real fix.

## Concrete Phase-2 change set (to validate on GPU hardware)

- `Dockerfile`: `FROM nvcr.io/nvidia/pytorch:26.01-py3` → **`26.03-py3`** (nvidia + spark stages).
- Set `VLLM_USE_PRECOMPILED_RUST=1` (wheel commit matched to v0.25.0) — no `cargo` in the image.
- `infra/requirements-megatron.txt`: lift `transformers<5.13` cap (keep `>=5.5.3`); confirm
  safetensors ≥0.8.0 on the base.
- Keep `--no-build-isolation` (already present) → container torch 2.11.0a0 is used.
- Re-verify `csrc/libtorch_stable/*` against torch 2.11.

**Gate (unchanged):** current v0.19 fork gets a real memorize-PASS on the 26.03 image (one
dense model). Not doable on this box — needs remote/Spark GPU.

## What could not be checked here

- Whether NGC 26.03 ships safetensors ≥0.8.0 and a compatible torchvision (needs the image).
- Whether any runtime torch-version assertion in v0.25 trips on `2.11.0a0` under
  `--no-build-isolation`.
- The actual build + serve (GPU-only).
