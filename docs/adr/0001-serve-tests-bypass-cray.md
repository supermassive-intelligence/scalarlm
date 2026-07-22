# Base-model serve-tests bypass the cray server

When checking whether a *base model* serves and produces sane output (test tiers
(a) "does it serve" and (b) "output sanity"), we drive `vllm serve` **directly
inside the per-device target image** and do **not** start the cray FastAPI server.

## Why

Serve-ability and raw output quality are properties of the **vLLM fork**, not of the
cray layer — `create_vllm.py` simply forwards `args.model` to vLLM's own loader, so
running these checks through `./scalarlm up` adds zero signal while forcing the slow
restart path (`kill_vllm_container()` kills *all* python and exits). The cray stack
only matters for the *integration* tiers — (c) the generate queue / tool calling /
chat templates and (d) tokenformer training — which test surfaces that don't exist in
bare vLLM.

## Shape

- A single committed manifest (`test/model_sweep/model-sweep.yaml`) lists real HF
  model IDs with per-target overrides; the same manifest is reused by the future
  (c)/(d) integration harness.
- The runner launches **one subprocess per model** for crash/hang/OOM isolation and
  clean VRAM teardown, then classifies the result into an **enum**
  (`SERVED_PASS` / `SERVED_FAIL_QUALITY` / `FAILED_TO_SERVE` / `OOM` / `GATED` /
  `MISSING` / `SKIPPED`) so a missing HF token or an OOM never masquerades as
  "architecture unsupported."
- **Device is a build-time split**: `VLLM_TARGET_DEVICE` is baked when the image
  compiles vLLM, so each hardware target (cuda for Blackwell/PC, cpu for CPU-only) is
  a separate image build. The harness runs once per target-image, filtering the
  manifest by `--target`.

## Consequences

- The (a)/(b) runner is a serve-only test driver: it shares no code with the eventual
  cray-stack integration harness (the two duplicate rather than abstract a common
  base), and the manifest is the one durable contract across both.
- Serve-tests lose fidelity to "how cray runs vLLM in-process" — acceptable, because
  that fidelity is exactly what the (c) integration tier is for.
