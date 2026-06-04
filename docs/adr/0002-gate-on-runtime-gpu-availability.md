# Serve-tests gate on runtime GPU availability, not a named platform

A base model's eligibility to run in the serve-test sweep is decided **at runtime**
by probing the GPUs visible to the container — their count and free VRAM
(`torch.cuda.mem_get_info` per device) — and matching each model's declared
`requires: {gpus, min_free_vram_gb}`. The manifest's `targets` no longer carry a
static GPU count; the named hardware envelopes (`blackwell-4x`, `pc`) collapse into
a single `cuda` device target.

## Why

The box this runs on has a **variable** number of free GPUs (1–4; the others are
often busy with Kubernetes pods or other workloads). The old model gated on a fixed
named envelope with a declared `gpus_available`, which meant the sweep was only
meaningful when a whole envelope happened to be free, and a wrong tensor-parallel
size only blew up late (download + NCCL hang). Probing live free VRAM instead lets
the *same* sweep run against whatever is free: the models that fit run, and the rest
are `SKIPPED` with a precise reason (`"needs 2 GPU(s) with >=40GiB free; 1 qualify"`).

## Shape

- The **operator** decides which/how many GPUs the container sees
  (`NVIDIA_VISIBLE_DEVICES` / `--gpus`). The runner **does not select** among GPUs —
  no `CUDA_VISIBLE_DEVICES` juggling — it only reads what is visible and gates. This
  keeps the runner simple and the operator's config authoritative.
- **Skip, never shrink.** A serve-test is a reproducible claim about a *specific*
  config ("Qwen3-32B serves at TP=2"). Auto-shrinking tensor-parallel to fit fewer
  GPUs would make PASS/FAIL non-comparable across runs and manufacture OOMs — the
  exact signal ADR 0001's outcome enum exists to isolate. Under-capacity therefore
  yields `SKIPPED`, never a reduced-TP run.
- **CPU** has no CUDA devices, so the `requires` gate is N/A. CPU is per-model
  opt-in via `cpu_ok: true` (default false); `requires` and tensor-parallel do not
  apply there.

## Consequences

- `min_free_vram_gb` values in the manifest are conservative **hand estimates**,
  tuned against real runs. An estimate set too low yields an OOM instead of a clean
  SKIP — acceptable, because OOM is a classified outcome, not a silent pass.
- The runner imports `torch` (present in the device image) solely for the probe; it
  still shares no code with the cray stack (ADR 0001 holds).
- Because the gate is a *live* probe and models run sequentially, each engine must be
  **fully** torn down before the next probe — the runner SIGKILLs the engine's whole
  process group and waits for freed VRAM to settle, else a previous model's
  not-yet-reclaimed VRAM reads as "busy" and the next model wrongly SKIPs.
- This refines, not reverses, ADR 0001: the **device/image** split (cuda vs cpu) is
  still build-time and still selected by `--target`; only the **GPU-count** dimension
  moved from a static manifest field to a runtime probe.
