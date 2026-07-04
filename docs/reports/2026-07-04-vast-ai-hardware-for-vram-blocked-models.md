# Testing VRAM-blocked models on vast.ai — hardware options

**Date:** 2026-07-04
**Context:** which sweep models are blocked/omitted by VRAM on the single DGX
Spark (GB10, 128 GiB unified), and what vast.ai hardware unblocks each tier.

---

## 1. What's actually blocked by VRAM

Read from `test/finetune_sweep/finetune-sweep.yaml` and
`docs/reports/2026-07-01-model-categories.md`. All footprints are **bf16
weights only** (2 bytes/param); training adds activations, serving adds KV
cache. The sweep trains a tiny LoRA (base weights frozen, so the dominant cost
is the base weights + the transient load-time copy + forward/backward
activations), then serves.

| Model | Total params | bf16 weights | Status today | Wall |
|---|---|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE) | 30B / 3B active | ~60 GiB | ✅ PASS (phase-scaled) | ceiling of the Spark |
| `Qwen/Qwen3.5-35B-A3B`, `Qwen3.6-35B-A3B` (MoE) | 35B / 3B active | ~70–75 GiB | staged, needs phase-scaling | borderline on Spark |
| `Qwen/Qwen3-32B` (dense) | 32B | ~64 GiB | staged, OOM-risk | **all 32B active** in train fwd/bwd → high activation memory |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` (MoE) | 47B / 12.9B active | ~94 GiB | ⛔ **BLOCKED — VRAM** | dies at phase-1 **train model-load** (~37% of shards → SIGKILL), takes co-resident API worker down, hangs the box |
| `zai-org/GLM-4.5-Air` (MoE) | ~106B | ~210 GiB | excluded before running | >128 GiB even phase-scaled |
| `meta-llama/Llama-4-Scout-17B-16E` (MoE, GATED) | 109B | ~210 GiB | excluded before running | >128 GiB even phase-scaled |
| 70–72B dense tier (e.g. Qwen2.5-72B) | 72B | ~144 GiB | omitted from the sweep entirely | >128 GiB |

**Two distinct memory ceilings matter:**

- **Serve peak** ≈ weights + KV cache. Phase-scaling gives vLLM the whole pool,
  so this is roughly `weights × 1.2`.
- **Train-load peak** is the real killer. The 30B cleared it only after the
  `on-device device_map` fix removed a `.to(device)` double-copy
  (`docs/reports/2026-07-01-30b-moe-phased-serving-four-fixes.md`). Mixtral-47B
  re-hit it: 47B co-located with the phase-1 API server exceeds 128 GiB.

So the question isn't just "bigger VRAM" — it's **enough headroom above the
weights for the load transient + the co-resident API worker.** Budget
`weights × ~1.3` as the floor.

## 2. The infra fork in the road (read this before picking hardware)

Your entire sweep architecture assumes **one GPU with one big memory pool**.
Phase-scaling (train, tear down, then serve — peak = 1 model) is a
single-GPU trick. The existing targets — `cuda-docker` (single-GPU Compose)
and `cuda-spark` (single-GPU phased) — map onto **one rented GPU with zero
infra changes.**

This splits the options cleanly:

- **Option A — one big-VRAM GPU.** A single H200 (141 GB) or B200 (180–192 GB)
  is a drop-in for the Spark: same "one unified pool, phase-scaled" model. Maps
  straight onto a `cuda-docker`-style target. **Unblocks everything up to ~72B
  on one card, and 100B+ MoE on a single B200.** No distributed code.
- **Option B — multi-GPU (2–8×).** Unblocks the 200 GiB+ tier, but you have
  **no target for it today.** vLLM serves tensor-parallel fine, but the megatron
  training phase and the runner are single-GPU; you'd be building
  distributed-training + TP-serve plumbing. Only worth it if 100B+ dense is a
  hard requirement.

**Recommendation: start with Option A.** Almost every blocked model in your
list fits on a single H200 or B200, and it costs you no new infra.

## 3. vast.ai hardware tiers (July 2026 rates)

Prices are marketplace medians and swing with supply; interruptible is cheaper.

| GPU | VRAM | ~$/GPU-hr | Notes |
|---|---|---|---|
| RTX 5090 | 32 GB | ~$0.4–0.7 | too small for anything blocked here |
| RTX 6000 Ada / A6000 | 48 GB | ~$0.5–0.9 | only helps ≤14B; not the blocked tier |
| A100 80GB (PCIe/SXM) | 80 GB | ~$1.0–1.8 | serves 35B MoE; tight for 47B train-load |
| H100 PCIe | 80 GB | ~$2.00 | same VRAM as A100, faster |
| H100 NVL | 94 GB | ~$2.12 | fits Mixtral-47B **serve**, tight for its train-load |
| **H200 NVL** | **141 GB** | **~$3.82** | **closest match to your GB10; unblocks ≤72B + Mixtral-47B with headroom** |
| **B200** | **180–192 GB** | **~$4.95** | **unblocks 72B comfortably and 100B+ MoE (GLM-4.5-Air, Llama-4-Scout) on ONE card** |

Multi-GPU (for Option B / the 200 GiB tier): vast.ai lists 2×/4×/8× H100 and
B200 boxes; e.g. 2×B200 ≈ 360 GB ≈ ~$10/hr, 4×H100 ≈ 320 GB ≈ ~$8/hr.

## 4. Recommended mapping

| Model tier | Recommended vast.ai box | Why |
|---|---|---|
| 32B dense / 35B MoE (`Qwen3-32B`, `Qwen3.5/3.6-35B-A3B`) | **1× H200 (141 GB)** | ~70 GiB weights + high train activation fits with margin; direct `cuda-docker` reuse |
| **Mixtral-8x7B (47B)** — the actual BLOCKED model | **1× H200 (141 GB)** | ~94 GiB weights + load transient + API worker fits in 141 GB (didn't in 128 GiB unified). B200 if you want comfort |
| 70–72B dense (currently omitted) | **1× B200 (180–192 GB)** | ~144 GiB weights fit one card; H200 too tight for train-load |
| GLM-4.5-Air / Llama-4-Scout (~106–109B) | **1× B200** (serve) or **2× B200 / 4× H100** (train) | ~210 GiB > any single card for train-load → Option B territory |

## 5. Cost to actually run it

A large-model end-to-end run (restart → train 450 steps → serve → check) is
roughly **1–1.5 GPU-hr** on Spark-class hardware; H200/B200 are faster, so
call it **≤1 hr each**. Concretely:

- **Unblock Mixtral-47B** on 1× H200: ~1 hr × $3.82 ≈ **~$4** for the run.
- **Sweep the whole 32B–72B tier** (4–5 models) on 1× B200: ~5 hr × $4.95 ≈
  **~$25**.
- These are interruptible-eligible workloads (checkpointed), so real cost can be
  ~30–50% lower.

The economics strongly favour: rent a single H200 or B200 on-demand, run the
large-model subset of the sweep, tear it down. No reservation needed.

## 6. What to do next

1. **Pick one card: H200 for the 32–47B blocked tier, B200 if you also want
   72B and 100B+ MoE.** H200 covers the single genuinely-BLOCKED model
   (Mixtral-47B) at ~$4/run.
2. **Add a `cuda-vast` target** (or reuse `cuda-docker`) — a rented vast.ai box
   is just a single-GPU Docker host over SSH, same shape as the 3090 target. The
   phase-scaled path isn't even required once VRAM > weights×1.3; co-located
   serving works with the extra headroom.
3. **Re-enable the Mixtral entry** (currently commented out at
   `finetune-sweep.yaml:432`) behind the new target — it's the cleanest single
   validation that bigger VRAM unblocks the separate-expert converter test that
   never got to run.
4. **Defer multi-GPU** until a 100B+ dense model is a real requirement; it's a
   distributed-infra project, not a hardware swap.

---

### Sources
- [Vast.ai GPU Pricing](https://vast.ai/pricing) · [H200 $3.82/hr](https://vast.ai/pricing/gpu/H200) · [B200 $4.95/hr](https://vast.ai/pricing/gpu/B200) · [H100 PCIe $2.00/hr](https://vast.ai/pricing/gpu/H100-PCIE) · [H100 NVL $2.12/hr](https://vast.ai/pricing/gpu/H100-NVL)
- Repo: `test/finetune_sweep/finetune-sweep.yaml`, `docs/reports/2026-07-01-model-categories.md`, `docs/reports/2026-07-01-30b-moe-phased-serving-four-fixes.md`
