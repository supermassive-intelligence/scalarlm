# Runbook: running the finetune sweep on vast.ai (`cuda-vast` target)

**Purpose:** test the VRAM-blocked / large models (Mixtral-8x7B/47B, the 32–35B
tier, 70–72B dense) that don't fit the DGX Spark's 128 GiB unified pool, by
renting a single big-VRAM GPU (H200 141 GiB / B200 180 GiB) on vast.ai.

**Sizing rationale:** `docs/reports/2026-07-04-vast-ai-hardware-for-vram-blocked-models.md`.
**Target config:** the `cuda-vast` entry in `test/finetune_sweep/finetune-sweep.yaml`.

---

## Mental model (why this works with zero code changes)

`cuda-vast` is `cuda-spark` on x86 hardware. Both are **scheduler-less
single-GPU Docker hosts**, so:

- The sweep runs **ON the rented box** (not from your laptop). The Compose path
  in `run_finetune_sweep.py` drives `docker compose exec/ps/rm/logs` against the
  **local** daemon — there is no remote-daemon mode.
- `restart_cmd` uses `./scalarlm up nvidia`, which builds `linux/amd64`, selects
  the generic `cray-nvidia` compose service, and **auto-detects `sm_arch`** from
  `nvidia-smi --query-gpu=compute_cap` (H200 → `9.0`, B200 → `10.0`). No new
  compose service, no arch flags to pass.
- `phase_scaled: true` is retained and **load-bearing**: co-located serving loads
  the weights twice (train copy + vLLM copy), so 47B ≈ 188 GiB — over any single
  card. Whole-GPU-per-phase (peak = 1 model) is what makes Mixtral fit at all.

## Which card

| Card | VRAM | sm_arch | Use for | Risk |
|---|---|---|---|---|
| **H200** | 141 GiB | 9.0 (Hopper) | 32–47B tier incl. the BLOCKED Mixtral | **low** — mature kernels |
| **B200** | 180 GiB | 10.0 (Blackwell) | + 70–72B, 100B+ MoE | verify the fork builds `sm_100` (≠ Spark's `sm_120`) before trusting a run |

Start on **H200** to validate the target; move to B200 only for the 72B+ tier.

## Prerequisites when picking the vast.ai offer

- **Docker daemon access, not just an app container.** The stack launches
  `docker compose` with `runtime: nvidia`. A plain vast.ai app-container
  (docker-in-docker) fights this — choose an offer with host/VM Docker access, or
  a template that exposes `/var/run/docker.sock` + the nvidia container runtime.
- **1× GPU, on-demand** (interruptible is fine for checkpointed runs but can
  vanish mid-model). Filter for H200 or B200, 1× count.
- **Enough disk** for the model download + job checkpoints (a 47–72B model in
  bf16 is 90–150 GiB on disk; budget ≥ 250 GiB).

## Startup process

**Shortcut:** `scripts/provision-vast.sh` automates steps 1–5 below (preflight →
clone/checkout → HF auth → cache-warm → pre-build) and prints the sweep command.
Run it on the box: `export HF_TOKEN=hf_...; ./scripts/provision-vast.sh mistralai/Mixtral-8x7B-Instruct-v0.1`.
The manual steps below are what it runs, for when you want to do it by hand or
debug a failure.

```bash
# 1. On the rented box: sanity-check the GPU + Docker/nvidia runtime
nvidia-smi
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 2. Clone the lab branch
git clone https://github.com/supermassive-intelligence/scalarlm.git   # or your remote
cd scalarlm
git checkout georgi/finetune-sweep

# 3. Auth for gated repos (also used by preflight's `docker compose run`)
export HF_TOKEN=hf_...

# 4. (Recommended) warm the HF cache into the bind-mounted ./models dir so the
#    big download doesn't happen inside the timed --train-timeout window.
#    ./models is bind-mounted to /root/.cache/huggingface (docker-compose.yaml).
pip install -U "huggingface_hub[cli]"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --local-dir-use-symlinks False --cache-dir ./models

# 5. (First run only) let ./scalarlm up nvidia build the image once, to pay the
#    CUDA compile up front rather than inside the sweep's first restart.
#    On B200 this is where an sm_100 build failure would surface — check here.
SCALARLM_MODEL=Qwen/Qwen2.5-0.5B ./scalarlm up nvidia   # Ctrl-C once /health is up

# 6. Re-enable the Mixtral entry (it's commented out in finetune-sweep.yaml
#    because it hangs the *Spark* — under cuda-vast it's the whole point).
#    Uncomment the `- id: mistralai/Mixtral-8x7B-Instruct-v0.1` block.

# 7. Run the sweep on the box, targeting cuda-vast
python3 test/finetune_sweep/run_finetune_sweep.py \
    --target cuda-vast \
    --models mistralai/Mixtral-8x7B-Instruct-v0.1
```

Add `--train-timeout <seconds>` if the model is large and the cache is cold.
Drop `--models` to sweep every model in the manifest on this target.

## Teardown

```bash
docker compose -f docker-compose.yaml down
```

Then **destroy the vast.ai instance** from the console — billing is per-hour and
the box holds nothing you need once results are copied off (`./jobs` and the
sweep's results dir).

## Cost expectation

A large-model end-to-end run (restart → train 450 steps → serve → check
memorization) is ≤ 1 GPU-hr on H200/B200. So:

- Unblock Mixtral-47B on 1× H200 (~$3.82/hr): **~$4**.
- Sweep the 32–72B tier (4–5 models) on 1× B200 (~$4.95/hr): **~$25**.

Rent on-demand, run the subset, destroy. No reservation needed.

## Gotchas

- **Don't run `cuda-vast` from your laptop via `DOCKER_HOST=ssh://`.** The compose
  file bind-mounts `./vllm`, `./ml`, `./jobs`, `./models`, which resolve on the
  *remote* filesystem — so the repo must be on the box anyway. Running the runner
  on the box is strictly simpler.
- **Don't run the re-enabled Mixtral entry on `cuda-spark`.** It dies at
  train-load and hangs the Spark (that's why it ships commented out). It's only
  safe on a ≥ 141 GiB card via `cuda-vast`.
- **B200 `sm_100` ≠ Spark `sm_120`.** A recompile that succeeds on the GB10 does
  not prove B200 support. The ~36 min build failing at launch is the failure
  mode — validate with step 5 before a long run.
