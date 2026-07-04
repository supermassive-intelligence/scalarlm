#!/bin/bash
#
# provision-vast.sh — prepare a freshly-rented vast.ai GPU box to run the
# finetune sweep's `cuda-vast` target (see docs/runbooks/finetune-sweep-on-vast-ai.md).
#
# Run this ON the rented box (the sweep's Compose path drives the LOCAL docker
# daemon — there is no remote-daemon mode). It is idempotent: safe to re-run.
#
# What it does, in order:
#   1. Preflight — GPU, docker, nvidia container runtime, disk headroom.
#   2. Repo    — clone + checkout the lab branch if not already inside it.
#   3. HF auth — verify HF_TOKEN (gated repos + preflight's `docker compose run`).
#   4. Cache   — pre-download each model into ./models so the big pull isn't
#                inside the timed --train-timeout window.
#   5. Build   — pay the CUDA compile up front (and surface an sm_100/B200 build
#                failure HERE, not on the sweep's first restart).
# It does NOT run the sweep — it prints the exact command to run at the end.
#
# Usage:
#   ./scripts/provision-vast.sh [MODEL_ID ...]
# Env knobs (all optional):
#   REPO_URL   git remote to clone if not already in a checkout
#              (default: https://github.com/supermassive-intelligence/scalarlm.git)
#   BRANCH     branch to check out (default: georgi/finetune-sweep)
#   HF_TOKEN   HuggingFace token for gated repos (export before running)
#   SKIP_CACHE_WARM=1   skip step 4 (models already cached / ungated)
#   SKIP_BUILD=1        skip step 5 (image already built)
#
# Example:
#   export HF_TOKEN=hf_...
#   ./scripts/provision-vast.sh mistralai/Mixtral-8x7B-Instruct-v0.1

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/supermassive-intelligence/scalarlm.git}"
BRANCH="${BRANCH:-georgi/finetune-sweep}"
MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=("mistralai/Mixtral-8x7B-Instruct-v0.1")
fi

log()  { printf '\033[1;36m[provision-vast]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[provision-vast] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[provision-vast] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. Preflight -----------------------------------------------------------
log "Preflight checks…"
command -v docker >/dev/null   || die "docker not found. Pick a vast.ai offer with Docker daemon access (not a plain app container)."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — no GPU visible on this box."
docker compose version >/dev/null 2>&1 || die "docker compose v2 plugin not available."

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader \
    || die "nvidia-smi failed to query the GPU."

# Confirm the nvidia container runtime actually works (the stack needs runtime: nvidia).
log "Verifying the nvidia container runtime…"
if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "GPU not visible inside a container. Ensure the nvidia container runtime is installed"
    warn "and the offer exposes GPUs to nested docker. The sweep will fail without this."
fi

# Disk headroom: a 47-72B bf16 model is 90-150GiB on disk + checkpoints.
avail_gib=$(df -Pk . | awk 'NR==2{printf "%d", $4/1024/1024}')
log "Free disk in $(pwd): ${avail_gib} GiB"
[ "${avail_gib}" -lt 200 ] && warn "Under 200 GiB free — a large model download + checkpoints may not fit."

# --- 2. Repo ----------------------------------------------------------------
if [ -f "./scalarlm" ] && [ -f "./docker-compose.yaml" ]; then
    log "Already inside a scalarlm checkout: $(pwd)"
else
    log "Cloning ${REPO_URL} (branch ${BRANCH})…"
    git clone "${REPO_URL}" scalarlm
    cd scalarlm
fi
log "Checking out ${BRANCH}…"
git checkout "${BRANCH}"
git rev-parse --short HEAD | xargs -I{} log "HEAD at {}"

mkdir -p models jobs

# --- 3. HF auth -------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    log "HF_TOKEN is set (gated repos + preflight auth enabled)."
    export HF_TOKEN
else
    warn "HF_TOKEN not set — gated repos (Llama, Gemma, some Mistral) will 401 -> SKIPPED."
    warn "  export HF_TOKEN=hf_... before running if you need them."
fi

# --- 4. Warm the HF cache ---------------------------------------------------
# ./models is bind-mounted to /root/.cache/huggingface in docker-compose.yaml.
if [ "${SKIP_CACHE_WARM:-0}" = "1" ]; then
    log "SKIP_CACHE_WARM=1 — skipping model pre-download."
else
    log "Warming HF cache for: ${MODELS[*]}"
    python3 -m pip install -q -U "huggingface_hub[cli,hf_transfer]" || warn "huggingface_hub install failed; skipping warm."
    for m in "${MODELS[@]}"; do
        log "  downloading ${m} -> ./models"
        HF_HUB_ENABLE_HF_TRANSFER=1 hf download "${m}" --cache-dir ./models \
            || warn "download of ${m} failed (gated without token? typo?) — the sweep will retry inside the container."
    done
fi

# --- 5. Pre-build the image -------------------------------------------------
# `./scalarlm build nvidia` auto-detects sm_arch via nvidia-smi and warms the
# SAME build cache the sweep's `docker compose up --build` reuses, so the first
# restart is fast. On B200 (sm_100) a compile failure surfaces HERE.
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    log "SKIP_BUILD=1 — skipping image pre-build."
else
    log "Pre-building the cray-nvidia image (auto-detects sm_arch)…"
    ./scalarlm build nvidia
fi

# --- Done -------------------------------------------------------------------
log "Provisioning complete."
cat <<EOF

Next steps:
  1. If testing Mixtral, un-comment its entry in
       test/finetune_sweep/finetune-sweep.yaml
     (it ships disabled because it hangs the *Spark*; on cuda-vast it's the point).

  2. Run the sweep ON THIS BOX:
       python3 test/finetune_sweep/run_finetune_sweep.py \\
           --target cuda-vast \\
           --models ${MODELS[*]}

  3. Copy results off (./jobs + the results dir), then DESTROY the vast.ai
     instance from the console — billing is per-hour.
EOF
