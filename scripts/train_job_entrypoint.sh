#!/bin/bash

# Thin shell wrapper: set env vars then hand off to the Python
# entrypoint. Per-slice signal forwarding, checkpoint-on-timeout
# coordination, and relaunch logic live in
# ml/cray_megatron/training_entrypoint.py so they can stay readable
# and testable.

set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# expandable_segments uses growable virtual address ranges so freed
# blocks of one size can satisfy a later allocation of a different
# size — without it, gradient checkpointing's recompute pattern
# fragments the caching allocator and reserved memory grows step
# over step (especially on Gemma-4-class models with alternating
# sliding/full-attention activation shapes). PyTorch 2.1+.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${LOCAL_DIRECTORY}/ml:${PYTHONPATH:-}"

exec python "${LOCAL_DIRECTORY}/ml/cray_megatron/training_entrypoint.py" "$@"
