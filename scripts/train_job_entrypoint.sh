#!/bin/bash

# Thin shell wrapper: set env vars then exec mpirun. The `exec`
# replaces the batch shell with mpirun in place — the slurm batch
# shell's PID becomes mpirun's, so slurm's `--signal=B:TERM@N`
# (sent to the batch shell) lands on mpirun directly. mpirun's
# standard SIGTERM forwarding then propagates to each rank's python
# process, whose handler in main.py sets the stop_flag. No bash trap,
# no Python wrapper — keeping the sbatch → mpirun → main.py path
# stock avoids cross-system behavior drift.

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

exec mpirun --allow-run-as-root python "${LOCAL_DIRECTORY}/ml/cray_megatron/main.py" "$@"
