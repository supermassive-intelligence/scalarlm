#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# expandable_segments uses growable virtual address ranges so freed
# blocks of one size can satisfy a later allocation of a different
# size — without it, gradient checkpointing's recompute pattern
# fragments the caching allocator and reserved memory grows step
# over step (especially on Gemma-4-class models with alternating
# sliding/full-attention activation shapes). PyTorch 2.1+.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"

# Put the current ml directory in the python path so that the modules can be imported
export PYTHONPATH=$LOCAL_DIRECTORY/ml:$PYTHONPATH

mpirun --allow-run-as-root python $LOCAL_DIRECTORY/ml/cray_megatron/main.py $*
