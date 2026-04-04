#!/usr/bin/env bash
# docker-entrypoint.sh
#
# Launches mpirun with NUM_GPUS ranks, forwarding CUDA_DEVICE_IDS to the
# test script so each rank can pin itself to the right physical GPU.
#
# Environment variables:
#   NUM_GPUS        — number of MPI ranks (default: 2)
#   CUDA_DEVICE_IDS — comma-separated CUDA device IDs, one per rank
#                     (default: "0,1,...,NUM_GPUS-1")
#   ITERATIONS      — number of test iterations per collective (default: 10)

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"

# Build a default device ID list if not provided: "0,1,...,N-1"
if [ -z "${CUDA_DEVICE_IDS:-}" ]; then
    ids=""
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        ids="${ids}${i},"
    done
    CUDA_DEVICE_IDS="${ids%,}"   # strip trailing comma
fi

ITERATIONS="${ITERATIONS:-10}"

echo "============================================================"
echo " gpu_aware_mpi test"
echo "  NUM_GPUS       = ${NUM_GPUS}"
echo "  CUDA_DEVICE_IDS= ${CUDA_DEVICE_IDS}"
echo "  ITERATIONS     = ${ITERATIONS}"
echo "============================================================"

# Use the HPC-X mpirun that matches the OpenMPI setup.py builds against.
MPIRUN=/opt/hpcx/ompi/bin/mpirun

# --allow-run-as-root   : required inside Docker (commonly runs as root)
# --mca pml ucx         : use UCX for point-to-point (GPU-aware RDMA / shm)
# --mca btl ^vader      : disable CMA/vader; our own /dev/shm channel
#                         handles intra-node transfers, so let UCX route them
# -x CUDA_VISIBLE_DEVICES: expose only the requested GPUs to every rank;
#                         the test script further maps rank→device via
#                         --cuda-device-ids so ranks don't all land on GPU 0
"${MPIRUN}" \
    --allow-run-as-root \
    -np "${NUM_GPUS}" \
    --mca pml ucx \
    --mca btl ^vader \
    -x CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS}" \
    python3 /build/gpu_aware_mpi/test_gpu_aware_mpi.py \
        --cuda-device-ids "${CUDA_DEVICE_IDS}" \
        --iterations "${ITERATIONS}"
