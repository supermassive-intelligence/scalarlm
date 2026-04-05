# ============================================================
# gpu_aware_mpi test image
#
# Directory structure:
#   gpu_aware_mpi/
#   ├── setup.py
#   ├── test_gpu_aware_mpi.py
#   ├── gpu_aware_mpi.cpp
#   ├── shm_channel.cpp / .h
#   ├── shm_transport.cpp / .h
#   ├── mpi_transport.cpp / .h
#   └── common.h
#   Dockerfile
#   docker-entrypoint.sh
#
# Build:
#   docker build -t gpu_aware_mpi_test .
#
# docker run --gpus all --shm-size=8g --ipc=host --rm  --ulimit memlock=-1:-1   -e NUM_GPUS=2   gpu_aware_mpi_test
#
#
# Extra options (passed as env vars):
#   ITERATIONS   — test iterations per collective (default 10)
# ============================================================

FROM nvcr.io/nvidia/pytorch:25.04-py3

# ---- build mpi4py against the HPC-X OpenMPI already in the image ----
# The nvcr pytorch image ships HPC-X at /opt/hpcx.  Build mpi4py from
# source so it links against that exact OpenMPI rather than a system one.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpmi2-0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN MPICC=/opt/hpcx/ompi/bin/mpicc \
    pip install --no-cache-dir --no-binary mpi4py mpi4py

# ---- copy the source folder ----
# setup.py resolves sources and include_dirs relative to cwd, both now
# pointing at the top-level gpu_aware_mpi/ directory.
WORKDIR /build

COPY gpu_aware_mpi/ infra/cray_infra/training/gpu_aware_mpi/gpu_aware_mpi/

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"
ENV FORCE_CUDA=1


# Copy setup.py to /build and run from there
RUN cp infra/cray_infra/training/gpu_aware_mpi/gpu_aware_mpi/setup.py . \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl \
    && rm -rf dist build *.egg-info

# ---- entrypoint ----
COPY ./docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
