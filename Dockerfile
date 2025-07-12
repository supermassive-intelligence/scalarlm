ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:25.06-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv slurm-wlm libslurm-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV --system-site-packages
RUN . $VIRTUAL_ENV/bin/activate

ARG MAX_JOBS=8

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=$PATH:/opt/hpcx/ompi/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hpcx/ompi/lib

# Use the NVIDIA PyTorch version (already 2.8.0-based) from the base image
ARG TORCH_VERSION="2.8.0-nv25.6"

# Set Blackwell architecture for compilation - 10.0 for Blackwell
ENV TORCH_CUDA_ARCH_LIST="12.0"
ARG TORCH_CUDA_ARCH_LIST="12.0"
# Force vLLM to use XFormers instead of FlashAttention for Blackwell compatibility
ENV VLLM_ATTENTION_BACKEND=TORCH_SPDA

# Set up CUDA environment for Blackwell compilation
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDA_CUDA_LIBRARY="/usr/local/cuda/lib64/stubs/libcuda.so"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN pip install uv
RUN uv pip install ninja

# Compile xformers with Blackwell support
RUN git clone --branch v0.0.28.post1 https://github.com/facebookresearch/xformers.git && \
    cd xformers && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} pip install . --no-deps

# Disable FlashAttention for Blackwell compatibility
ENV VLLM_ATTENTION_BACKEND=XFORMERS

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=nvidia

###############################################################################
# CPU BASE IMAGE
FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y python3 python3-pip python3-venv \
    openmpi-bin libopenmpi-dev libpmix-dev slurm-wlm libslurm-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG MAX_JOBS=4

ARG TORCH_VERSION="2.4.0"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
FROM gdiamos/rocm-base:v0.94 AS amd
ARG MAX_JOBS=8

ENV BASE_NAME=amd

RUN pip install pyhip>=1.1.0
ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

###############################################################################
# VLLM BUILD STAGE
FROM ${BASE_NAME} AS vllm

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt
COPY ./test/requirements-pytest.txt ${INSTALL_ROOT}/requirements-pytest.txt
COPY ./infra/requirements-vllm-build.txt ${INSTALL_ROOT}/requirements-vllm-build.txt

RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm-build.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-pytest.txt

WORKDIR ${INSTALL_ROOT}

COPY ./infra/cray_infra/vllm ${INSTALL_ROOT}/infra/cray_infra/vllm
COPY ./infra/setup.py ${INSTALL_ROOT}/infra/cray_infra/setup.py

COPY ./infra/CMakeLists.txt ${INSTALL_ROOT}/infra/cray_infra/CMakeLists.txt
COPY ./infra/cmake ${INSTALL_ROOT}/infra/cray_infra/cmake
COPY ./infra/csrc ${INSTALL_ROOT}/infra/cray_infra/csrc

COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/infra/cray_infra/requirements.txt

WORKDIR ${INSTALL_ROOT}/infra/cray_infra

ARG VLLM_TARGET_DEVICE=cpu

# Fix missing CUDA library by creating symbolic link
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

# Build vllm python package with Blackwell support for NVIDIA builds
RUN \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0"} \
    VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} \
    python ${INSTALL_ROOT}/infra/cray_infra/setup.py bdist_wheel && \
    pip install ${INSTALL_ROOT}/infra/cray_infra/dist/*.whl && \
    rm -rf ${INSTALL_ROOT}/infra/cray_infra/dist

WORKDIR ${INSTALL_ROOT}

###############################################################################
# MAIN IMAGE
FROM vllm AS infra

# Build GPU-aware MPI
COPY ./infra/cray_infra/training/gpu_aware_mpi ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi
RUN python3 ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi/setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

RUN apt-get update -y  \
    && apt-get install -y build-essential \
    less curl wget net-tools vim iputils-ping strace gdb \
    && rm -rf /var/lib/apt/lists/*

# Setup python path
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/test"

RUN mkdir -p ${INSTALL_ROOT}/jobs
RUN mkdir -p ${INSTALL_ROOT}/nfs

# Copy the rest of the platform code
COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

# Build SLURM plugin
RUN /app/cray/infra/slurm_src/compile.sh

ENV LD_LIBRARY_PATH="${PYTHONPATH}:/usr/local/lib/slurm"

ENV SLURM_CONF=${INSTALL_ROOT}/nfs/slurm.conf

