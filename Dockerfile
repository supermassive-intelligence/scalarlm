ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv slurm-wlm libslurm-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV --system-site-packages
RUN . $VIRTUAL_ENV/bin/activate

ARG MAX_JOBS=8

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=$PATH:/opt/hpcx/ompi/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/opt/hpcx/ompi/lib

ARG TORCH_VERSION="2.7.1"
ARG TORCH_CUDA_ARCH_LIST="7.5"

RUN pip install uv

RUN git clone --branch v0.0.28.post1 https://github.com/facebookresearch/xformers.git
RUN uv pip install ninja
RUN cd xformers && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} pip install . --no-deps

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
#ENV DNNL_DEFAULT_FPMATH_MODE=F32

ARG TORCH_VERSION="2.7.1"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

# Put torch on the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/app/.venv/lib64/python3.12/site-packages/torch/lib

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
FROM gdiamos/rocm-base:v0.95 AS amd
ARG MAX_JOBS=8

ENV BASE_NAME=amd

RUN pip install pyhip>=1.1.0
ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/venv/lib/python3.12/site-packages/torch/lib:/usr/local/rdma-lib

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
COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/requirements-vllm.txt

RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm-build.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-pytest.txt

WORKDIR ${INSTALL_ROOT}

# Install build dependencies FIRST
RUN pip install numpy packaging setuptools-scm wheel cmake ninja

# Configure vLLM source - can use either local directory or remote repo
ARG VLLM_SOURCE=remote
ARG VLLM_BRANCH=rschiavi/vllm-adapter
ARG VLLM_REPO=https://github.com/supermassive-intelligence/vllm.git

# Handle vLLM source - keep it simple with bind mount approach
COPY scripts/build-copy-vllm.sh ${INSTALL_ROOT}/build-copy-vllm.sh
RUN --mount=type=bind,source=.,target=/workspace \
bash ${INSTALL_ROOT}/build-copy-vllm.sh local ${INSTALL_ROOT}/vllm /workspace/vllm ${VLLM_REPO} ${VLLM_BRANCH}

# Remove torch requirements to use pre-installed PyTorch from base image
RUN cd ${INSTALL_ROOT}/vllm && python use_existing_torch.py


# Set build environment variables for CPU compilation
ARG VLLM_TARGET_DEVICE=cpu
ENV VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}
ENV CMAKE_BUILD_TYPE=Release
ENV MAX_JOBS=${MAX_JOBS}

# Build vLLM from source with CPU target  
WORKDIR ${INSTALL_ROOT}/vllm

# Set fallback version for setuptools-scm in case git metadata is missing
# This handles cases where git history might be incomplete
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.6.3.post1

RUN echo "Building vLLM for CPU with target device: ${VLLM_TARGET_DEVICE}" && \
    echo "Max jobs: ${MAX_JOBS}" && \
    VLLM_TARGET_DEVICE=cpu python setup.py build_ext --inplace && \
    VLLM_TARGET_DEVICE=cpu pip install -e . --verbose

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
# Add vLLM clone directly to Python path (primary method)
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/vllm"

RUN mkdir -p ${INSTALL_ROOT}/jobs
RUN mkdir -p ${INSTALL_ROOT}/nfs

# Copy the rest of the platform code
COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts



COPY ./pyproject.toml ${INSTALL_ROOT}/pyproject.toml
COPY ./setup.py ${INSTALL_ROOT}/setup.py

WORKDIR ${INSTALL_ROOT}
# Fix setuptools-scm version detection in Docker
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.2.0

# Set vLLM environment variables for CPU mode
ENV VLLM_TARGET_DEVICE=cpu
ENV VLLM_USE_V1=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_LOGGING_LEVEL=DEBUG
# https://github.com/vllm-project/vllm-ascend/issues/1048
ENV VLLM_PLUGINS=ascend,ascend_enhanced_model
# For CPU mode, don't set CUDA_VISIBLE_DEVICES at all since there are no CUDA devices
# ENV CUDA_VISIBLE_DEVICES="0"
RUN pip install -e .[training]

# Build SLURM plugin
RUN /app/cray/infra/slurm_src/compile.sh

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYTHONPATH}:/usr/local/lib/slurm

ENV SLURM_CONF=${INSTALL_ROOT}/nfs/slurm.conf

