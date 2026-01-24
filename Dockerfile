ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:25.10-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv slurm-wlm libslurm-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV --system-site-packages && \
    . $VIRTUAL_ENV/bin/activate

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=$PATH:/opt/hpcx/ompi/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/opt/hpcx/ompi/lib

ARG TORCH_VERSION="2.9.1"
ARG TORCH_CUDA_ARCH_LIST="7.5"

RUN pip install uv && \
    uv pip install ninja && \
    pip install --upgrade "protobuf>=6.30.0"

ENV PIP_CONSTRAINT=""

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=nvidia

ENV TORCHINDUCTOR_MAX_AUTOTUNE=0
ENV TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
ENV TORCH_COMPILE_DISABLE=1

###############################################################################
# CPU BASE IMAGE
FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y python3 python3-pip python3-venv \
    openmpi-bin libopenmpi-dev libpmix-dev slurm-wlm libslurm-dev \
    cmake

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate

ARG TORCH_VERSION="2.7.1"

RUN pip install uv && \
    uv pip install torch==${TORCH_VERSION}+cpu --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install ninja

# Put torch on the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/app/.venv/lib64/python3.12/site-packages/torch/lib

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
FROM gdiamos/rocm-base:v0.997 AS amd

ENV BASE_NAME=amd

RUN pip install pyhip>=1.1.0
ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/venv/lib/python3.12/site-packages/torch/lib:/usr/local/rdma-lib

###############################################################################
# FRONTEND BUILD STAGE

FROM ${BASE_NAME} AS ui_base

RUN apt-get update -y && \
    apt-get install -y git curl libgomp1 libcurl4 dnsutils nano

# Install node 24.0
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && \
    apt-get install -y nodejs && \
    node --version && \
    npm --version

ARG INSTALL_ROOT=/app
WORKDIR /app

# Configure Huggingface Chat UI source - can use either local directory or remote repo
ARG UI_SOURCE=remote
ARG UI_BRANCH=main
ARG UI_REPO=https://github.com/supermassive-intelligence/chat-ui-fork.git

# Handle Chat UI source - support both local and remote modes
COPY scripts/build-copy-chat-ui.sh ${INSTALL_ROOT}/build-copy-chat-ui.sh

# Handle Chat UI source - single RUN command with conditional mount
# For remote: clone from repository
# For local: mount and copy from ./chat-ui directory
RUN --mount=type=bind,source=./chat-ui,target=/workspace/chat-ui,rw \
    bash ${INSTALL_ROOT}/build-copy-chat-ui.sh ${UI_SOURCE} ${INSTALL_ROOT}/chat-ui \
    /workspace/chat-ui ${UI_REPO} ${UI_BRANCH}

# install dotenv-cli
RUN npm install -g dotenv-cli

USER root

# mkdir for ui and adjust ownership
RUN mkdir -p /app/ui && \
    touch /app/ui/.env.local && \
    cp ${INSTALL_ROOT}/chat-ui/.env /app/ui/.env && \
    cp ${INSTALL_ROOT}/chat-ui/entrypoint.sh /app/ui/entrypoint.sh && \
    cp ${INSTALL_ROOT}/chat-ui/package.json /app/ui/package.json && \
    cp ${INSTALL_ROOT}/chat-ui/package-lock.json /app/ui/package-lock.json && \
    chmod +x /app/ui/entrypoint.sh

FROM node:24 AS ui_builder

WORKDIR /app
ARG INSTALL_ROOT=/temp

USER root
RUN \
    apt-get update -y \
    && apt-get install -y git

# Configure Huggingface Chat UI source - can use either local directory or remote repo
ARG UI_SOURCE=remote
ARG UI_BRANCH=main
ARG UI_REPO=https://github.com/supermassive-intelligence/chat-ui-fork.git

# Handle Chat UI source - support both local and remote modes
COPY scripts/build-copy-chat-ui.sh ${INSTALL_ROOT}/build-copy-chat-ui.sh

# Handle Chat UI source - single RUN command with conditional mount
# For remote: clone from repository
# For local: mount and copy from ./chat-ui directory
RUN --mount=type=bind,source=./chat-ui,target=/workspace/chat-ui,rw \
    bash ${INSTALL_ROOT}/build-copy-chat-ui.sh ${UI_SOURCE} ${INSTALL_ROOT}/chat-ui \
    /workspace/chat-ui ${UI_REPO} ${UI_BRANCH}

RUN cp ${INSTALL_ROOT}/chat-ui/package-lock.json ${INSTALL_ROOT}/chat-ui/package.json ./

ARG APP_BASE=
ARG PUBLIC_APP_COLOR=
ENV BODY_SIZE_LIMIT=15728640

RUN --mount=type=cache,target=/app/.npm \
    npm set cache /app/.npm && \
    npm ci

RUN cp -R ${INSTALL_ROOT}/chat-ui/. /app/ && \
    npm install -D @sveltejs/adapter-static

RUN git config --global --add safe.directory /app && \
    npm run build

# mongo image
FROM mongo:7 AS mongo

# image to be used if INCLUDE_DB is true
FROM ui_base AS local_db

# copy mongo from the other stage
COPY --from=mongo /usr/bin/mongo* /usr/bin/

ENV MONGODB_URL=mongodb://localhost:27017
USER root
RUN mkdir -p /data/db

# final image
FROM local_db AS ui_final

# build arg to determine if the database should be included
ENV INCLUDE_DB=true

# svelte requires APP_BASE at build time so it must be passed as a build arg
ARG APP_BASE=
ARG PUBLIC_APP_COLOR=
ARG PUBLIC_COMMIT_SHA=
ENV PUBLIC_COMMIT_SHA=${PUBLIC_COMMIT_SHA}
ENV BODY_SIZE_LIMIT=15728640

#import the build & dependencies
COPY --from=ui_builder /app/build /app/build
COPY --from=ui_builder /app/node_modules /app/node_modules
COPY frontend/entrypoint.sh /app/ui/entrypoint.sh
COPY frontend/.env.local /app/ui/.env.local

#CMD ["/bin/bash", "-c", "/app/entrypoint.sh"]

###############################################################################
# VLLM BUILD STAGE
FROM ui_final AS vllm

# Copy all of the frontend libraries and code
COPY --from=ui_final --chown=1000 /app /app/ui

# Copy all of the mongo binaries
COPY --from=ui_final /usr/bin/mongo* /usr/bin/

# Set environment variables from ui
ENV MONGODB_URL=mongodb://localhost:27017
ENV INCLUDE_DB=true

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl git ccache vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

WORKDIR ${INSTALL_ROOT}

# Install build dependencies FIRST
RUN pip install setuptools-scm

# Configure vLLM source - can use either local directory or remote repo
ARG VLLM_SOURCE=remote
ARG VLLM_BRANCH=main
ARG VLLM_REPO=https://github.com/supermassive-intelligence/vllm-fork.git

# Handle vLLM source - support both local and remote modes
COPY scripts/build-copy-vllm.sh ${INSTALL_ROOT}/build-copy-vllm.sh

# Handle vLLM source - single RUN command with conditional mount
# For remote: clone from repository
# For local: mount and copy from ./vllm directory
RUN --mount=type=bind,source=./vllm,target=/workspace/vllm,rw \
    bash ${INSTALL_ROOT}/build-copy-vllm.sh ${VLLM_SOURCE} ${INSTALL_ROOT}/vllm \
    /workspace/vllm ${VLLM_REPO} ${VLLM_BRANCH}

WORKDIR ${INSTALL_ROOT}/vllm

# Set build environment variables for CPU compilation
ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG VLLM_TARGET_DEVICE=cpu

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}
ENV CMAKE_BUILD_TYPE=Release

# vLLM dependencies
COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/requirements-vllm.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm.txt && \
    python ${INSTALL_ROOT}/vllm/use_existing_torch.py

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/app/cray/vllm/.deps \
    export MAX_JOBS=$(($(nproc) < $(free -g | awk '/^Mem:/ {print int($2/4)}') ? $(nproc) : $(free -g | awk '/^Mem:/ {print int($2/4)}'))) && \
    pip install --no-build-isolation -e . --verbose

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
    less curl wget net-tools vim iputils-ping strace gdb python3-dbg python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup python path
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/test"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/vllm"

# Megatron dependencies (GPU only)
# note this has to happen after vllm because it overrides some packages installed by vllm
COPY ./infra/requirements-megatron.txt ${INSTALL_ROOT}/requirements-megatron.txt
COPY ./infra/requirements-megatron-cpu.txt ${INSTALL_ROOT}/requirements-megatron-cpu.txt
COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt

RUN if [ "$VLLM_TARGET_DEVICE" != "cpu" ]; then \
        uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron.txt; \
    fi && \
    if [ "$VLLM_TARGET_DEVICE" != "cuda" ]; then \
        uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron-cpu.txt; \
    fi && \
    uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt

RUN mkdir -p ${INSTALL_ROOT}/jobs ${INSTALL_ROOT}/nfs

# Copy the rest of the platform code
COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

WORKDIR ${INSTALL_ROOT}

# Build SLURM plugin
RUN /app/cray/infra/slurm_src/compile.sh

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYTHONPATH}:/usr/local/lib/slurm
ENV SLURM_CONF=${INSTALL_ROOT}/nfs/slurm.conf
ENV VLLM_CPU_MOE_PREPACK=0


