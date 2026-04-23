inspect_args

target=${args[target]}
sm_arch=${args[sm_arch]}

declare -a vllm_target_device
declare -a docker_compose_service
declare -a docker_platform

if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    docker_compose_service="cray"
    if [ "$(uname -m)" == "x86_64" ]; then
        docker_platform=("linux/amd64")
    else
        docker_platform=("linux/arm64/v8")
    fi
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_compose_service="cray-amd"
    docker_platform="linux/amd64"
    sm_arch="gfx942"
elif [ "$target" == "mlx" ]; then
    # MLX target - native execution on Apple Silicon (no Docker)
    if [ "$(uname)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
        echo "Error: MLX target requires Apple Silicon Mac"
        exit 1
    fi

    # Setup Python venv
    VENV_DIR=".venv-mlx"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating Python virtual environment: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    # Ensure pip is up to date to support pyproject.toml correctly
    echo "Upgrading pip, setuptools, and wheel..."
    pip install -q --upgrade pip setuptools wheel

    # Uninstall any existing vllm/vllm-mlx to ensure clean state
    pip uninstall -y vllm vllm-mlx 2>/dev/null || true

    # Install vllm (MLX version)
    VLLM_SOURCE_PATH=""
    if [ -d "vllm-mlx" ]; then
        VLLM_SOURCE_PATH="./vllm-mlx"
        echo "Installing vLLM from local MLX source: vllm-mlx/"
    elif [ -d "../vllm-mlx" ]; then
        VLLM_SOURCE_PATH="../vllm-mlx"
        echo "Installing vLLM from parent MLX source: ../vllm-mlx/"
    fi

    if [ -n "$VLLM_SOURCE_PATH" ]; then
        pip install "$VLLM_SOURCE_PATH"
        echo "✅ Installed vLLM from MLX source"
    else
        VLLM_MLX_REPO="${VLLM_MLX_REPO:-https://github.com/funston/vllm-mlx.git}"
        VLLM_MLX_BRANCH="${VLLM_MLX_BRANCH:-feature/scalarlm}"
        echo "Installing vLLM from MLX repo: ${VLLM_MLX_REPO}@${VLLM_MLX_BRANCH}"
        pip install "git+${VLLM_MLX_REPO}@${VLLM_MLX_BRANCH}"
        echo "✅ Installed vLLM from MLX repo"
    fi

    # Install MLX-specific packages (includes numpy>=2.0.0 for MLX)
    echo "Installing MLX-specific packages..."
    pip install -q mlx mlx-lm 'numpy>=2.0.0'

    # Install base requirements (exclude mpi4py - not needed for inference)
    echo "Installing ScalarLM dependencies (excluding mpi4py)"
    grep -v "mpi4py" requirements.txt | pip install -q -r /dev/stdin

    # Set env vars for native execution
    export SCALARLM_NATIVE_EXECUTION=true
    export SCALARLM_INFERENCE_ONLY=true
    export SCALARLM_SERVER_LIST="api,vllm"

    # Create necessary directories
    mkdir -p jobs nfs/logs models inference_requests

    # Start ScalarLM natively (no Docker)
    echo "Starting ScalarLM (native execution)"
    cd infra
    python3 -m cray_infra.one_server.main
    exit 0
elif [ "$target" == "spark" ]; then
    # NVIDIA DGX Spark: aarch64 Grace CPU + Blackwell GPU (SM 12.0).
    vllm_target_device=("cuda")
    docker_compose_service="cray-spark"
    docker_platform="linux/arm64"
    if [ "$sm_arch" == "auto" ]; then
        sm_arch="12.0"
    fi
else
    vllm_target_device=("cuda")
    docker_compose_service="cray-nvidia"
    docker_platform="linux/amd64"
    if [ "$sm_arch" == "auto" ]; then
        echo "Autodetect sm_arch"
        # Auto-detect the architecture of the GPU using nvidia-smi
        sm_arch=($(nvidia-smi --query-gpu=compute_cap --format=csv,noheader))
    fi
fi

mkdir -p models
mkdir -p vllm
mkdir -p chat-ui

echo "SM arch is ${sm_arch}"

BASE_NAME=${target} VLLM_TARGET_DEVICE=${vllm_target_device} \
    DOCKER_PLATFORM=${docker_platform} TORCH_CUDA_ARCH_LIST=${sm_arch} \
    docker compose -f docker-compose.yaml up ${docker_compose_service} --build --force-recreate
