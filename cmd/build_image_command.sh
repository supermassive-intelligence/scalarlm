inspect_args

target=${args[target]}
sm_arch=${args[sm_arch]}

declare -a vllm_target_device
declare -a docker_platform

# If target is cpu, build the image with the cpu base image
if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    if [ "$(uname -m)" == "x86_64" ]; then
        docker_platform=("linux/amd64")
    else
        docker_platform=("linux/arm64/v8")
    fi
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_platform=("linux/amd64")
    sm_arch="gfx942"
elif [ "$target" == "spark" ]; then
    # NVIDIA DGX Spark: aarch64 Grace CPU + Blackwell GPU (SM 12.0).
    vllm_target_device=("cuda")
    docker_platform=("linux/arm64")
    if [ "$sm_arch" == "auto" ]; then
        sm_arch="12.0"
    fi
else
    vllm_target_device=("cuda")
    docker_platform=("linux/amd64")
    if [ "$sm_arch" == "auto" ]; then
        # Auto-detect the architecture of every GPU using nvidia-smi, deduped
        # and space-joined into a single TORCH_CUDA_ARCH_LIST-compatible string.
        # (A bash array here would silently collapse to element 0 below.)
        sm_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd' ')
    fi
fi

# TORCH_CUDA_ARCH_LIST can be a space-separated list (multi-GPU auto-detect);
# it must stay single-quoted here so the eval below re-parses it as one
# --build-arg token instead of splitting it into extra stray arguments.
docker_build_command="docker build --platform ${docker_platform} --build-arg BASE_NAME=${target} --build-arg TORCH_CUDA_ARCH_LIST='${sm_arch}' --build-arg VLLM_TARGET_DEVICE=${vllm_target_device} -t cray:latest ."

mkdir -p vllm
mkdir -p chat-ui

# Run docker build command
echo $(green_bold Building image with command: ${docker_build_command})
eval $docker_build_command

echo $(green_bold Successfully built image)
