inspect_args

target=${args[target]}

declare -a vllm_target_device
declare -a docker_compose_service
declare -a docker_platform

if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    docker_compose_service="cray"
    docker_platform="linux/arm64"
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_compose_service="cray-amd"
    docker_platform="linux/amd64"
else
    vllm_target_device=("cuda")
    docker_compose_service="cray-nvidia"
    docker_platform="linux/amd64"
fi

BASE_NAME=${target} VLLM_TARGET_DEVICE=${vllm_target_device} DOCKER_PLATFORM=${docker_platform} docker compose -f docker-compose.yaml up ${docker_compose_service} --build --force-recreate
