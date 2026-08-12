#!/bin/bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: run_live_server_tests.sh --tag IMAGE --model MODEL --target TARGET --timeout SECONDS

Starts a temporary ScalarLM server, waits for API and vLLM readiness, runs the
live inference smoke tests from a second container, and always tears down the
temporary Docker resources.
EOF
}

tag=""
model=""
target=""
readiness_timeout=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag)
            tag=$2
            shift 2
            ;;
        --model)
            model=$2
            shift 2
            ;;
        --target)
            target=$2
            shift 2
            ;;
        --timeout)
            readiness_timeout=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$tag" ] || [ -z "$model" ] || [ -z "$target" ] || [ -z "$readiness_timeout" ]; then
    usage >&2
    exit 2
fi

if ! [[ "$readiness_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "--timeout must be a positive integer" >&2
    exit 2
fi

case "$target" in
    cpu|nvidia|amd|spark) ;;
    *)
        echo "Unsupported target: $target" >&2
        exit 2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for the live-server profile" >&2
    exit 1
fi

if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "Docker image not found: $tag" >&2
    echo "Build it first or omit --no-build when invoking ./scalarlm test." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
run_id="${BASHPID:-$$}-$(date +%s)"
server_name="scalarlm-live-server-$run_id"
network_name="scalarlm-live-network-$run_id"
model_cache="${SCALARLM_MODEL_CACHE:-$REPO_ROOT/models}"

mkdir -p "$model_cache"

cleanup() {
    docker rm -f "$server_name" >/dev/null 2>&1 || true
    docker network rm "$network_name" >/dev/null 2>&1 || true
}

dump_server_logs() {
    echo "---- ScalarLM live-server logs (last 400 lines) ----" >&2
    docker logs --tail 400 "$server_name" >&2 || true
    echo "---- end ScalarLM live-server logs ----" >&2
}

trap cleanup EXIT
trap 'exit 130' INT TERM

docker network create "$network_name" >/dev/null

declare -a device_args=()
case "$target" in
    nvidia|spark)
        device_args=("--gpus" "all")
        ;;
    amd)
        device_args=(
            "--device=/dev/kfd"
            "--device=/dev/dri"
            "--security-opt" "seccomp=unconfined"
        )
        ;;
esac

echo "Starting $server_name from $tag with model $model ($target)"
docker run --detach \
    --name "$server_name" \
    --network "$network_name" \
    --init \
    --shm-size=8g \
    "${device_args[@]}" \
    -e "PYTHONDONTWRITEBYTECODE=1" \
    -e "SCALARLM_ENABLE_LORA=false" \
    -e "SCALARLM_ENABLE_TOKENFORMER=true" \
    -e "SCALARLM_MODEL=$model" \
    -v "$model_cache:/root/.cache/huggingface" \
    -v "$REPO_ROOT/infra/cray_infra:/app/cray/infra/cray_infra:ro" \
    -v "$REPO_ROOT/scripts:/app/cray/scripts:ro" \
    -v "$REPO_ROOT/ml:/app/cray/ml:ro" \
    -v "$REPO_ROOT/test:/app/cray/test:ro" \
    "$tag" \
    /app/cray/scripts/start_one_server.sh >/dev/null

echo "Waiting up to ${readiness_timeout}s for API and vLLM readiness"
started_at=$(date +%s)
last_progress=0
while true; do
    now=$(date +%s)
    elapsed=$((now - started_at))

    if [ "$elapsed" -ge "$readiness_timeout" ]; then
        echo "Timed out waiting for the live server after ${elapsed}s" >&2
        dump_server_logs
        exit 1
    fi

    running=$(docker inspect --format '{{.State.Running}}' "$server_name" 2>/dev/null || true)
    if [ "true" != "$running" ]; then
        echo "Live-server container exited before becoming ready" >&2
        dump_server_logs
        exit 1
    fi

    health=$(
        docker exec "$server_name" \
            curl --fail --silent --show-error --max-time 5 \
            http://127.0.0.1:8000/v1/health 2>/dev/null || true
    )
    if printf '%s' "$health" | grep -Eq '"api"[[:space:]]*:[[:space:]]*"up"' && \
       printf '%s' "$health" | grep -Eq '"vllm"[[:space:]]*:[[:space:]]*"up"'; then
        echo "Live server ready after ${elapsed}s: $health"
        break
    fi

    if [ $((elapsed - last_progress)) -ge 15 ]; then
        echo "Still waiting (${elapsed}s); latest health: ${health:-unavailable}"
        last_progress=$elapsed
    fi
    sleep 2
done

if ! docker run --rm --init \
    --network "$network_name" \
    -e "PYTHONDONTWRITEBYTECODE=1" \
    -e "SCALARLM_LIVE_URL=http://$server_name:8000" \
    -e "SCALARLM_LIVE_MODEL=$model" \
    -v "$REPO_ROOT/infra/cray_infra:/app/cray/infra/cray_infra:ro" \
    -v "$REPO_ROOT/sdk:/app/cray/sdk:ro" \
    -v "$REPO_ROOT/test:/app/cray/test:ro" \
    -v "$REPO_ROOT/pytest.ini:/app/cray/pytest.ini:ro" \
    --entrypoint sh \
    "$tag" -c \
    'pip install --quiet --disable-pip-version-check -r test/requirements-pytest.txt && python -m pytest -p no:cacheprovider test/live/test_server_smoke.py -vv -rA'; then
    dump_server_logs
    exit 1
fi

echo "Live-server smoke tests passed."
