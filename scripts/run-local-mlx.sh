#!/bin/bash
# Run ScalarLM locally on Apple Silicon (no Docker)

set -e

echo "🍎 Running ScalarLM natively on Apple Silicon with MLX"

# Check platform
if [ "$(uname)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "❌ Error: This script is for Apple Silicon Macs only"
    exit 1
fi

# Use whatever python3 is in PATH
PYTHON_BIN="python3"

echo "✅ Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# Create or activate clean venv
VENV_DIR=".venv-mlx"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating clean venv at $VENV_DIR..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✅ Activated venv: $VIRTUAL_ENV"

# Install ScalarLM dependencies (MLX variant without MPI)
echo "📦 Installing ScalarLM dependencies..."
if [ -f "requirements-mlx.txt" ]; then
    pip install -q -r requirements-mlx.txt
fi

# Check if vllm-mlx is installed
if ! python -c "import vllm_mlx" 2>/dev/null; then
    echo "📦 vllm-mlx not found. Installing..."

    # Check if ../vllm-mlx exists (local dev)
    if [ -d "../vllm-mlx" ]; then
        echo "Installing vllm-mlx from ../vllm-mlx (local development)"
        pip install ../vllm-mlx
    else
        echo "Installing vllm-mlx from GitHub"
        pip install git+https://github.com/waybarrios/vllm-mlx.git
    fi
fi

# Check if MLX is installed
if ! python -c "import mlx.core" 2>/dev/null; then
    echo "📦 MLX not found. Installing..."
    pip install mlx mlx-lm
fi

# Get absolute path to scalarlm directory
SCALARLM_DIR="$(cd "$(dirname "$0")" && pwd)"

# Set environment for native execution
export PYTHONPATH="${PYTHONPATH}:${SCALARLM_DIR}/infra:${SCALARLM_DIR}/sdk:${SCALARLM_DIR}/ml:${SCALARLM_DIR}/test"

# Create necessary directories
mkdir -p "${SCALARLM_DIR}/jobs" "${SCALARLM_DIR}/nfs/logs" "${SCALARLM_DIR}/models" "${SCALARLM_DIR}/inference_requests"

# Override Docker paths for native execution
export SCALARLM_LOG_DIRECTORY="${SCALARLM_DIR}/nfs/logs"
export SCALARLM_JOBS_DIR="${SCALARLM_DIR}/jobs"
export SCALARLM_TRAINING_JOB_DIRECTORY="${SCALARLM_DIR}/jobs"
export SCALARLM_INFERENCE_WORK_QUEUE_PATH="${SCALARLM_DIR}/inference_work_queue.sqlite"
export SCALARLM_UPLOAD_BASE_PATH="${SCALARLM_DIR}/inference_requests"
export SCALARLM_TRAIN_JOB_ENTRYPOINT="${SCALARLM_DIR}/scripts/train_job_entrypoint.sh"

# Configure for MLX backend
export SCALARLM_VLLM_BACKEND="mlx"
export SCALARLM_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Disable megatron/SLURM features for native execution (no SLURM installed)
export SCALARLM_SERVER_LIST="api,vllm"

echo "✅ Starting ScalarLM API server on port 8000"
echo "   Platform: Apple Silicon (MLX)"
echo "   Root: ${SCALARLM_DIR}"
echo "   Logs: ${SCALARLM_DIR}/nfs/logs"
echo "   Press Ctrl+C to stop"
echo ""

# Start vllm-mlx server first (in background)
cd "${SCALARLM_DIR}"
echo "🚀 Starting vllm-mlx server on port 8001..."
python3 -m vllm_mlx.server \
  --model "mlx-community/Qwen2.5-0.5B-Instruct-4bit" \
  --host "0.0.0.0" \
  --port 8001 \
  --max-tokens 256 &

VLLM_PID=$!
sleep 3  # Wait for vllm-mlx to start

# Run FastAPI server directly with uvicorn (bypass broken main.py)
cd "${SCALARLM_DIR}/infra"
echo "🚀 Starting ScalarLM FastAPI server on port 8000..."
uvicorn cray_infra.api.fastapi.main:app --host 0.0.0.0 --port 8000

# Clean up vllm-mlx on exit
kill $VLLM_PID 2>/dev/null
