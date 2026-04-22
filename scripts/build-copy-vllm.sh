#!/bin/bash
# Script used by Docker build process to handle vLLM source:
#   - either copy local or clone remote
set -e

VLLM_SOURCE=$1
DEST_DIR=$2
LOCAL_PATH=$3
REPO_URL=$4
BRANCH=$5

echo "🔧 Setting up vLLM source..."
echo "   Source type: $VLLM_SOURCE"

if [ "$VLLM_SOURCE" = "local" ]; then
    echo "📁 Using local vLLM from: $LOCAL_PATH"

    if [ ! -d "$LOCAL_PATH" ]; then
        echo "❌ Error: Local vLLM directory not found at $LOCAL_PATH"
        echo ""
        echo "   For local development, vLLM must be cloned into the ScalarLM directory:"
        echo "   cd /path/to/scalarlm"
        echo "   git clone -b main https://github.com/supermassive-intelligence/vllm-fork.git vllm"
        echo ""
        echo "   This will create: scalarlm/vllm/"
        exit 1
    fi

    echo "📋 Copying local vLLM to $DEST_DIR..."
    cp -r "$LOCAL_PATH" "$DEST_DIR"

    # Copy only essential git metadata for version detection (not objects)
    rm -rf $DEST_DIR/.git

    # Create a minimal but valid git repository
    cd "$DEST_DIR"
    git init
    git config user.name "Docker Build"
    git config user.email "build@docker.local"
    git add -A
    git commit -m "Build snapshot" --no-verify
    git tag -a v0.6.5 -m "Version 0.6.5"
    cd -
    echo "📌 Keeping git metadata for version detection"

    echo "✅ Local vLLM copied successfully"

else
    echo "🌐 Cloning vLLM from remote repository"
    echo "   Repository: $REPO_URL"
    echo "   Branch: $BRANCH"

    git clone -b "$BRANCH" "$REPO_URL" "$DEST_DIR"

    echo "✅ Remote vLLM cloned successfully"
fi

echo "📍 vLLM is ready at: $DEST_DIR"
ls -la "$DEST_DIR" | head -5

# ScalarLM fork patches (see scripts/vllm_patches/apply_patches.py for why
# each one exists and what anchors it's guarding). Runs after the vLLM
# tree is staged, before the pip install compiles anything.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="${SCRIPT_DIR}/vllm_patches/apply_patches.py"
if [ -x "${PATCHER}" ]; then
    echo "🩹 Applying ScalarLM vLLM-fork patches"
    python3 "${PATCHER}" "${DEST_DIR}"
else
    echo "⚠️  Patcher not found at ${PATCHER}; skipping"
fi
