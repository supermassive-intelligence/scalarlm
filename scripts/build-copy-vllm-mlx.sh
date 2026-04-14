#!/bin/bash
# Script used by Docker build process to handle vllm-mlx source:
#   - either copy local or clone remote
set -e

VLLM_MLX_SOURCE=$1
DEST_DIR=$2
LOCAL_PATH=$3
REPO_URL=$4
BRANCH=$5

echo "🔧 Setting up vllm-mlx source..."
echo "   Source type: $VLLM_MLX_SOURCE"

if [ "$VLLM_MLX_SOURCE" = "local" ]; then
    echo "📁 Using local vllm-mlx from: $LOCAL_PATH"

    if [ ! -d "$LOCAL_PATH" ]; then
        echo "❌ Error: Local vllm-mlx directory not found at $LOCAL_PATH"
        echo ""
        echo "   For local development, vllm-mlx must be cloned into the ScalarLM directory:"
        echo "   cd /path/to/scalarlm"
        echo "   git clone https://github.com/waybarrios/vllm-mlx.git vllm-mlx"
        echo ""
        echo "   Or create a symlink:"
        echo "   ln -s ../vllm-mlx ./vllm-mlx"
        echo ""
        echo "   This will create: scalarlm/vllm-mlx/"
        exit 1
    fi

    echo "📋 Copying local vllm-mlx to $DEST_DIR..."
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
    git tag -a v0.2.5 -m "Version 0.2.5"
    cd -
    echo "📌 Keeping git metadata for version detection"

    echo "✅ Local vllm-mlx copied successfully"

else
    echo "🌐 Cloning vllm-mlx from remote repository"
    echo "   Repository: $REPO_URL"
    echo "   Branch: $BRANCH"

    git clone -b "$BRANCH" "$REPO_URL" "$DEST_DIR"

    echo "✅ Remote vllm-mlx cloned successfully"
fi

echo "📍 vllm-mlx is ready at: $DEST_DIR"
ls -la "$DEST_DIR" | head -5
