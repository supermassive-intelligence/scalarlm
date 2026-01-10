#!/bin/bash
# Script used by Docker build process to handle Chat UI source:
#   - either copy local or clone remote
set -e

UI_SOURCE=$1
DEST_DIR=$2
LOCAL_PATH=$3
REPO_URL=$4
BRANCH=$5

echo "🔧 Setting up Chat UI source..."
echo "   Source type: $UI_SOURCE"

if [ "$UI_SOURCE" = "local" ]; then
    echo "📁 Using local Chat UI from: $LOCAL_PATH"

    if [ ! -d "$LOCAL_PATH" ]; then
        echo "❌ Error: Local Chat UI directory not found at $LOCAL_PATH"
        echo ""
        echo "   For local development, Chat UI must be cloned into the ScalarLM directory:"
        echo "   cd /path/to/scalarlm"
        echo "   git clone -b main https://github.com/supermassive-intelligence/chat-ui-fork.git vllm"
        echo ""
        echo "   This will create: scalarlm/chat-ui/"
        exit 1
    fi

    echo "📋 Copying local Chat UI to $DEST_DIR..."
    cp -r "$LOCAL_PATH" "$DEST_DIR"

    # Keep .git directory for setuptools-scm version detection
    # setuptools-scm needs git metadata to determine version
    echo "📌 Keeping git metadata for version detection"

    echo "✅ Local Chat UI copied successfully"

else
    echo "🌐 Cloning Chat UI from remote repository"
    echo "   Repository: $REPO_URL"
    echo "   Branch: $BRANCH"

    git clone -b "$BRANCH" "$REPO_URL" "$DEST_DIR"

    echo "✅ Remote Chat UI cloned successfully"
fi

echo "📍 Chat UI is ready at: $DEST_DIR"
ls -la "$DEST_DIR" | head -5
