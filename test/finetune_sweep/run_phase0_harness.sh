#!/bin/bash
# Phase-0 model-load regression harness runner (vLLM-fork upgrade plan, Phase 0).
#
# Runs the fork's meta-device model-load regression + symbol-drift harness
# (vllm/tests/tokenformer/test_model_load_regression.py, test_symbol_drift.py)
# inside the cray image, on CPU, in ~1 minute. Green on the untouched fork; goes
# red the moment a version bump drifts a core symbol or reshapes a model tree the
# adapter layer targets.
#
# Local dev (default): bind-mounts the CURRENT fork python + tests over the built
# image, so no recompile is needed as long as the branch is still ABI-compatible
# with the image's compiled extensions (true while the branch remains v0.19-based;
# once the v0.25 toolchain lands, rebuild the image and drop the bind-mounts).
#
# Usage:
#   test/finetune_sweep/run_phase0_harness.sh            # bind-mount current source
#   IMAGE=scalarlm-cray:latest test/finetune_sweep/run_phase0_harness.sh
#   NO_BIND=1 test/finetune_sweep/run_phase0_harness.sh  # test the image as-built (CI)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-scalarlm-cray:latest}"
cd "$REPO_ROOT"

MOUNTS=()
if [ "${NO_BIND:-0}" != "1" ]; then
  MOUNTS+=(-v "$REPO_ROOT/vllm/vllm:/app/cray/vllm/vllm:ro")
  MOUNTS+=(-v "$REPO_ROOT/vllm/tests/tokenformer:/app/cray/vllm/tests/tokenformer:ro")
fi

exec docker run --rm --entrypoint bash "${MOUNTS[@]}" "$IMAGE" -lc '
  uv pip install -q pytest 2>/dev/null || pip install -q pytest 2>/dev/null
  cd /app/cray/vllm
  python3 -m pytest \
    tests/tokenformer/test_symbol_drift.py \
    tests/tokenformer/test_model_load_regression.py \
    -q --no-header --noconftest -p no:cacheprovider
'
