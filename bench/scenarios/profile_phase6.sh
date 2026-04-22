#!/usr/bin/env bash
# Profile the openai proxy's in-process path (Phase 6) at N=100 with
# distinct prompts. Runs py-spy against the APIServer PID for the
# duration of a 10-iteration sweep and drops a flamegraph SVG +
# speedscope JSON under bench/profiles/<timestamp>/.
#
# Must run INSIDE the pod so py-spy can see the target process.
#
# Usage:
#   bash bench/scenarios/profile_phase6.sh <model> [base-url]

set -euo pipefail

MODEL="${1:-}"
URL="${2:-http://localhost:8000}"
if [[ -z "${MODEL}" ]]; then
  echo "usage: $0 <model> [base-url]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/bench/profiles/${STAMP}"
mkdir -p "${OUT}"
echo "Output: ${OUT}"

# Find the APIServer uvicorn process — the FastAPI app hosting the
# openai proxy. The main.py process spawns a uvicorn worker via
# multiprocessing.spawn; that worker is the actual ASGI server whose
# handlers we want to profile. On this fork it shows up with
# `spawn_main` in argv, and is the parent of VLLM::EngineCore.
PID=$(pgrep -f "VLLM::EngineCore" | head -1)
if [[ -n "${PID}" ]]; then
  # Walk up to the parent spawn_main worker.
  PID=$(ps -o ppid= -p "${PID}" | tr -d ' ')
fi
if [[ -z "${PID}" ]]; then
  PID=$(pgrep -f "spawn_main" | head -1)
fi
if [[ -z "${PID}" ]]; then
  echo "could not find APIServer uvicorn PID" >&2
  ps -ef | grep -E "python|uvicorn" | grep -v grep | head
  exit 3
fi
echo "Profiling PID=${PID}"

# Resolve py-spy: prefer PATH, fall back to the user-site path we install
# into when the image doesn't bake it in.
PYSPY=$(command -v py-spy || true)
if [[ -z "${PYSPY}" && -x "${HOME}/.local/bin/py-spy" ]]; then
  PYSPY="${HOME}/.local/bin/py-spy"
fi
if [[ -z "${PYSPY}" ]]; then
  echo "py-spy not found — install with: python3 -m pip install --break-system-packages py-spy" >&2
  exit 4
fi

# py-spy: Python-only (no --native), rate 50 Hz, ONLY the uvicorn worker
# — no --subprocesses. We tried --subprocesses once and the EngineCore
# child (~200 CUDA threads doing shm_broadcast IPC) drowned out the API
# server's signal. The question this profile needs to answer is "where
# does time go in the proxy's in-process code path?", so sampling just
# the uvicorn worker PID is exactly right.
#
# Speedscope format: opens in https://speedscope.app for humans, and is
# a simple JSON schema we can diff programmatically.
DURATION="${PROFILE_SECONDS:-120}"
# --gil: only sample threads holding the Python GIL. The uvicorn worker
# has ~260 threads because torch / NCCL / CUDA spin up C++ worker pools;
# those threads hold no Python state and sampling them at 50 Hz had py-spy
# falling hundreds of seconds behind (only 48 samples landed in 120 s).
# With --gil py-spy only walks Python threads, which is all we care about
# for "where does the Phase 6 in-process dispatch spend time".
"${PYSPY}" record \
  --pid "${PID}" \
  --output "${OUT}/profile.speedscope.json" \
  --format speedscope \
  --rate 50 \
  --duration "${DURATION}" \
  --gil \
  > "${OUT}/py-spy.log" 2>&1 &
SPY_PID=$!
echo "py-spy running pid=${SPY_PID} duration=${DURATION}s"

# Run the distinct-prompts sweep while py-spy samples.
sleep 2  # let py-spy attach
mkdir -p "${OUT}/runs"
for i in $(seq 1 10); do
  echo ">>> run ${i}/10"
  python3 "${ROOT}/bench/client/pathb_completions_array.py" \
    --url "${URL}" --model "${MODEL}" --prompt-count 100 --max-tokens 16 \
    --distinct-prompts --prompt "bench ${i} $(date +%N)" \
    > "${OUT}/runs/run_$(printf %02d "$i").json"
done

# Let py-spy finish if it hasn't already.
wait "${SPY_PID}" 2>/dev/null || true
echo "profile complete: ${OUT}/flamegraph.svg"
ls -la "${OUT}" | head

python3 - "${OUT}/runs" <<'PY'
import json, os, sys, statistics as s
d = sys.argv[1]
pps = []
for f in sorted(os.listdir(d)):
    if f.startswith("run_"):
        try: pps.append(json.load(open(os.path.join(d, f)))["prompts_per_second"])
        except: pass
if pps:
    print(f"openai N=100 distinct-prompts  mean={s.mean(pps):.2f} stdev={s.stdev(pps) if len(pps)>1 else 0:.2f} range=[{min(pps):.2f}, {max(pps):.2f}]")
PY
