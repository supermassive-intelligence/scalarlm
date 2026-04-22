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
# openai proxy. It's the parent python that imports cray_infra.one_server
# and whose children include the vLLM EngineCore workers.
PID=$(pgrep -f "cray_infra.one_server.main" | head -1)
if [[ -z "${PID}" ]]; then
  echo "could not find cray_infra.one_server.main PID" >&2
  ps -ef | grep -E "python|uvicorn" | grep -v grep | head
  exit 3
fi
echo "Profiling PID=${PID}"

# py-spy: sample every 10 ms (rate=100), native frames on, duration
# slightly longer than the sweep wall-clock so we catch the full run.
DURATION="${PROFILE_SECONDS:-90}"
py-spy record \
  --pid "${PID}" \
  --output "${OUT}/flamegraph.svg" \
  --rate 100 \
  --duration "${DURATION}" \
  --native \
  --subprocesses \
  > "${OUT}/py-spy.log" 2>&1 &
SPY_PID=$!
echo "py-spy running pid=${SPY_PID} duration=${DURATION}s"

# Run the distinct-prompts sweep while py-spy samples.
sleep 2  # let py-spy attach
mkdir -p "${OUT}/runs"
for i in $(seq 1 10); do
  echo ">>> run ${i}/10"
  /app/.venv/bin/python "${ROOT}/bench/client/pathb_completions_array.py" \
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
