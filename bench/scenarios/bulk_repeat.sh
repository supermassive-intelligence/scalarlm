#!/usr/bin/env bash
# Repeated runs at a specific N to get statistics for scalarlm /v1/generate
# vs openai /v1/completions array. First iteration of each scenario is a
# warmup and excluded; next ITER runs are measurement.
#
# Usage:
#   bench/scenarios/bulk_repeat.sh <model> <platform-label> <N> [base-url]

set -euo pipefail

MODEL="${1:-}"
PLATFORM="${2:-}"
N="${3:-}"
URL="${4:-http://localhost:8000}"
ITER="${BULK_ITERATIONS:-10}"
WARMUP="${BULK_WARMUP:-1}"

if [[ -z "${MODEL}" || -z "${PLATFORM}" || -z "${N}" ]]; then
  echo "usage: $0 <model> <platform-label> <N> [base-url]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/bench/results/${PLATFORM}-n${N}-repeat/${STAMP}"
mkdir -p "${OUT}/scalarlm" "${OUT}/openai"

echo "Model:    ${MODEL}"
echo "N:        ${N}"
echo "Out:      ${OUT}"
echo "Warmup:   ${WARMUP} iteration(s) per scenario (discarded)"
echo "Measure:  ${ITER} iteration(s) per scenario"

run_many() {
  local script="$1" subdir="$2"
  echo ">>> ${subdir} warmup"
  for w in $(seq 1 "${WARMUP}"); do
    # Warmup uses a distinct tag too so scalarlm's request-id dedup
    # (SHA-256 of the request list) doesn't feed cached results into
    # the measurement runs.
    python3 "${ROOT}/bench/client/${script}" \
      --url "${URL}" --model "${MODEL}" --prompt-count "${N}" --max-tokens 16 \
      --prompt "w${subdir}${w}n${N} Say hello in five words." \
      > /dev/null
  done
  echo ">>> ${subdir} measurement"
  for i in $(seq 1 "${ITER}"); do
    python3 "${ROOT}/bench/client/${script}" \
      --url "${URL}" --model "${MODEL}" --prompt-count "${N}" --max-tokens 16 \
      --prompt "${subdir} run ${i} n${N} $(date +%N) Say hello in five words." \
      > "${OUT}/${subdir}/run_$(printf %02d "$i").json"
  done
}

run_many pathb_completions_array.py openai
run_many patha_generate_bulk.py     scalarlm

python3 - "${OUT}" <<'PY'
import json, os, statistics as s, sys
root = sys.argv[1]
report = {}
for scenario in ("openai", "scalarlm"):
    pps, tps, dur = [], [], []
    for name in sorted(os.listdir(os.path.join(root, scenario))):
        if not name.startswith("run_"): continue
        with open(os.path.join(root, scenario, name)) as fh:
            r = json.load(fh)
        pps.append(r.get("prompts_per_second") or 0.0)
        dur.append(r.get("duration_seconds") or 0.0)
        if r.get("tokens_per_second"):
            tps.append(r["tokens_per_second"])
    report[scenario] = {
        "runs": len(pps),
        "prompts_per_second": {
            "mean": s.mean(pps), "stdev": s.stdev(pps) if len(pps) > 1 else 0.0,
            "min": min(pps), "max": max(pps),
        },
        "duration_seconds": {
            "mean": s.mean(dur), "stdev": s.stdev(dur) if len(dur) > 1 else 0.0,
            "min": min(dur), "max": max(dur),
        },
    }
    if tps:
        report[scenario]["tokens_per_second"] = {
            "mean": s.mean(tps), "stdev": s.stdev(tps) if len(tps) > 1 else 0.0,
            "min": min(tps), "max": max(tps),
        }
with open(os.path.join(root, "summary.json"), "w") as fh:
    json.dump(report, fh, indent=2)
for k, v in report.items():
    pps = v["prompts_per_second"]
    line = f"{k:10s} n={v['runs']:2d}  mean={pps['mean']:.2f} p/s  stdev={pps['stdev']:.3f}  range=[{pps['min']:.2f}, {pps['max']:.2f}]"
    print(line)
PY
