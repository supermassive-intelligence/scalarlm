#!/usr/bin/env bash
# Run the nop scenario across a concurrency sweep and write one summary
# JSON per concurrency level under bench/results/<platform>/<timestamp>/.
#
# Usage: bench/scenarios/nop_sweep.sh <platform-label> [base-url]
#
# Platform labels: mac-m5, dgx-spark, blackwell-4gpu.
# Caps per platform live in bench/platforms.yaml; this script reads the
# 'nop_concurrency' list for the requested platform.

set -euo pipefail

PLATFORM="${1:-}"
if [[ -z "${PLATFORM}" ]]; then
  echo "usage: $0 <platform-label> [base-url]" >&2
  exit 2
fi
URL_BASE="${2:-http://localhost:8000}"
DURATION="${BENCH_DURATION_SECONDS:-10}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ROOT}/bench/results/${PLATFORM}/${STAMP}"
mkdir -p "${OUT_DIR}"

# Extract the concurrency list for this platform from platforms.yaml.
# Minimal yaml reader: we only need one list of ints under a known key,
# so awk is less dependency-heavy than pulling in PyYAML for the shell.
concurrencies=$(awk -v plat="${PLATFORM}" '
  /^[^ ]/                         { in_platform = ($1 == plat":") }
  in_platform && /nop_concurrency:/ { in_list = 1; next }
  in_list && /^[^ ]|^  [a-z_]+:/  { in_list = 0 }
  in_list && /^    - /            { gsub("- ", ""); print $1 }
' "${ROOT}/bench/platforms.yaml")

if [[ -z "${concurrencies}" ]]; then
  echo "no nop_concurrency entry for platform '${PLATFORM}' in platforms.yaml" >&2
  exit 3
fi

echo "Sweeping nop on ${PLATFORM}: ${concurrencies}" | tr '\n' ' '
echo

for c in ${concurrencies}; do
  echo ">>> concurrency=${c}"
  python3 "${ROOT}/bench/client/nop.py" \
    --url "${URL_BASE}/v1/bench/nop" \
    --concurrency "${c}" \
    --duration "${DURATION}" \
    > "${OUT_DIR}/nop_c${c}.json"
  echo "    -> ${OUT_DIR}/nop_c${c}.json"
done

# Roll up the per-concurrency files into one summary.json for the run.
python3 - "${OUT_DIR}" <<'PY'
import json, os, sys
out_dir = sys.argv[1]
runs = []
for name in sorted(os.listdir(out_dir)):
    if not name.startswith("nop_c") or not name.endswith(".json"):
        continue
    with open(os.path.join(out_dir, name)) as fh:
        runs.append(json.load(fh))
with open(os.path.join(out_dir, "summary.json"), "w") as fh:
    json.dump({"scenario": "nop", "runs": runs}, fh, indent=2)
print(f"wrote {os.path.join(out_dir, 'summary.json')} ({len(runs)} runs)")
PY
