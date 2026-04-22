#!/usr/bin/env bash
# Scoped first-pass inference sweep for Blackwell. Runs pathb_chat_single,
# pathb_completions_array, pathb_batches, patha_generate_bulk,
# patha_upload_download at bounded concurrency — enough to see the queue
# kick in and the parity curve between Path A bulk and Path B array, but
# not the 1 M levels that the full plan calls for. Bump caps once the
# baseline run looks healthy.
#
# Usage:
#   bench/scenarios/blackwell_first_pass.sh <model> [base-url]

set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "usage: $0 <model> [base-url]" >&2
  exit 2
fi
URL_BASE="${2:-http://localhost:8000}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PLATFORM="blackwell-4gpu-first-pass"
OUT_DIR="${ROOT}/bench/results/${PLATFORM}/${STAMP}"
mkdir -p "${OUT_DIR}"

CHAT_CONC="1 10 50 200"
BULK_COUNTS="1 10 100 1000"

echo "Model:   ${MODEL}"
echo "URL:     ${URL_BASE}"
echo "Out:     ${OUT_DIR}"

for c in ${CHAT_CONC}; do
  echo ">>> pathb_chat_single c=${c}"
  python3 "${ROOT}/bench/client/pathb_chat_single.py" \
    --url "${URL_BASE}" --model "${MODEL}" \
    --concurrency "${c}" --duration 20 --max-tokens 16 \
    > "${OUT_DIR}/pathb_chat_single_c${c}.json" || true
done

run_bulk() {
  local script="$1" label="$2"
  for n in ${BULK_COUNTS}; do
    echo ">>> ${label} n=${n}"
    python3 "${ROOT}/bench/client/${script}" \
      --url "${URL_BASE}" --model "${MODEL}" --prompt-count "${n}" --max-tokens 16 \
      > "${OUT_DIR}/${label}_n${n}.json" || true
  done
}

run_bulk pathb_completions_array.py pathb_completions_array
run_bulk pathb_batches.py            pathb_batches
run_bulk patha_generate_bulk.py      patha_generate_bulk
run_bulk patha_upload_download.py    patha_upload_download

python3 - "${OUT_DIR}" <<'PY'
import json, os, re, sys
out_dir = sys.argv[1]
groups = {}
for name in sorted(os.listdir(out_dir)):
    m = re.match(r"^(?P<scenario>[a-z_]+)_(c|n)\d+\.json$", name)
    if not m:
        continue
    groups.setdefault(m.group("scenario"), []).append(name)
summary = {}
for scenario, files in groups.items():
    runs = []
    for name in files:
        with open(os.path.join(out_dir, name)) as fh:
            runs.append(json.load(fh))
    summary[scenario] = runs
with open(os.path.join(out_dir, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"wrote {os.path.join(out_dir, 'summary.json')} with {len(summary)} scenarios")
PY
