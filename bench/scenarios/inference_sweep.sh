#!/usr/bin/env bash
# Run the inference scenarios (pathb_chat_single, pathb_completions_array,
# pathb_batches, patha_generate_bulk, patha_upload_download) across the
# concurrency / prompt-count caps listed in platforms.yaml for the
# requested platform.
#
# Usage:
#   bench/scenarios/inference_sweep.sh <platform-label> [base-url]
#
# Writes bench/results/<platform>/<timestamp>/{scenario}_n{N}.json plus a
# rolled-up summary.json per scenario.

set -euo pipefail

PLATFORM="${1:-}"
if [[ -z "${PLATFORM}" ]]; then
  echo "usage: $0 <platform-label> [base-url]" >&2
  exit 2
fi
URL_BASE="${2:-http://localhost:8000}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ROOT}/bench/results/${PLATFORM}/${STAMP}"
mkdir -p "${OUT_DIR}"

read_list() {
  # $1 platform label, $2 key name
  awk -v plat="$1" -v key="$2" '
    /^[^ ]/                 { in_platform = ($1 == plat":") }
    in_platform && $0 ~ ("^  "key":") { in_list = 1; next }
    in_list && /^[^ ]|^  [a-z_]+:/ { in_list = 0 }
    in_list && /^    - /    { gsub("- ", ""); print $1 }
  ' "${ROOT}/bench/platforms.yaml"
}

read_scalar() {
  # $1 platform label, $2 key name
  awk -v plat="$1" -v key="$2" '
    /^[^ ]/                 { in_platform = ($1 == plat":") }
    in_platform && $0 ~ ("^  "key":") { sub(".*: *", ""); print; exit }
  ' "${ROOT}/bench/platforms.yaml"
}

MODEL="$(read_scalar "${PLATFORM}" "model")"
CHAT_CONC=$(read_list "${PLATFORM}" "pathb_chat_single_concurrency")
BULK_COUNTS=$(read_list "${PLATFORM}" "bulk_prompt_count")

if [[ -z "${MODEL}" ]]; then
  echo "no model entry for platform '${PLATFORM}' in platforms.yaml" >&2
  exit 3
fi

echo "Platform: ${PLATFORM}"
echo "Model:    ${MODEL}"
echo "URL:      ${URL_BASE}"
echo "Out:      ${OUT_DIR}"
echo

run_chat_single() {
  for c in ${CHAT_CONC}; do
    echo ">>> pathb_chat_single c=${c}"
    python3 "${ROOT}/bench/client/pathb_chat_single.py" \
      --url "${URL_BASE}" --model "${MODEL}" \
      --concurrency "${c}" --duration "${BENCH_DURATION_SECONDS:-30}" \
      > "${OUT_DIR}/pathb_chat_single_c${c}.json"
  done
}

run_bulk() {
  local script="$1" label="$2"
  for n in ${BULK_COUNTS}; do
    echo ">>> ${label} n=${n}"
    python3 "${ROOT}/bench/client/${script}" \
      --url "${URL_BASE}" --model "${MODEL}" --prompt-count "${n}" \
      > "${OUT_DIR}/${label}_n${n}.json"
  done
}

run_chat_single
run_bulk pathb_completions_array.py pathb_completions_array
run_bulk pathb_batches.py            pathb_batches
run_bulk patha_generate_bulk.py      patha_generate_bulk
run_bulk patha_upload_download.py    patha_upload_download

# Roll up every *_c*.json or *_n*.json under OUT_DIR into per-scenario summaries.
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
