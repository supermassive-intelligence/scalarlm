#!/usr/bin/env bash
# Smoke-test the Phase 3 features against a running ScalarLM instance.
# Expects the server on http://localhost:8000. Safe to re-run; each check
# prints a PASS/FAIL line so grepping FAIL is enough.
set -u

BASE="${1:-http://localhost:8000}"
PASS=0
FAIL=0

check() {
  local name="$1"; shift
  if "$@"; then
    echo "PASS  $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $name"
    FAIL=$((FAIL + 1))
  fi
}

# 1. /v1/health responds and carries X-Request-Id
check_health_with_request_id() {
  local hdrs
  hdrs=$(curl -s -D - "${BASE}/v1/health" -o /dev/null)
  echo "${hdrs}" | head -1 | grep -q " 200"
  local body_ok=$?
  echo "${hdrs}" | tr -d '\r' | grep -qi "^x-request-id: [0-9a-f]\{32\}$"
  local hdr_ok=$?
  [ $body_ok -eq 0 ] && [ $hdr_ok -eq 0 ]
}

# 2. Caller-provided X-Request-Id is honoured
check_request_id_round_trips() {
  local hdrs
  hdrs=$(curl -s -D - -H "X-Request-Id: smoke-phase3" "${BASE}/v1/health" -o /dev/null)
  echo "${hdrs}" | tr -d '\r' | grep -qi "^x-request-id: smoke-phase3$"
}

# 3. /v1/bench/nop is reachable (bench_endpoints_enabled active)
check_bench_nop_available() {
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/v1/bench/nop")
  [ "${code}" = "200" ]
}

# 4. Path A deprecation log fires when /v1/generate is hit
check_path_a_deprecation_logged() {
  # Trigger with a tiny request; worker may or may not drain it, we only
  # care that the log line is emitted.
  curl -s -o /dev/null -X POST "${BASE}/v1/generate" \
    -H "Content-Type: application/json" \
    -H "User-Agent: smoke-phase3/1.0" \
    -d '{"prompts":["hi"],"max_tokens":1}' &
  local pid=$!
  sleep 2
  kill -9 "${pid}" 2>/dev/null || true
  docker logs --since 30s cray-smoke 2>&1 | grep -q '"event": "path_a_deprecation"'
}

# 5. Batch API submit round-trips — submit → immediate status
check_batches_submit_status() {
  local resp batch_id status
  resp=$(curl -s -X POST "${BASE}/v1/batches" \
    -H "Content-Type: application/json" \
    -d '{"input":"{\"custom_id\":\"a\",\"method\":\"POST\",\"url\":\"/v1/chat/completions\",\"body\":{\"model\":\"tiny-random/gemma-4-dense\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}}","endpoint":"/v1/chat/completions"}')
  batch_id=$(echo "${resp}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "")
  [ -n "${batch_id}" ] || return 1
  # Polling status endpoint
  status=$(curl -s "${BASE}/v1/batches/${batch_id}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo "")
  case "${status}" in
    validating|in_progress|completed|failed|cancelled) return 0 ;;
    *) echo "  unexpected status: ${status}"; return 1 ;;
  esac
}

# 6. Batch DELETE transitions to cancelled (or already-terminal)
check_batches_delete() {
  local resp batch_id status
  resp=$(curl -s -X POST "${BASE}/v1/batches" \
    -H "Content-Type: application/json" \
    -d '{"input":"{\"custom_id\":\"b\",\"method\":\"POST\",\"url\":\"/v1/chat/completions\",\"body\":{\"model\":\"tiny-random/gemma-4-dense\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}}","endpoint":"/v1/chat/completions"}')
  batch_id=$(echo "${resp}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "")
  [ -n "${batch_id}" ] || return 1
  status=$(curl -s -X DELETE "${BASE}/v1/batches/${batch_id}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo "")
  case "${status}" in
    cancelled|completed|failed) return 0 ;;
    *) echo "  unexpected status after DELETE: ${status}"; return 1 ;;
  esac
}

check "health_with_request_id"        check_health_with_request_id
check "request_id_round_trips"        check_request_id_round_trips
check "bench_nop_available"           check_bench_nop_available
check "path_a_deprecation_logged"     check_path_a_deprecation_logged
check "batches_submit_status"         check_batches_submit_status
check "batches_delete_round_trip"     check_batches_delete

echo
echo "Passed: ${PASS}  Failed: ${FAIL}"
[ ${FAIL} -eq 0 ]
