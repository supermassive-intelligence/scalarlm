#!/bin/bash
# Comprehensive MLX Performance Testing Suite
# Tests ScalarLM + vllm-mlx performance with statistical rigor

set -e

# Configuration
API_URL="${SCALARLM_API_URL:-http://localhost:8000}"
MODEL="${SCALARLM_MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
OUTPUT_DIR="${OUTPUT_DIR:-./perf-results}"
RUNS_PER_TEST="${RUNS_PER_TEST:-10}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"

# Test parameters
BATCH_SIZES=(1 4 8 16 32)
SEQ_LENGTHS=(128 256 512 1024)
CONCURRENT_LEVELS=(1 2 4 8 16)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  ScalarLM + vllm-mlx Performance Suite  "
echo "=========================================="
echo ""
echo "Configuration:"
echo "  API URL:       $API_URL"
echo "  Model:         $MODEL"
echo "  Output:        $OUTPUT_DIR"
echo "  Runs/test:     $RUNS_PER_TEST"
echo "  Warmup runs:   $WARMUP_RUNS"
echo ""

# Check if server is running
if ! curl -s "$API_URL/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Server not responding at $API_URL${NC}"
    echo "Start the server with: ./run-local-mlx.sh"
    exit 1
fi

echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Helper function: Make a single request and measure performance
make_request() {
    local prompt="$1"
    local max_tokens="$2"
    local stream="${3:-false}"

    local start=$(python3 -c 'import time; print(time.time())')

    local response=$(curl -s "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": $max_tokens,
            \"stream\": $stream
        }")

    local end=$(python3 -c 'import time; print(time.time())')

    # Parse response and calculate metrics
    python3 - <<EOF "$response" "$start" "$end"
import json
import sys

response = sys.argv[1]
start = float(sys.argv[2])
end = float(sys.argv[3])

try:
    data = json.loads(response)

    if 'error' in data:
        print(f"ERROR: {data['error']}")
        sys.exit(1)

    usage = data.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)

    elapsed = end - start
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    total_throughput = total_tokens / elapsed if elapsed > 0 else 0

    # Output metrics as JSON
    metrics = {
        'elapsed': elapsed,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'tokens_per_sec': tokens_per_sec,
        'total_throughput': total_throughput
    }
    print(json.dumps(metrics))

except json.JSONDecodeError:
    print(f"ERROR: Invalid JSON response")
    sys.exit(1)
EOF
}

# Helper function: Run statistical analysis on results
analyze_results() {
    local test_name="$1"
    shift
    local results=("$@")

    python3 - "${results[@]}" <<'EOF'
import sys
import json
import statistics

results = []
for arg in sys.argv[1:]:
    try:
        results.append(json.loads(arg))
    except:
        pass

if not results:
    print("No valid results")
    sys.exit(1)

# Extract metrics
tokens_per_sec = [r['tokens_per_sec'] for r in results]
elapsed = [r['elapsed'] for r in results]
completion_tokens = [r['completion_tokens'] for r in results]

# Calculate statistics
def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = int(k) + 1
    if f == c:
        return sorted_data[f]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1

print("\nStatistical Analysis:")
print(f"  Tokens/second (generation):")
print(f"    Mean:   {statistics.mean(tokens_per_sec):.1f} tok/s")
print(f"    Median: {statistics.median(tokens_per_sec):.1f} tok/s")
print(f"    StdDev: {statistics.stdev(tokens_per_sec):.1f} tok/s")
print(f"    Min:    {min(tokens_per_sec):.1f} tok/s")
print(f"    Max:    {max(tokens_per_sec):.1f} tok/s")
print(f"    p95:    {percentile(tokens_per_sec, 0.95):.1f} tok/s")
print(f"    p99:    {percentile(tokens_per_sec, 0.99):.1f} tok/s")
print(f"  Latency:")
print(f"    Mean:   {statistics.mean(elapsed):.3f}s")
print(f"    Median: {statistics.median(elapsed):.3f}s")
print(f"    p50:    {percentile(elapsed, 0.50):.3f}s")
print(f"    p95:    {percentile(elapsed, 0.95):.3f}s")
print(f"    p99:    {percentile(elapsed, 0.99):.3f}s")
print(f"  Tokens generated:")
print(f"    Mean:   {statistics.mean(completion_tokens):.0f} tokens")

# Save summary to CSV
summary = {
    'mean_tps': statistics.mean(tokens_per_sec),
    'median_tps': statistics.median(tokens_per_sec),
    'p95_tps': percentile(tokens_per_sec, 0.95),
    'p99_tps': percentile(tokens_per_sec, 0.99),
    'mean_latency': statistics.mean(elapsed),
    'p95_latency': percentile(elapsed, 0.95),
    'p99_latency': percentile(elapsed, 0.99),
}
print(f"\nSUMMARY_JSON:{json.dumps(summary)}")
EOF
}

# Test 1: Cold Start Performance
echo "=========================================="
echo "Test 1: Cold Start Performance"
echo "=========================================="
echo "Measuring first request latency..."
echo ""

PROMPT="Explain quantum computing in one sentence."
result=$(make_request "$PROMPT" 50 false)
echo "$result" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Cold start latency: {data['elapsed']:.3f}s\")
print(f\"Tokens/second:      {data['tokens_per_sec']:.1f} tok/s\")
"

echo ""

# Test 2: Warm Performance (Sustained Throughput)
echo "=========================================="
echo "Test 2: Warm Performance (Sustained)"
echo "=========================================="
echo "Running $RUNS_PER_TEST requests after $WARMUP_RUNS warmup runs..."
echo ""

PROMPT="Write a paragraph about machine learning."

# Warmup
echo "Warming up..."
for i in $(seq 1 $WARMUP_RUNS); do
    make_request "$PROMPT" 100 false > /dev/null 2>&1
    echo -n "."
done
echo ""

# Actual test runs
results=()
echo "Running test..."
for i in $(seq 1 $RUNS_PER_TEST); do
    result=$(make_request "$PROMPT" 100 false)
    if [ $? -eq 0 ]; then
        results+=("$result")
        echo -n "."
    fi
done
echo ""

analyze_results "warm_performance" "${results[@]}"

# Test 3: Sequence Length Sweep
echo ""
echo "=========================================="
echo "Test 3: Sequence Length Impact"
echo "=========================================="

seq_results_csv="$OUTPUT_DIR/seq_length_results.csv"
echo "max_tokens,mean_tps,p95_tps,mean_latency,p95_latency" > "$seq_results_csv"

for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo ""
    echo "Testing with max_tokens=$seq_len..."

    PROMPT="Write a detailed article about artificial intelligence."

    # Warmup
    make_request "$PROMPT" "$seq_len" false > /dev/null 2>&1

    # Test runs
    results=()
    for i in $(seq 1 $RUNS_PER_TEST); do
        result=$(make_request "$PROMPT" "$seq_len" false)
        if [ $? -eq 0 ]; then
            results+=("$result")
            echo -n "."
        fi
    done
    echo ""

    summary=$(analyze_results "seq_$seq_len" "${results[@]}" | grep "SUMMARY_JSON:" | cut -d: -f2-)

    # Parse summary and append to CSV
    echo "$summary" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"$seq_len,{data['mean_tps']:.1f},{data['p95_tps']:.1f},{data['mean_latency']:.3f},{data['p95_latency']:.3f}\")
" >> "$seq_results_csv"
done

echo ""
echo -e "${GREEN}✓ Sequence length results saved to: $seq_results_csv${NC}"

# Test 4: Concurrent Request Handling
echo ""
echo "=========================================="
echo "Test 4: Concurrent Request Handling"
echo "=========================================="

concurrent_results_csv="$OUTPUT_DIR/concurrent_results.csv"
echo "concurrency,requests_per_sec,mean_latency,p95_latency,error_rate" > "$concurrent_results_csv"

for concurrency in "${CONCURRENT_LEVELS[@]}"; do
    echo ""
    echo "Testing with concurrency=$concurrency..."

    PROMPT="Summarize the benefits of cloud computing."

    # Run concurrent requests
    temp_results="/tmp/mlx_concurrent_$$"
    rm -rf "$temp_results"
    mkdir -p "$temp_results"

    start_time=$(python3 -c 'import time; print(time.time())')

    for i in $(seq 1 $concurrency); do
        (
            result=$(make_request "$PROMPT" 100 false 2>&1)
            echo "$result" > "$temp_results/$i.json"
        ) &
    done

    # Wait for all background jobs
    wait

    end_time=$(python3 -c 'import time; print(time.time())')

    # Analyze results
    python3 - "$temp_results" "$start_time" "$end_time" "$concurrency" <<'EOF'
import sys
import json
import os
import statistics

results_dir = sys.argv[1]
start_time = float(sys.argv[2])
end_time = float(sys.argv[3])
concurrency = int(sys.argv[4])

results = []
errors = 0

for filename in os.listdir(results_dir):
    filepath = os.path.join(results_dir, filename)
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if 'ERROR' in content:
                errors += 1
            else:
                data = json.loads(content)
                results.append(data)
    except:
        errors += 1

if results:
    latencies = [r['elapsed'] for r in results]

    total_elapsed = end_time - start_time
    requests_per_sec = len(results) / total_elapsed
    error_rate = errors / concurrency

    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        if len(sorted_data) == 1:
            return sorted_data[0]
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = min(int(k) + 1, len(sorted_data) - 1)
        if f == c or c >= len(sorted_data):
            return sorted_data[f]
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return d0 + d1

    print(f"\nConcurrency: {concurrency}")
    print(f"  Requests/sec:   {requests_per_sec:.2f}")
    print(f"  Mean latency:   {statistics.mean(latencies):.3f}s")
    print(f"  p95 latency:    {percentile(latencies, 0.95):.3f}s")
    print(f"  Error rate:     {error_rate*100:.1f}%")

    print(f"CSV_LINE:{concurrency},{requests_per_sec:.2f},{statistics.mean(latencies):.3f},{percentile(latencies, 0.95):.3f},{error_rate:.3f}")
else:
    print(f"\nConcurrency: {concurrency} - ALL REQUESTS FAILED")
    print(f"CSV_LINE:{concurrency},0,0,0,1.0")
EOF

    # Append to CSV
    result=$(python3 - "$temp_results" "$start_time" "$end_time" "$concurrency" <<'EOF'
import sys, json, os, statistics

results_dir = sys.argv[1]
start_time = float(sys.argv[2])
end_time = float(sys.argv[3])
concurrency = int(sys.argv[4])

results = []
errors = 0

for filename in os.listdir(results_dir):
    filepath = os.path.join(results_dir, filename)
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if 'ERROR' in content:
                errors += 1
            else:
                data = json.loads(content)
                results.append(data)
    except:
        errors += 1

if results:
    latencies = [r['elapsed'] for r in results]
    total_elapsed = end_time - start_time
    requests_per_sec = len(results) / total_elapsed
    error_rate = errors / concurrency

    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        if len(sorted_data) == 1:
            return sorted_data[0]
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = min(int(k) + 1, len(sorted_data) - 1)
        if f == c or c >= len(sorted_data):
            return sorted_data[f]
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return d0 + d1

    print(f"{concurrency},{requests_per_sec:.2f},{statistics.mean(latencies):.3f},{percentile(latencies, 0.95):.3f},{error_rate:.3f}")
else:
    print(f"{concurrency},0,0,0,1.0")
EOF
)

    echo "$result" >> "$concurrent_results_csv"

    rm -rf "$temp_results"
done

echo ""
echo -e "${GREEN}✓ Concurrency results saved to: $concurrent_results_csv${NC}"

# Test 5: Memory Usage
echo ""
echo "=========================================="
echo "Test 5: Memory Usage"
echo "=========================================="

if command -v vm_stat &> /dev/null; then
    echo "Measuring memory usage (macOS unified memory)..."

    # Get memory before
    mem_before=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')

    # Run memory-intensive test
    PROMPT="Write a comprehensive guide to deep learning, covering neural networks, training algorithms, optimization techniques, and practical applications in computer vision, natural language processing, and reinforcement learning."

    echo "Running large generation task..."
    make_request "$PROMPT" 2048 false > /dev/null 2>&1

    # Get memory after
    mem_after=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')

    # Calculate difference (pages * 4096 bytes)
    mem_diff=$(( ($mem_before - $mem_after) * 4096 / 1024 / 1024 ))

    echo "  Memory delta: ${mem_diff}MB"
else
    echo "vm_stat not available - skipping memory test"
fi

# Test 6: Time-to-First-Token (Streaming)
echo ""
echo "=========================================="
echo "Test 6: Time-to-First-Token (Streaming)"
echo "=========================================="
echo "Measuring TTFT with streaming enabled..."

PROMPT="Explain machine learning algorithms."

# Note: This is a simplified TTFT test. A full implementation would parse SSE stream
for i in $(seq 1 5); do
    start=$(python3 -c 'import time; print(time.time())')

    # Start streaming request and capture first chunk
    curl -s "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -N \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 100,
            \"stream\": true
        }" | head -n 2 > /dev/null 2>&1

    end=$(python3 -c 'import time; print(time.time())')
    ttft=$(python3 -c "print(f'{$end - $start:.3f}')")
    echo "  Run $i: ${ttft}s"
done

# Final Summary
echo ""
echo "=========================================="
echo "Performance Test Summary"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - $seq_results_csv"
echo "  - $concurrent_results_csv"
echo ""
echo -e "${GREEN}Performance testing complete!${NC}"
echo ""
echo "To visualize results:"
echo "  1. Open CSV files in spreadsheet software"
echo "  2. Plot sequence length vs throughput"
echo "  3. Plot concurrency vs latency"
echo ""
