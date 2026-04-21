# ScalarLM OpenAI-API benchmarks

Harness for the benchmark plan in [`enhance-openai-api.md`](../enhance-openai-api.md). The goal isn't a single number — it's a grid across **three platforms** (MacBook Pro M5, NVIDIA DGX Spark, 4× Blackwell) and **six workloads** (nop, per-request chat, array-prompt completions, Batch API, Path A generate bulk, Path A upload/download) at concurrency from 10 to 1 M.

This README covers the scripts that exist today. Scripts for the inference-heavy scenarios will land alongside Level 3b.

## Directory layout

```
bench/
  client/
    nop.py                     # hammers /v1/bench/nop
    pathb_chat_single.py       # (TODO) per-request chat completions
    pathb_completions_array.py # (TODO) array-prompt /v1/completions
    pathb_batches.py           # (TODO) submit/poll /v1/batches
    patha_generate_bulk.py     # (TODO) /v1/generate with prompts=[...]
    patha_upload_download.py   # (TODO) /v1/generate/upload + /download
  scenarios/
    nop_sweep.sh               # runs nop across platforms.yaml caps
  platforms.yaml               # per-platform model + concurrency caps
  results/
    <platform>/<timestamp>/    # one tree per run
```

## Requirements

A Python 3.10+ venv with `httpx[http2]` installed:

```bash
python3 -m venv /tmp/bench-venv
/tmp/bench-venv/bin/pip install 'httpx[http2]'
```

The benchmark scripts shell out to the venv's python via the current shell's PATH — activate the venv or point `python3` at it before running.

## Enabling `/v1/bench/nop`

The nop endpoint is **off by default** so production builds don't accidentally expose a free no-auth request sink. Turn it on per-deploy:

```bash
# As env for scalarlm up:
SCALARLM_BENCH_ENDPOINTS_ENABLED=true scalarlm up ...
```

Or set `bench_endpoints_enabled: true` in the deployed `cray-config.yaml`.

## Running the nop sweep

```bash
bench/scenarios/nop_sweep.sh mac-m5
bench/scenarios/nop_sweep.sh dgx-spark
bench/scenarios/nop_sweep.sh blackwell-4gpu  http://scalarlm.blackwell:8000
```

The script reads `nop_concurrency` for the requested platform from `platforms.yaml`, runs `bench/client/nop.py` at each, and writes one JSON file per concurrency plus a rolled-up `summary.json` under `bench/results/<platform>/<timestamp>/`.

Each run's JSON shape:

```json
{
  "concurrency": 1000,
  "duration_seconds": 10.04,
  "requests_total": 14387,
  "requests_per_second": 1433.01,
  "p50_ms": 3.2,
  "p95_ms": 11.7,
  "p99_ms": 28.9,
  "max_ms": 112.1,
  "status_counts": {"200": 14387},
  "errors": 0
}
```

## Interpreting results

- `requests_per_second` under `nop` is the FastAPI / uvicorn / middleware ceiling on this platform. It gives the denominator for proxy-overhead claims in the plan.
- Non-200 entries in `status_counts` or non-zero `errors` at low concurrency are a bug — middleware should survive every request.
- Compare `p99_ms` across concurrency levels: flat → plenty of headroom; curving up → approaching the uvicorn ceiling; cliffs → broke.
- Under HTTP/2 the same concurrency should work with a small TCP pool. Pass `--max-connections 32` to verify — if throughput collapses, HTTP/2 multiplexing isn't active (check the server/TLS path).

## What this doesn't measure

- Inference throughput (that's the non-nop scenarios, coming in Level 3b).
- Streaming semantics — the nop endpoint returns JSON, not SSE.
- Path A's queue throughput — separate script under `patha_*.py` once the harness is completed.
