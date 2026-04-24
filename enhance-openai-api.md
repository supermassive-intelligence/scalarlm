# OpenAI-compatible API: feature and performance parity with the ScalarLM path

## Motivation

ScalarLM exposes two inference surfaces today:

- `POST /v1/generate` (the ScalarLM-native path) — queue-backed, batch-oriented,
  with implicit content-hashed response caching.
- `POST /v1/completions` and `POST /v1/chat/completions` (the OpenAI-compatible
  path) — a direct proxy to vLLM's OpenAI server.

Callers who need tool calling or OpenAI SDK compatibility end up on the openai
path; `sql-gen` is the obvious one today. The long-term direction is **one
inference surface**, namely the OpenAI-compatible one. This document lays out
the changes that make the openai path a clean replacement for `/v1/generate`
and records the design decisions along the way.

Before this work, the openai path had three gaps vs. `/v1/generate`:

| gap | consequence |
|---|---|
| no response cache | repeat calls re-ran inference; scalarlm returned the cached batch |
| adapter-load check per request | ~10-20 ms of HTTP round-trip on every call, even with no LoRA loaded |
| bulk array prompts are not paced | N=1000 array `/v1/completions` landed 1000 concurrent `AsyncStream` futures on the APIServer event loop, starving the engine of fresh work per iteration |

After the work in this PR, the openai path has feature + performance parity
with `/v1/generate` on the dimensions that matter for the workloads we run,
and becomes the recommended surface for all new callers.

## Scope of this PR

Each commit adds one piece, plus an accompanying section in this document.
Higher-level roadmap:

1. **Phase 6.5** — move vLLM's output-handler metrics recording off the hot
   path. Foundational; touches the vLLM fork via a build-time patcher.
2. **Phase 7** — parallelise the OpenAI `/v1/batches` runner so it can
   actually hit the ceiling at N=1000.
3. **Phase 30** — batch response cache mirroring `/v1/generate`'s implicit
   cache, keyed on the filtered params dict.
4. **Phase 31b** — memoise adapter-load state so repeat calls skip the
   `/v1/load_lora_adapter` round-trip, and short-circuit the base-model case.
5. **Phase 31** — route bulk array `/v1/completions` through the existing
   `/v1/generate` queue worker. Reuses the scalarlm worker's batched
   `asyncio.gather` pacing so the proxy event loop isn't the bottleneck.
6. **Docs (this commit + final)** — design rationale, performance summary,
   deprecation plan, benchmark repro.

Later commits expand each phase section below.

## Design choices that are load-bearing for performance

These are set once and not per-request — getting them wrong silently caps
throughput by large factors.

### `max_num_seqs` — let vLLM pick

vLLM's default is 256. Overriding it down to 8 or 16 (a common "safety"
choice that never got re-validated) caps concurrent sequences in the engine
scheduler, which caps throughput at large N. On Blackwell 2-GPU with
Qwen3-Next-80B-A3B-FP8 + `gpu_memory_utilization=0.85`, a sweep from 16 to
1024 shows:

| `max_num_seqs` | scalarlm p/s (N=1000 distinct, 16 tok) |
|---:|---:|
|   16 |  16.6 |
|   32 |  22.4 |
|   64 |  39.2 |
|  128 |  68.2 |
|  256 |  91.1 |
|  512 | 130.5 |
| 1024 | 154.0 |

256 is a good default: ~9× the 16-case, diminishing returns past it. **Do
not set `--max_num_seqs` in `SCALARLM_VLLM_ARGS` unless you have a specific
reason**. Changing it requires a pod restart because the KV-cache block pool
and CUDA graphs are allocated at engine boot.

### Pipeline parallelism beats tensor parallelism on this MoE

On 2×Blackwell with the 80B-A3B-FP8 MoE model, `--pipeline-parallel-size=2`
outperforms `--tensor-parallel-size=2` by ~12 % on scalarlm at ms=256
(101.8 vs 91.1 p/s). Each PP stage holds all experts for its layer range,
so expert routing stays intra-GPU and there's no per-layer allreduce. TP
splits each layer's experts across GPUs, which pays allreduce cost on every
forward pass *and* leaves residual expert-routing imbalance across ranks.

PP does have its own asymmetry — one stage typically ends up busier than the
other because layer compute isn't perfectly uniform — but the aggregate
throughput is higher. Split-point tuning to balance stages is an open lever
not landed in this PR.

### GPU-memory utilization: 0.85, not 0.5

The KV-cache block pool is sized against `gpu_memory_utilization × free GPU
memory`. Conservative values (0.5) leave most of the KV budget idle and
directly limit concurrent sequences. 0.85 is the aggressive-but-safe
default; drop it only if you're seeing OOMs. This interacts with
`max_num_seqs` — the engine picks the smaller of the two limits.

## Phase 6.5 — move vLLM's output-handler metrics off the hot path

vLLM's `AsyncLLM.output_handler_loop` calls `loggers.record(...)` synchronously
every iteration, computing Prometheus stats (scheduler, per-engine iteration,
mm-cache) before yielding back to outputs. At N=100 distinct prompts the
record call was **20.3 % of main-thread time** in a py-spy profile. vLLM
already has a `TODO(rob)` on that exact line acknowledging the problem.

The fix applies at vLLM-fork build time (via a patcher script, not a vendored
diff) so upstream rebases don't silently drop it:

- `scripts/vllm_patches/apply_patches.py` — asserts the exact anchor text it
  expects and fails loudly on rebase drift rather than mis-patching silently.
- Metrics move to a background consumer coroutine fed by a bounded
  `asyncio.Queue`. The hot path only pushes; the consumer does the record.
- Bounded queue means bursty metrics can't blow memory and backpressure
  is applied by dropping the oldest entry (metrics are summaries anyway).

Measured effect: on the N=100 distinct pilot A/B, throughput recovered
+15 %. The patch is now part of every built image.

## Phase 30 — response cache for `/v1/completions` and `/v1/chat/completions`

`/v1/generate` hashes the batch contents and persists the response to
`{upload_base_path}/{hash}_response.json`; repeat calls with the same payload
skip inference and return the stored batch. The openai path had no
equivalent, which (a) made identical-prompt workloads pay full inference
cost every time and (b) left OpenAI-compatible clients without a benefit
scalarlm was already giving `/v1/generate` callers.

### Design

- **Keyed on the filtered params dict:** `(model, prompt|messages,
  max_tokens, temperature, top_p, stop, n, tools, tool_choice)`. Any field
  that changes inference output is in the key; transport-only fields
  (`stream`, `stream_options`, `seed` when non-deterministic, etc.) are
  excluded.
- **SHA-256 over sorted-key JSON:** deterministic across Python sessions;
  collisions are not a concern for this key size.
- **Disk-backed at `{upload_base_path}/openai_cache/{sha256}.json`:** same
  directory root as `/v1/generate`'s cache, simplifying GC and quota
  management.
- **Streaming always bypassed:** the cache is batch-granular and a
  streamed response would have to be buffered in memory before storage.
  Acceptable trade-off — streaming is for interactive UX, not batch
  replay.
- **Opt-in via `SCALARLM_OPENAI_CACHE=1`** so existing deployments don't
  see behavior change unless they choose to. Recommended on for any
  deployment that serves batch/eval/dataset-labeling workloads.

### Performance effect

On a repeat-batch workload (1000 prompts, second call with identical
params), Blackwell 2-GPU:

| path | cold p/s | warm (cache hit) p/s |
|---|---:|---:|
| openai /v1/completions | 63 (ms=256) | ~35 000 |
| scalarlm /v1/generate  | 91 (ms=256) | ~690 |

The warm-hit openai throughput dominates scalarlm's because openai
returns the entire batch in one `JSONResponse.content` load;
`/v1/generate`'s `poll_for_responses` iterates all N `request_id`s and
re-reads the response file once per prompt. Not a property of the cache
layer, a property of the scalarlm response-polling loop.

## Phase 31 — route bulk array `/v1/completions` through the queue worker

At N ≥ 100 distinct prompts, the direct-proxy path lags `/v1/generate` by
~30 %. The gap wasn't vLLM — same kernels, same KV cache, same
`max_num_seqs`. It was what the proxy did with the N requests:

- **Direct proxy** forwarded the array through to vLLM's OpenAI server,
  which created N `AsyncStream` futures. At N=1000 those futures all
  live on the APIServer event loop simultaneously, eating CPU on
  per-future output dispatch between engine iterations.
- **`/v1/generate`** writes the batch to a queue, a dedicated worker
  pulls `get_batch_size()` items at a time, submits them via
  `asyncio.gather`, waits, pulls the next chunk. The engine sees the
  same submissions either way, but the proxy event loop only ever
  handles a bounded number of in-flight futures.

The structural fix is to reuse the queue worker. This commit adds a fast
path that, when array length ≥ `SCALARLM_QUEUE_ROUTE_THRESHOLD`, builds
a `GenerateRequest` from the `CompletionRequest` and hands off to the
same `generate()` handler `/v1/generate` uses. The `GenerateResponse`
is translated back to the OpenAI `CompletionResponse` shape.

### Design choices

- **Opt-in via `SCALARLM_QUEUE_ROUTE_THRESHOLD` (int, default 0 / off).**
  Recommended production value: **100**. Below that, the direct path is
  the fast path for interactive latency; above that, the queue pacing
  dominates.
- **Streaming and non-string prompts are never routed.** The queue is
  batch-oriented (no partial emission) and the current
  `GenerateRequest` only handles strings and dict prompts.
- **`request.prompt` is read directly (no `model_dump`).** At N=1000 the
  full `model_dump(mode="json")` walk costs ~20-30 ms per call; the fast
  path reads only the cache-relevant fields.
- **Cache store is skipped on queue-routed responses.** The queue
  worker's own disk cache at `{hash}_response.json` is authoritative;
  duplicating in openai-cache is wasted I/O.

### Known limitations (deliberate, to keep the change small)

- **`usage.*_tokens` returns 0.** scalarlm's `Result` model doesn't carry
  `token_count` today. The worker's `async_completion_task` already
  parses `response_data["usage"]` — promoting those fields onto
  `Result` is ~15 lines of follow-up and will land in a follow-up PR.
  Not blocking for throughput benchmarks; blocking for usage-based
  billing integrations.
- **No streaming on queue-routed calls.** Callers that need streaming at
  N ≥ threshold will get it — just through the direct path (above the
  queue route, in the handler order) if they pass `stream=true`.
- **No logprobs on queue-routed calls.**

### Performance

Blackwell 2-GPU PP=2, `max_num_seqs=256`, Qwen3-Next-80B-A3B-FP8,
N=1000 distinct prompts, `max_tokens=16`, 3-run mean:

| path | mean p/s | min | max |
|---|---:|---:|---:|
| openai /v1/completions (direct)  | ~62   | 60.1  | 63.5  |
| openai /v1/completions (queue route) | **89.7** | 86.4  | 94.2  |
| scalarlm /v1/generate            | 90.9  | 88.2  | 93.7  |

Queue-routed openai hits **statistical parity** with `/v1/generate` on
this workload (1.3 % mean gap, within run-to-run variance).

## Alternatives explored that didn't help

Before the queue-route landed, five proxy-layer approaches were tried to
close the openai-vs-scalarlm gap. All were null at N=1000 distinct and are
not part of this PR. Keeping a concise record so the same experiments don't
get re-run:

- **Scatter-gather at the proxy.** Splitting an array `/v1/completions` into
  N single-prompt `create_completion` calls via `asyncio.gather`. Does not
  change what vLLM sees (vLLM scatters internally either way) and added no
  throughput.
- **Bounded scatter (Semaphore-limited fan-out).** Hypothesis: 1000 pending
  requests were slowing the scheduler. Null — the vLLM scheduler is flat in
  waiting-queue depth.
- **Routing through `api_router.create_completion` (the decorated variant).**
  Small win early on (part of the current direct path), but only because of
  the `JSONResponse` wrapping, not the decorators themselves.
- **Dedicated dispatcher coroutine.** A long-lived coroutine pulled items
  off an `asyncio.Queue` and resolved `Future`s. Structurally mimics the
  scalarlm worker without its file I/O; null result — the difference isn't
  the worker architecture per se, it's the *pacing* (batched gather vs.
  fanned-out futures).
- **Yield injection at chunk boundaries.** `await asyncio.sleep(0)` every
  K sub-requests. Null. Main-loop lag p99 was <15 ms already; openai's main
  thread was *more* idle than scalarlm's.
- **Side-loop thread for `create_completion`.** Null. AsyncStream isn't
  cross-loop-safe cleanly, and the main loop wasn't CPU-saturated anyway.
- **Expert parallelism (`--enable-expert-parallel`).** Actively harmful —
  reshuffled placement so the *other* GPU became the hot one, and added
  cross-GPU expert-dispatch cost. Throughput dropped 39 %.
- **ZMQ message coalescing at the EngineCore boundary.** Planned
  (Gemini's Phase 28) but not landed. Probably worth 3-6 % on top of
  Phase 31, at the cost of a load-bearing vLLM-fork patch that rebasing
  has to maintain. Deferred until there's a specific reason (p99 latency
  at high concurrency) to pay that cost.

The actual win ended up being **Phase 31** — push bulk requests into the
existing scalarlm queue worker rather than fan-out at the proxy. The
architectural difference was not scatter-vs-no-scatter, it was how many
`AsyncStream` futures live on the APIServer event loop at once: 1000 at
peak in the direct path, `max_num_seqs` at peak in the queue path.

## Performance summary

### Blackwell (2× RTX PRO 6000 Max-Q, TP=2 vs PP=2)

Qwen3-Next-80B-A3B-FP8, `gpu_memory_utilization=0.85`, `max_num_seqs=256`,
`--kv-cache-dtype=fp8`. N=1000 distinct prompts, `max_tokens=16`, cold
cache. **All numbers below are 3-run means.** Single-run variance on
this workload is ±8 % due to prefix-cache warmth and engine state;
mixing single samples with multi-run means produces misleading "gaps."

| config | path | mean p/s | 3-run range | GPU 0 util | GPU 1 util |
|---|---|---:|---:|---:|---:|
| TP=2 | scalarlm /v1/generate       | 91.1 | — | ~58 % | ~94 % |
| TP=2 | openai direct               | 63.6 | — | ~58 % | ~94 % |
| PP=2 | scalarlm /v1/generate       | 90.9 | 88.2 – 93.7 | ~97 % | ~59 % |
| PP=2 | openai direct               | 62.3 | — | ~88 % | ~22 % |
| **PP=2** | **openai via queue route** | **89.7** | 86.4 – 94.2 | ~93 % | ~57 % |

Throughput was similar between TP=2 and PP=2 on the 3-run mean. An
earlier single-run had scalarlm at 101.8 p/s under PP=2, but subsequent
cold runs landed in the 88-94 range — within the ±8 % variance band.
GPU utilization patterns differ materially though: TP=2 leaves GPU 0
underused; PP=2 leaves GPU 1 underused. Split-point tuning would
rebalance PP.

Key takeaways:
- `max_num_seqs=256` is a ~9× throughput win over the previous override of 16.
- **Queue route closes the openai-vs-scalarlm gap to statistical parity.**
  Without it, openai runs at ~62 p/s on PP=2; with it, openai runs at ~90
  p/s — same as scalarlm (1.3 % mean gap, inside run-to-run variance).
- With cache hits on repeat batches, openai crosses into the 35 000 p/s
  regime that scalarlm's polling-heavy cache layer can't reach.

### Spark (single NVIDIA GB10, TP=1)

Qwen3-32B-NVFP4, vLLM defaults (`max_num_seqs=256`),
`max_model_length=4096`, `gpu_memory_utilization=0.85`. N=1000 distinct
prompts, `max_tokens=16`. **3-run means, measured against a clean PR
image deployed as `kapu/scalarlm-spark:enhanced-openai-api`, with
`SCALARLM_OPENAI_CACHE=1` and `SCALARLM_QUEUE_ROUTE_THRESHOLD=100`.**
Cache-busted with a unique prefix on each run so scalarlm doesn't
cache-hit.

| path | mean p/s | 3-run |
|---|---:|---:|
| openai /v1/completions (queue-routed) | **41.2** | 40.6, 41.6, 41.3 |
| scalarlm /v1/generate                 | **41.1** | 40.4, 41.9, 41.1 |

**0.2 % gap — statistical parity.** Same story as Blackwell: the
queue-routed openai path matches scalarlm because they share the same
underlying code path.

Both paths on the PR image measure **~2× the previous production
Spark pod** (which had been running `max_num_seqs=12,
gpu_memory_utilization=0.55` in its pinned `cray-config.yaml` and
measured ~18-22 p/s). That doubling is the same lesson as Blackwell —
**don't override vLLM's defaults**. The PR image honors the defaults
and picks them up automatically.

On single-GPU there's no TP/PP asymmetry to fight, so GPU utilization
is consistently ~96 % on both paths and the queue-route closure is
small in absolute terms (both paths are already near the compute
ceiling). The win is that the openai path no longer pays the proxy
fanout cost it would have paid otherwise.

## Best ways to use the openai path

- **Interactive / single request:** just call `/v1/completions` or
  `/v1/chat/completions` with OpenAI SDK conventions. This takes the direct
  path — fastest per-request latency (no queue, no extra hop).
- **Bulk batch of prompts:** pass `prompt=[...]` with the full array and
  enable `SCALARLM_QUEUE_ROUTE_THRESHOLD=100` in the pod env. Requests
  with ≥ 100 prompts transparently go through the queue worker and land
  at `/v1/generate` throughput.
- **Repeated identical batches (eval / sweep loops):** enable
  `SCALARLM_OPENAI_CACHE=1` in the pod env. First call pays full
  inference; subsequent calls with the same params return from disk in
  tens of ms.
- **Tool calling:** use `/v1/chat/completions` — this is the only path
  that supports tools. `/v1/generate` has never exposed them.
- **Streaming:** use `/v1/completions` / `/v1/chat/completions` with
  `stream=true`. The queue route is bypassed for streaming (it's
  batch-oriented); responses land directly from vLLM's streaming proxy.
- **Don't set `--max_num_seqs` in `SCALARLM_VLLM_ARGS`.** The vLLM default
  of 256 is usually the right choice for this hardware class; lower
  values cap throughput and higher values hit diminishing returns.

## Deprecation plan for `/v1/generate`

`/v1/generate` is **kept** internally — Phase 31 reuses its worker
loop — but **deprecated as a public surface**. New external callers
should use `/v1/completions` with the queue-route threshold set.

**Phase 1 — this PR:**
- Document `/v1/generate` as deprecated. Add a `Deprecation: true` HTTP
  response header (follow-up PR, trivial) and a logged warning on
  external use distinguished by request origin.
- OpenAI path reaches feature + performance parity for bulk workloads.
- SDKs that wrap `/v1/generate` (`masint`, etc.) keep working unchanged;
  they call internally and `generate()` still works.

**Phase 2 — after the `usage.token_count` plumb-through:**
- Flip `SCALARLM_QUEUE_ROUTE_THRESHOLD` default to 100 in the base pod
  templates so new deployments inherit the optimised behavior.
- Update internal docs pointing callers at `/v1/completions`.

**Phase 3 — after at least one release with Phase 2 in effect:**
- Remove the `/v1/generate` external route registration from
  `main.py`; keep `generate()` as an internal function called by the
  queue-route path.
- `masint` SDK is updated to call `/v1/completions` directly. The
  `SupermassiveIntelligence` API stays the same shape externally; only
  the wire call changes.

Estimated cadence: Phase 2 follows within a release of Phase 1; Phase 3
follows after one full release cycle of Phase 2 to catch long-running
clients.

## Benchmark repro

All measurements in this doc use two clients:

- `bench/client/path_openai_completions_array.py` — posts a single array
  `/v1/completions` with N prompts, reports p/s + total tokens.
- `bench/client/path_scalarlm_generate_bulk.py` — posts a single array
  `/v1/generate` with N prompts, reports p/s.

To repeat the headline benchmark (N=1000 distinct cold):

```bash
# Inside the pod, clear caches to force cold inference:
rm -rf /app/cray/inference_requests/openai_cache
rm -f /app/cray/inference_requests/*.json

# openai path (direct or queue-routed depending on
# SCALARLM_QUEUE_ROUTE_THRESHOLD env):
python3 bench/client/path_openai_completions_array.py \
  --url http://localhost:8000 \
  --model "$SCALARLM_MODEL" \
  --prompt-count 1000 --max-tokens 16 --distinct-prompts

# scalarlm path:
python3 bench/client/path_scalarlm_generate_bulk.py \
  --url http://localhost:8000 \
  --model "$SCALARLM_MODEL" \
  --prompt-count 1000 --max-tokens 16 --distinct-prompts
```

Both clients default to 1000 identical prompts if `--distinct-prompts`
is omitted; the identical case hits cache on the second call and is
useful for validating cache behavior but not for measuring inference.

To measure cache-hit throughput:
```bash
# Set SCALARLM_OPENAI_CACHE=1 in the pod env first.
# First call populates cache; second call should return in <50 ms.
python3 bench/client/path_openai_completions_array.py \
  --url http://localhost:8000 \
  --model "$SCALARLM_MODEL" \
  --prompt-count 1000 --max-tokens 16 --distinct-prompts  # cold
python3 bench/client/path_openai_completions_array.py \
  --url http://localhost:8000 \
  --model "$SCALARLM_MODEL" \
  --prompt-count 1000 --max-tokens 16 --distinct-prompts  # warm
```

To sample GPU utilization alongside the bench:
```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,nounits -l 1 > /tmp/gpu.csv &
SMI=$!
# ... run the bench client ...
kill $SMI
```

For the PP-vs-TP A/B on Blackwell: change `SCALARLM_VLLM_ARGS` between
`--tensor-parallel-size=2 --max_num_seqs=256 --max-cudagraph-capture-size=8 --kv-cache-dtype=fp8`
and
`--pipeline-parallel-size=2 --max_num_seqs=256 --max-cudagraph-capture-size=8 --kv-cache-dtype=fp8`,
with `SCALARLM_TENSOR_PARALLEL_SIZE=1` for the PP variant. Each config
change requires a pod restart (KV-cache pool + CUDA graphs allocate at
engine boot).
