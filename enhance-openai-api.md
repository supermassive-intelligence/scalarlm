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

## Deprecation plan for `/v1/generate`

See final commit in this PR. Short version: keep the handler, deprecate as
a public surface, leave the internal queue worker in place because Phase 31
relies on it.

## Performance summary

See final commit in this PR (populated once all phases are committed).

## Benchmark repro

See final commit in this PR.
