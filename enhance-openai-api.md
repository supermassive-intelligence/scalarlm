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
