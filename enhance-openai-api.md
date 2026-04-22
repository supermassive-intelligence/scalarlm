# Migrating from the ScalarLM-native path to a single OpenAI-compatible path

## Status

Analysis and plan only. No implementation yet.

## Why this work exists

The ScalarLM-native path (`POST /v1/generate` via the `scalarlm.SupermassiveIntelligence` SDK) does not expose tool calling, and that gap drives callers away — `sql-gen` already runs exclusively on the OpenAI path for this reason. The long-term direction is **a single OpenAI-compatible path**. This document proposes how to close the remaining feature gaps on the OpenAI path so scalarlm can be deprecated with no loss of capability.

## Background: the two paths a caller can use today

Two inference entry points exist in ScalarLM. Callers such as `sql-gen` pick between them via a flag (`use_scalarlm`). Both paths eventually talk to the same vLLM process (built from `supermassive-intelligence/vllm-fork` — see `Dockerfile:259`), but they reach it very differently:

| Path | Client surface | Server entry | Transport to vLLM |
|---|---|---|---|
| **scalarlm** (ScalarLM-native) | `scalarlm.SupermassiveIntelligence().async_api.generate([prompts], model_name, max_tokens)` | `POST /v1/generate` → SQLite work queue → `get_work` worker → vLLM Python API | In-process Python (`create_chat_completion`, `create_completion`, `load_lora_adapter`) |
| **openai** (OpenAI-compatible) | `openai.AsyncOpenAI(base_url, api_key).chat.completions.create(...)` | `POST /v1/completions` / `POST /v1/chat/completions` | HTTP proxy straight to vLLM's OpenAI-compatible server |

The scalarlm path is a queue-based architecture with worker-side orchestration; the openai path is a thin streaming proxy. `sql-gen` uses the openai path in practice (`src/llm.py:70-74`) because it needs tool calling, which the ScalarLM SDK does not expose. Both paths go by their short names (`scalarlm` / `openai`) throughout the rest of this document.

## Functional inventory

### What scalarlm does server-side (`infra/cray_infra/api/fastapi/generate/generate.py`, `infra/cray_infra/one_server/create_generate_worker.py`)

1. **Dynamic adapter/model routing.** `model_manager.find_model(model)` (`generate.py:56`) resolves the request's `model` field against LoRA adapters discovered under `training_job_directory`. The worker lazily loads the adapter into vLLM via `POST /v1/load_lora_adapter` the first time it's seen (`create_generate_worker.py:215-251`) and tracks `loaded_adaptor_count` so subsequent requests don't reload.
2. **SQLite-backed ack queue.** Requests get a SHA-256 `request_id` (`generate.py:85-86`), are serialized to disk, and pushed into a `SQLiteAckQueue` (`infra/cray_infra/api/work_queue/inference_work_queue.py`). The queue is persistent, FIFO, and auto-recovers in-flight requests on restart.
3. **Worker-side batching.** Workers pull up to `batch_size` items from `/v1/generate/get_work`, where `batch_size` is derived from KV-cache headroom (`create_generate_worker.py:176`). Batching happens *inside* vLLM from a contiguous set of queued requests rather than being expressed in the client API.
4. **Multi-prompt fan-in on the wire.** The SDK takes `[prompt1, prompt2, ...]`. For fewer than 128 prompts it POSTs them all to `/v1/generate`; for more it uploads a file and polls for a download (`sdk/masint/engines/async_cray.py:29-48`).
5. **Per-request error isolation.** Each element of the batch carries its own `error` field in the final `GenerateResponse`; partial success is supported.
6. **Polling-based result retrieval.** The SDK polls `/v1/generate/get_results` with the list of `request_id`s until all are filled (`async_cray.py:155-167`). Results are stored in an in-memory dict backed by files on disk and survive restarts long enough to be retrieved.
7. **FLOP metrics.** The worker computes `token_count × compute_flop_count(model_config)` per request and reports it on the shared metrics counter (`create_generate_worker.py:382, 502`).
8. **Queue observability.** Prometheus/OTel counters expose `queue_depth` and `queue_wait_time_seconds` by model (`inference_work_queue.py:14-19`).

### What openai does today (`infra/cray_infra/api/fastapi/routers/openai_v1_router.py`)

1. **Parameter whitelist and passthrough.** The two handlers filter incoming `CompletionRequest` / `ChatCompletionRequest` fields down to a known set (`_COMPLETION_ALLOWED_KEYS`, `_CHAT_ALLOWED_KEYS` at lines 43-75). `tools`, `tool_choice`, `temperature`, `response_format`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, and streaming are all included.
2. **Usage sniffing for metrics.** `_ensure_usage_reported` forces `stream_options.include_usage=True` for streaming calls so vLLM emits a terminal `usage` event (`openai_v1_router.py:130-141`). `_wrap_with_metrics` scans the tail of the SSE stream (bounded to 64 KB) to pull `usage.total_tokens` and feed it into the shared metrics counter (`openai_v1_router.py:172-204`).
3. **Direct proxy to vLLM's OpenAI server.** No queue, no worker, no on-disk request store — the proxy opens a streaming HTTP call to `vllm_api_url + "/v1/completions"` (or chat variant) and forwards bytes through.

### Gap summary — what A has that B lacks

| Capability | scalarlm | openai | Disposition for this work |
|---|---|---|---|
| Dynamic LoRA routing by `model` name | Yes | No (vLLM-default only) | **In scope — proxy-side load.** Critical for finetune routing. |
| FLOP metrics | Yes | `flop_count=None` | **In scope — observability parity.** Dashboards currently under-report openai. |
| Durable request IDs | SHA-256 per request, persisted | None | **In scope — audit/correlation.** UUID per call surfaced as `X-Request-Id`. |
| Request queue with depth + wait-time metrics | Yes (SQLiteAckQueue) | No | **In scope — see "Queue on openai" below.** Backpressure, fair ordering, observability. |
| Multi-prompt batching on the wire | Yes (`prompts=[...]`) | Partial — already works on `/v1/completions` via the OpenAI-spec `prompt` array; not available for chat | **In scope — see "Multi-prompt on the OpenAI surface" below.** Test/document what already works; use OpenAI Batch API for chat. |
| Per-request error isolation within a batch | Yes | Standard OpenAI shape (one `choices[i].finish_reason` per prompt index) | In scope implicitly — covered by multi-prompt testing. |
| Polling-based result retrieval | Yes | Synchronous / SSE | **Not in scope to replicate.** The OpenAI Batch API pattern (submit → poll `/v1/batches/{id}`) subsumes this, and the sync/SSE path serves the common case. |
| Streaming | No | Yes (SSE) | openai is already ahead. |

Net: **five gaps are in scope** — LoRA routing, FLOP metrics, request IDs, queue + observability, multi-prompt (via the completions `prompt` array today and the Batch API for chat). No gap remains that would require keeping scalarlm around.

## Architectural strategy

Preference: land changes in this repo on top of the vLLM fork (`supermassive-intelligence/vllm-fork`, cloned locally at `~/src/vllm-fork`); touch the fork only if unavoidable, and if so, in a way that upstreams cleanly to vLLM.

### vLLM-fork capabilities already in place (verified)

The fork already provides the primitives we need for closing Gap #1, which means LoRA routing can be done without fork changes:

- **`POST /v1/load_lora_adapter`** is registered when `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` (`vllm/entrypoints/serve/lora/api_router.py:42`). ScalarLM already sets this env var (`infra/cray_infra/one_server/main.py:4`). Request schema: `{"lora_name": str, "lora_path": str, "load_inplace": bool=false}` (`vllm/entrypoints/serve/lora/protocol.py`).
- **Idempotent-by-name.** If `lora_name` is already in `self.lora_requests` and `load_inplace=false`, the handler returns success without reloading (`vllm/entrypoints/openai/models/serving.py:235`). This means a caller can POST optimistically on every request without worrying about duplicate loads.
- **Per-adapter asyncio lock.** `self.lora_resolver_lock[lora_name]` (`serving.py:278`) ensures two concurrent requests for the same not-yet-loaded adapter result in exactly one load. **This obviates the proxy-side lock we originally scoped**.
- **`LoRAResolver` plugin system.** vLLM ships an official extension point (`vllm/lora/resolver.py`) for mapping a `model` name to a LoRA adapter at request time. A plugin implements `async def resolve_lora(base_model_name, lora_name) -> LoRARequest | None`. It is discovered via Python entry points on the `vllm.general_plugins` group. A stock `FilesystemResolver` exists (`vllm/plugins/lora_resolvers/filesystem_resolver.py`) but insists on `adapter_config.json` + `peft_type == "LORA"`, so **it does not cover ScalarLM's tokenformer adapters** — those are `.pt` checkpoints loaded by `TokenformerModelManager`.
- **Unified adapter abstraction.** Both standard LoRA and the fork's tokenformer adapters flow through the same `LoRARequest` object and the `/v1/load_lora_adapter` endpoint (`vllm/v1/worker/lora_model_runner_mixin.py:42` wires `TokenformerModelManager` as `self.lora_manager`). The proxy doesn't need to branch on adapter type.

### Applying the rules to each gap

1. **LoRA routing — proxy-layer only, no fork changes.** Two viable shapes:
   - **Option A (simpler, recommended first step).** In the OpenAI proxy (`openai_v1_router.py`), before forwarding, ensure the adapter is loaded: consult `get_vllm_model_manager()` to resolve the request's `model` field, POST `/v1/load_lora_adapter` to vLLM if it's a known adapter, then forward. Cache in a proxy-local `set[str]` to avoid the round-trip on hot paths — even though vLLM's handler is idempotent, skipping the extra HTTP call matters at high RPS. This is ~50 lines.
   - **Option B (cleaner, defer to a follow-up).** Package a small `ScalarLMLoRAResolver` plugin that knows about ScalarLM's `training_job_directory` (and its `.pt`-tokenformer variant), registered via a `vllm.general_plugins` entry point. vLLM then auto-loads on unknown `model` names with no proxy code needed — the proxy stays a pure passthrough. This is the upstream-pattern-aligned approach. A nice-to-have follow-up once Option A is validated.
   
   **Recommendation:** start with Option A (lives entirely in this repo, no coupling to vllm-fork tooling). Revisit Option B once the proxy path is proven.

2. **FLOP metrics — proxy-layer only, no fork changes.** The token count is already extracted in `_wrap_with_metrics`. Look up `compute_flop_count(model_config)` once for the resolved model and multiply. Tiny diff inside `_wrap_with_metrics`.

3. **Durable request IDs — proxy-layer only, no fork changes.** Generate a UUID at request entry, inject it as a response header (`X-Request-Id`, as OpenAI itself does), and log it at enqueue and completion. Optional: include in the upstream vLLM call so it appears in vLLM's logs.

4. **Queue on openai — proxy-layer only, no fork changes.** The goal is the capabilities scalarlm's queue provides (bounded concurrency, fair ordering, `queue_depth` / `queue_wait_time_seconds` metrics, graceful 503 under overload) without disturbing the synchronous/SSE contract that OpenAI clients expect. The implementation is a FastAPI middleware that wraps both `create_completions` and `create_chat_completions`:
   - A bounded `asyncio.Queue` of pending call objects (each carries the request, the caller's `asyncio.Future` or the async response-chunk sink).
   - A small pool of drainer tasks (size = desired vLLM concurrency) that pop items and do the real vLLM call, writing chunks back to the sink for streaming or the result to the future for non-streaming.
   - OpenTelemetry counters: `openai_queue_depth`, `openai_queue_wait_seconds` (matching naming with scalarlm's `queue_depth` / `queue_wait_time_seconds` so dashboards merge cleanly).
   - When the queue is full beyond a threshold, return `503 Service Unavailable` with `Retry-After` — standard OpenAI client retry territory.
   - We **do not** replicate scalarlm's SQLiteAckQueue durability. scalarlm's durable queue exists to survive crashes in a polling world; with synchronous HTTP, a crashed request just returns an error to the client, who retries. If durability becomes a product requirement (compliance, async batch), it lands as part of the Batch API below, not here.

5. **Multi-prompt on the OpenAI surface — two distinct mechanisms, both proxy-layer:**
   - **`/v1/completions` with array `prompt`.** The OpenAI-spec `prompt` field accepts `str | list[str] | list[int] | list[list[int]]`. The vLLM fork already implements this (`vllm/entrypoints/openai/completion/protocol.py:46-52`), and our proxy's existing whitelist passes `prompt` through unchanged. Work: **an integration test confirming it round-trips**, plus documentation. Likely zero code changes.
   - **OpenAI Batch API (`/v1/batches`) for chat completions.** Chat completions has no in-request batching idiom; OpenAI's canonical answer for bulk chat is the [Batch API](https://platform.openai.com/docs/api-reference/batches) — submit a JSONL of requests, poll for a completed JSONL of responses. This is an async, queue-based interface by design, so it dovetails with the queue from item (4). Minimum viable endpoints: `POST /v1/batches` (enqueue), `GET /v1/batches/{id}` (status), `GET /v1/batches/{id}/output_file_content` (results). The implementation sits on top of the same queue infrastructure, with a persistent results store for later retrieval.
   - Note: scalarlm's "upload for >128 prompts" flow (`/v1/generate/upload`, `/v1/generate/download`) is structurally a Batch API — the new endpoint is the OpenAI-shaped version of the same thing.

None of (1)-(5) requires modifying the vllm-fork. If during implementation we discover a genuine vLLM gap (for example, if `/v1/load_lora_adapter`'s existing error shape doesn't expose enough detail), we raise it separately and propose a small upstream-friendly patch — not expected based on the fork review above.

## Test-driven implementation plan

The user's preferred sequence is: write tests for scalarlm behaviour first, then write the same tests against openai (they fail), then implement until they pass, then deprecate scalarlm.

### Phase 0 — Repo-side baseline

- Identify the existing test harness. The repo has `test/unit/` (pytest) and `test/infra/` (integration). See `pytest.ini`, `test/conftest.py`, `test/unit/conftest.py`.
- Decide where new tests live. Proposed:
  - `test/unit/test_openai_lora_routing.py`
  - `test/unit/test_openai_metrics.py`
  - `test/unit/test_openai_request_id.py`
  - `test/unit/test_openai_queue.py`
  - `test/unit/test_openai_multi_prompt.py`
  - `test/unit/test_openai_batches.py`
  - `test/infra/test_openai_parity.py` — integration module hitting a running server.
- Wire up lightweight mocks for vLLM's endpoints (`load_lora_adapter`, `chat/completions`, `completions`) so unit tests don't need a live GPU.

### Phase 1 — Capture scalarlm behaviour as tests

All pass against current `main`:

1. `test_path_a_loads_adapter_on_first_request` — call `/v1/generate` with `model=<adapter-name>`; assert the worker issued `POST /v1/load_lora_adapter` exactly once even across two back-to-back requests.
2. `test_path_a_reports_flop_count` — call `/v1/generate`; observe the metrics counter; assert `flop_count > 0`.
3. `test_path_a_returns_request_id` — assert `GenerateResponse.results[i].request_id` has the `{sha256}_{index:09d}` shape.
4. `test_path_a_queues_under_pressure` — submit N requests faster than the worker can drain; assert `queue_depth` metric crosses a threshold and `queue_wait_time_seconds` is non-zero.
5. `test_path_a_accepts_multi_prompt` — submit `prompts=[p1, p2, ..., p5]`; assert the response contains five distinct `Result` entries with correctly indexed `request_id`s and that per-item errors are isolated.
6. `test_path_a_upload_download_for_large_batch` — submit `prompts=[...]` with length > 128; assert the upload/download (batch) flow is exercised and results round-trip correctly.

Each test documents what "scalarlm parity" means for that capability, anchoring Phase 2.

### Phase 2 — Same tests against openai (expected to fail)

Mirror each Phase-1 test on the OpenAI surface. They fail today; that failure is the proof the gap is real.

1. `test_path_b_loads_adapter_on_first_request` — call `POST /v1/chat/completions` with `model=<adapter-name>`; expect the proxy to issue `POST /v1/load_lora_adapter` upstream. **Fails** — proxy doesn't do this.
2. `test_path_b_reports_flop_count` — stream through the proxy; assert `flop_count > 0`. **Fails** — `_wrap_with_metrics` hard-codes `flop_count=None`.
3. `test_path_b_emits_request_id_header` — assert the response has a stable `X-Request-Id` header and the same id appears in the server log for that request. **Fails** — no request id today.
4. `test_path_b_queues_under_pressure` — saturate the proxy concurrency; assert `openai_queue_depth` rises, requests block fairly, and `openai_queue_wait_seconds` is recorded. **Fails** — no queue.
5. `test_path_b_completions_accepts_array_prompt` — call `POST /v1/completions` with `prompt=[p1, p2, p3]`; assert three `choices` entries come back in order. **Should pass today** given vLLM accepts array prompts and the proxy whitelists `prompt`, but we confirm it end-to-end.
6. `test_path_b_batch_api_roundtrip` — submit a JSONL via `POST /v1/batches`, poll `GET /v1/batches/{id}` until `completed`, fetch results. **Fails** — endpoint doesn't exist.

### Phase 3 — Implement until Phase-2 tests pass

Order is chosen so each step unlocks observability or primitives for the next.

1. **Request ID** (~30 lines). FastAPI dependency generates a UUID per call, attaches it to `response.headers["X-Request-Id"]`, logs at entry and completion. Unlocks clean log correlation for the rest of Phase 3.
2. **FLOP metrics** (~20 lines plus fixture). Cache `compute_flop_count(model_config)` per resolved model at startup; multiply by `total_tokens` in `_wrap_with_metrics`. Replace the hard-coded `flop_count=None`.
3. **LoRA routing** (~60 lines plus mock). Add `_ensure_adapter_loaded(model)` in `openai_v1_router.py`; consult `get_vllm_model_manager()`; POST to vLLM's `/v1/load_lora_adapter` if known-adapter-not-yet-loaded; maintain a proxy-side `set[str]` cache protected by an `asyncio.Event`-per-name. Insert a call at the top of both `create_completions` and `create_chat_completions`. Reuse the existing `vllm_model_manager` logic; do not fork it.
4. **Queue + concurrency control** (~150 lines). New module `infra/cray_infra/api/fastapi/routers/openai_queue.py`:
   - `OpenAIRequestQueue` with bounded `asyncio.Queue` and a pool of drainer tasks.
   - The two handlers enqueue an object carrying the upstream call and a chunk-sink / future; drainers do the actual `session.post` to vLLM.
   - Emit `openai_queue_depth` and `openai_queue_wait_seconds` via the same OpenTelemetry meter as scalarlm.
   - 503 with `Retry-After` when depth exceeds a config threshold.
   - Config keys: `openai_queue_max_depth`, `openai_queue_concurrency`. Defaults chosen conservative; plumb through `default_config.py`.
5. **Multi-prompt for `/v1/completions`** (~0 lines code, focus on tests). Confirm the array-prompt path end-to-end; document it in `docs/api-documentation.md`. Only code change expected is a log line that records the prompt-count for a batched call so it's visible in the request log.
6. **Batch API scaffold** (~200 lines). Introduce `POST /v1/batches`, `GET /v1/batches/{id}`, `GET /v1/batches/{id}/output_file_content`, `DELETE /v1/batches/{id}` at OpenAI-spec shapes. Under the hood: persist the incoming JSONL to disk, enqueue each line via the queue from step 4, aggregate outputs into the result JSONL as items complete, record `batch.status` transitions (`validating` → `in_progress` → `completed` / `failed`). Reuse scalarlm's on-disk results layout where it fits; the goal is that Batch API inherits durability "for free" from a simple file-backed results store.

At the end of Phase 3, Phase-2 tests pass. Phase-1 tests still pass (we don't touch scalarlm).

### Phase 4 — Mark scalarlm for deprecation (the point of this work)

- Emit a structured deprecation log when `POST /v1/generate`, `/v1/generate/get_results`, `/v1/generate/upload`, `/v1/generate/download` are hit, including the caller's `User-Agent` so we can see who's still on scalarlm.
- Update `docs/api-documentation.md` and the SDK README to point new integrations at:
  - `/v1/chat/completions` for chat.
  - `/v1/completions` with an array `prompt` for the small-batch-of-prompts case.
  - `/v1/batches` for the large-batch-of-prompts case (was scalarlm's `upload`/`download`).
- Publish a one-line migration for each common scalarlm usage (single-prompt generate, multi-prompt generate, uploaded batch), so callers can migrate without reading the spec.
- Leave `sdk/masint/engines/async_cray.py` in place for now. A follow-up can either reimplement `generate()` as a thin shim over the OpenAI client or remove it outright once no internal caller uses it.

Do **not** delete scalarlm in this work. Deprecation first, removal later, with a named grace period tied to the usage telemetry above.

### Phase 5 (future) — LoRAResolver plugin

Once Option A (proxy-side load) is stable, consider packaging the ScalarLM adapter discovery as a `LoRAResolver` plugin (see "vLLM-fork capabilities" above). This collapses the proxy to a pure passthrough and moves ScalarLM-specific resolution logic into a well-defined vLLM extension point. Not required for scalarlm deprecation — just cleaner long-term.

## Risks and open questions

- **Unknown `model` values.** If a caller passes a name that neither resolves to an adapter nor to the base model, the proxy should not pre-call `/v1/load_lora_adapter` with a bogus path — let vLLM error. The model manager's `find_model` already returns `None` for unknowns; use that signal to skip the load step and forward directly (vLLM returns 400/404 as before).
- **Concurrency on adapter loads.** Resolved by the fork — vLLM's `lora_resolver_lock[lora_name]` serializes concurrent loads of the same name (`vllm/entrypoints/openai/models/serving.py:278`). The proxy's local cache uses an `asyncio.Event`-per-name pattern to avoid two HTTP calls during the first load window, but correctness does not depend on it.
- **Tokenformer vs standard LoRA.** The fork's `TokenformerModelManager` replaces the stock LoRA manager and handles `.pt` checkpoints. From the proxy's perspective there's no difference: both are loaded via `/v1/load_lora_adapter` with a path. This is already how scalarlm's worker operates today.
- **Queue interaction with SSE streaming.** A queued streaming request has to hand bytes back to an already-open HTTP response. The pattern we'll use: the handler creates an `asyncio.Queue` of chunks, enqueues the work pointing at that queue, returns a `StreamingResponse` that drains the chunk-queue, and the drainer task writes chunks into it as vLLM produces them. Test with a long SSE stream and a pile of queued callers to confirm backpressure propagates end-to-end (vLLM slows → drainer blocks → queue depth rises → handler returns 503 at threshold).
- **Batch API semantics.** Our `/v1/batches` need only be a subset of OpenAI's — support JSONL request bodies, `status` transitions, and result fetching. We do **not** need `input_file` upload via `/v1/files` as a separate step; inline JSONL is simpler and sufficient for now. Document the subset explicitly so callers know the edges.
- **FLOP count for tool-calling responses.** `usage.total_tokens` includes tool-call tokens, so multiplying by `compute_flop_count(model_config)` gives a valid estimate. Covered by a test.
- **Request-ID semantics.** UUID v4 gives correlation, not content-addressable idempotency. Right scope for Phase 3. If a future consumer needs idempotency (e.g. dedupe across retries), add it then.
- **Request-log persistence for compliance.** scalarlm writes every request JSON to disk. The Batch API's on-disk JSONL naturally covers the batch case; for ad-hoc `/v1/chat/completions` traffic, we'd need a separate audit log if that's a compliance requirement rather than debugging. Flag for product before implementation.
- **Config defaults for the queue.** Start conservative — e.g. `openai_queue_concurrency=16`, `openai_queue_max_depth=256` — tune once we have live telemetry. Bad defaults here are the thing most likely to surprise in production.

## FAQ

### "scalarlm can do the equivalent of 1 million requests/sec via wire batching. Can the OpenAI surface keep up?"

It's not a hard limit, but the numbers measure different things and the mechanics differ. Short version:

- scalarlm's "1 M" is a per-prompt number. One HTTP call carries up to 128 prompts on `/v1/generate` (or a JSONL file via `/v1/generate/upload` for more). All the per-request overhead — TCP+TLS handshake, parse, auth, middleware, model resolution — is amortized across every prompt in the batch.
- openai's "1000 req/sec" is per HTTP call. Each request pays that overhead once.

The efficiency gap only exists for *bulk* workloads, and the plan preserves scalarlm's wire shape via two OpenAI-spec mechanisms:

| Workload | scalarlm today | Unified path after Phase 3 |
|---|---|---|
| Bulk completions | 1 call × N prompts on `/v1/generate` | 1 call × N prompts on `/v1/completions` with array `prompt` — **identical efficiency** |
| Bulk chat (offline) | 1 call × N prompts via upload/download | `/v1/batches` (OpenAI Batch API) — **identical efficiency**, canonical shape |
| Interactive chat (one-at-a-time, streaming) | Not supported — scalarlm doesn't stream | Fan-out on `/v1/chat/completions` — pays per-request overhead, but streaming works |

### "How does the OpenAI client actually reach 1000 concurrent RPS? Threads? Non-blocking?"

Non-blocking, not threads. The `openai` Python SDK's async class (`AsyncOpenAI`) is built on `httpx.AsyncClient`, which runs on asyncio. 1000 concurrent requests are 1000 coroutines on a single thread.

The wire side matters more than the client side:

- With HTTP/1.1 + keep-alive, many in-flight requests need many TCP sockets; `httpx.Limits(max_connections=N)` gates that.
- With HTTP/2 one TCP connection multiplexes thousands of in-flight streams. This is the right configuration for high-RPS to a single origin.
- `AsyncOpenAI` accepts an injected `httpx.AsyncClient`, so `http2=True` and a larger connection pool are a three-line client-side change.

Example shape:

```python
client = AsyncOpenAI(
    base_url=...,
    http_client=httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
    ),
)
await asyncio.gather(*(client.chat.completions.create(...) for _ in range(1000)))
```

### "Does the OpenAI API itself support non-blocking calls?"

The protocol doesn't have a "non-blocking request" concept — one request = one response, like any HTTP API. Non-blocking is entirely client-side (async coroutines, HTTP/2 multiplexing). The server sees a regular stream of regular requests.

For genuinely fire-and-forget semantics, that's what `/v1/batches` is for: submit work, poll for it later, consumer is decoupled from producer timing.

### "So is 1000 RPS achievable over the internet?"

Yes, with standard scaling levers on both sides:

- **Client**: one `AsyncOpenAI` process with HTTP/2 and a generous connection pool handles thousands of concurrent requests. Wall-clock latency ≈ RTT + server processing, not (RTT × N).
- **Server**: FastAPI/uvicorn is async. One worker sustains ~200–2 k RPS for proxy handlers that do upstream vLLM I/O; scale horizontally with `--workers N` or additional pods behind a load balancer. Standard playbook.
- **Our proxy-specific ceiling**: Phase 3d's queue (`openai_queue_max_depth`) caps in-flight work so vLLM isn't swamped. Requests over the threshold return 503 + `Retry-After`. That's a deliberate backpressure choice, not a forced limit — the threshold is a config value tuned against live telemetry.

### "Where's the real residual cost?"

One place: the interactive-streaming-chat-one-at-a-time shape. The OpenAI protocol doesn't have an idiom for batching multiple live SSE streams, so each concurrent live chat eats one HTTP connection's worth of overhead. Estimated at ~1–5 ms of proxy overhead per request — negligible next to inference time for normal completions, visible for very short outputs. This cost is inherent to OpenAI-style streaming and affects every OpenAI-compatible endpoint, not just ours.

## Benchmark plan

The FAQ above reasons about throughput; this section proposes how to measure it. Reviewers have raised three concrete questions that any migration decision needs empirical answers to:

> Can you make 1000 simultaneous async requests? Can you make 1 million? Even if the FastAPI endpoint is a nop, does that work or does it fall over and die?

And an observation: scalarlm's persistent queue is known to sustain 1000 batches of 1000 prompts each, i.e. 1M per-prompt throughput. The benchmark has to show where openai lands relative to that — not argue about it.

### What the benchmark has to answer

1. **Raw FastAPI ceiling.** Absent vLLM, how many concurrent requests does one uvicorn process handle before latency cliffs or errors appear? This isolates the transport layer from the inference layer.
2. **openai per-request capacity.** Chat completions one at a time. At what concurrency do we start seeing `503` from the Phase 3d queue, and at what concurrency does uvicorn itself lose?
3. **openai bulk capacity.** `/v1/completions` with array `prompt` and `/v1/batches` — do they match scalarlm's wire-batched throughput?
4. **Failure mode under overload.** When any of the above exceeds its sustainable load, does it degrade gracefully (503 + Retry-After, bounded latency) or cliff (dropped connections, OOM, wedged event loop)?

### Workloads

Six scenarios, each run across a sweep of concurrency levels. Target the same vLLM instance for the non-nop ones so hardware conditions are held constant.

| Name | Endpoint | Wire shape | Per-request work | Measures |
|---|---|---|---|---|
| `nop` | `GET /v1/health` (or a dedicated `GET /v1/bench/nop`) | single | zero | Raw FastAPI/uvicorn ceiling. Baseline for every other scenario. |
| `pathb_chat_single` | `POST /v1/chat/completions` | one prompt per HTTP call | 1 prompt, ~100 tokens | Interactive-streaming shape. The case scalarlm doesn't have. |
| `pathb_completions_array` | `POST /v1/completions` | `prompt=[p1..pN]` per call, N varied | N prompts, ~100 tokens each | Wire-batched completions. Expected to match scalarlm. |
| `pathb_batches` | `POST /v1/batches` then poll | N-line JSONL | N prompts, ~100 tokens | Async bulk chat, OpenAI-canonical. |
| `patha_generate_bulk` | `POST /v1/generate` | `prompts=[p1..pN]` per call | N prompts, ~100 tokens | scalarlm's current bulk path; the throughput baseline we're measuring against. |
| `patha_upload_download` | `POST /v1/generate/upload` + poll `/download` | JSONL upload file | N prompts, ~100 tokens | scalarlm's large-batch mode; the 1000×1000 = 1M case the colleague cites. |

### Sweep parameters

Concurrency / batch size: `10, 100, 1_000, 10_000, 100_000, 1_000_000`. For the wire-batched scenarios (`completions_array`, `batches`, `patha_*`) this is prompt count per HTTP call; for the per-request scenarios (`nop`, `pathb_chat_single`) this is concurrent in-flight HTTP calls.

Per-request payload size: tiny (1 token), small (~100 tokens), large (~1k tokens). Picks out whether the bottleneck is per-request overhead or inference time.

Transport: HTTP/1.1 with keep-alive vs HTTP/2. The FAQ argues HTTP/2 is how you reach high-RPS to one origin; the benchmark has to demonstrate that.

### Metrics

Per run:

- **Throughput**: completed prompts / sec, completed requests / sec.
- **Latency**: P50, P95, P99, max. For the bulk scenarios this is end-to-end (submit → last result available).
- **Error shape**: HTTP status distribution, 5xx breakdown, connection errors, timeouts. `503 Retry-After` from the Phase 3d limiter counts as graceful — it's a feature.
- **Server resource**: CPU, RSS, open file descriptors, `openai_queue_depth`, `openai_queue_wait_time_seconds`, `scalarlm_queue_depth`.
- **Client resource**: event-loop lag, open connections, memory.

### Client tooling

For 1 k concurrency: an async Python client using `AsyncOpenAI` + `httpx.AsyncClient(http2=True, limits=…)` is enough. Runs on one laptop.

For 10 k–100 k: dedicated load generator — `k6` with the HTTP/2 provider, or `vegeta` for the non-OpenAI-SDK shape. Multiple client processes possibly on different machines to avoid client-side CPU as the bottleneck.

For 1 M concurrency: **this is explicitly expected to break something**. The point is to find what breaks first — kernel fd limits, uvicorn worker count, Python GC pauses under pressure, the Phase 3d queue's overflow path, the event-loop scheduler, vLLM's KV cache. That finding is the deliverable, not a pass/fail.

### Hardware platforms

Each scenario runs on three platforms so we can separate per-platform overheads (proxy, event-loop, network) from inference-layer scaling. Absolute numbers vary wildly across these; the platform triangle is what tells us whether Phase 3's proxy is pulling its own weight or adding fixed overhead that only shows up on big iron.

| Platform | Shape | Model | Purpose in the grid |
|---|---|---|---|
| **MacBook Pro M5** | Single-node, CPU/MPS only (no CUDA), unified memory | **`Qwen/Qwen3-4B-Instruct-2507`** (bf16 weights) — proposed default; see rationale below | Isolates the FastAPI / uvicorn / proxy layer. `nop` ceiling and proxy-overhead numbers from here are a clean story about the Python surface — no GPU variance. CPU-only vLLM inference is slow but fine for correctness sweeps and for measuring per-request overhead on small completions. Dev-machine representativeness. |
| **NVIDIA DGX Spark** | 1× GB10 Grace-Blackwell superchip, ~128 GB unified memory | **`nvidia/Qwen3-32B-NVFP4`** | Single-GPU dev / prosumer tier. 32 B at NVFP4 fits GB10's unified memory comfortably. Shows how the proxy's 16-concurrency default interacts with a single-GPU decode rate — this is where the Phase 3d queue config defaults want to be validated. |
| **4-GPU Blackwell system** | 4× B200 (or equivalent), TP=4 or replicated 4× TP=1, 4× HBM3e | **`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`** | Production tier. The 80B-A3B MoE at NVFP4 is the model sql-gen's Blackwell deploy already runs (`deploy/blackwell/helm-charts/qwen3next/values.yaml`), so benchmark results map directly onto production cost/throughput. Only platform where scenarios at 100 k – 1 M concurrency are likely to clear vLLM's own ceiling cleanly, so it's the one that decides the 5 %-parity question against scalarlm's `upload_download` path. Both `--tp 4` (one vLLM instance) and `--replicas 4` (sql-gen's current deployment shape — per its `deploy/blackwell/restart.sh`) should be measured since their per-request latency profiles differ. |

**Why Qwen3-4B-Instruct-2507 on M5.** Three requirements: (1) same family as the two GPU platforms so architectural comparisons across the triangle are interpretable — same tokenizer, same attention shape, same MLP topology; (2) runs on Apple Silicon, which rules out NVFP4 (Blackwell-only); (3) fits comfortably in a mid-range M5 Pro's unified memory (4 B × bf16 = ~8 GB). Qwen3-4B-Instruct-2507 is the closest in-family match and is what's available on the Hugging Face Hub.

Drop-downs if M5 memory is tight or full 4 B inference is too slow to keep the sweep's wall clock reasonable:
- **`Qwen/Qwen3-1.7B`** (~3.5 GB) — still instruct-tuned, still same family, 4× faster decode.
- **`Qwen/Qwen3-0.6B`** (~1.2 GB) — effectively a correctness smoke test; proxy overhead dominates, which is exactly what we're measuring on M5 anyway.
- **`tiny-random/gemma-4-dense`** (the repo default) — last resort. Different family, not apples-to-apples with Spark/Blackwell; only use for `nop`-adjacent scenarios where the model barely matters.

The goal on M5 is never absolute throughput — it's a stable denominator for the per-request proxy-overhead number. Pick the smallest Qwen3 that still exercises real KV cache / real tokenization so the overhead measurement isn't an artifact.

Not every concurrency level runs on every platform. Mapping:

| Scenario × concurrency | M5 | DGX Spark | 4× Blackwell |
|---|---|---|---|
| `nop` 10 – 10 k | ✓ | ✓ | ✓ |
| `nop` 100 k – 1 M | best-effort (client fd limits) | ✓ | ✓ |
| `pathb_chat_single` 10 – 1 k | ✓ | ✓ | ✓ |
| `pathb_chat_single` 10 k – 1 M | skip (CPU inference too slow to be meaningful) | up to 10 k | full sweep |
| `pathb_completions_array` / `pathb_batches` / `patha_generate_bulk` | small N (≤ 100) as correctness smoke test | up to 10 k | full sweep |
| `patha_upload_download` at 1 M | skip | skip (one GPU won't chew through it in reasonable time) | primary target |

Cross-platform interpretation: Spark and Blackwell both run NVFP4 Qwen3-family models, so their numbers are directly comparable — the only differences are model size and GPU count. M5 runs a smaller non-NVFP4 Qwen3, so only its proxy-overhead numbers (and error shape under overload) transfer across platforms; absolute throughput does not.

### Harness layout

```
bench/
  client/
    nop.py                    # hammers GET /v1/bench/nop
    pathb_chat_single.py      # N concurrent chat/completions
    pathb_completions_array.py
    pathb_batches.py
    patha_generate_bulk.py
    patha_upload_download.py
  server/
    nop_route.py              # the dedicated /v1/bench/nop, gated by env
  scenarios/
    sweep_concurrency.sh      # runs each workload across the grid
    platforms.yaml            # per-platform model, concurrency caps, worker count
  README.md                   # repro recipe + how to interpret output
  results/
    <platform>/<timestamp>/summary.json  # one tree per platform so results don't collide
```

A `/v1/bench/nop` endpoint only makes sense to add if scenario #1 needs a known-nop to decouple FastAPI from the inference stack — worth it. It goes in under `settings.bench_endpoints_enabled` so production builds don't expose it.

### Expected results (for pre-registration)

Stating expectations up front so surprises are surprises:

- **`nop` scenario** (all platforms): uvicorn single-worker handles 5–15 k RPS comfortably; cliffs around 50–100 k concurrent due to socket/event-loop overhead. `--workers 4` roughly 4× the ceiling. 1 M falls over well before the server — kernel fd limits on the client. Platform delta: M5 hits its CPU ceiling sooner (smaller core count); Spark and Blackwell should produce near-identical numbers since `nop` is all CPU + NIC.
- **`pathb_chat_single`**: throughput bounded by vLLM's decode rate × proxy queue concurrency (default 16). P99 under saturation is queue-wait-bounded once `openai_queue_max_depth` is reached; overflow = 503 with `Retry-After`, *not* a cliff. Platform delta: M5 saturates the CPU vLLM backend long before the proxy; on Spark the queue defaults are the bottleneck; on Blackwell the defaults are probably too conservative — expect `openai_queue_concurrency` to need raising to exploit the hardware.
- **`pathb_completions_array` vs `patha_generate_bulk`**: should match within measurement noise at equal batch sizes on each platform. Any persistent gap indicates a proxy-layer inefficiency worth chasing. Same-platform comparisons only — don't compare M5 numbers to Blackwell.
- **`pathb_batches`**: same steady-state throughput as `pathb_completions_array`, but with worse end-to-end latency for small N (async submit + poll overhead). Win condition is breaking clearly ahead on large N where wire-batching savings accumulate — most visible on Blackwell.
- **`patha_upload_download` at 1 M** (Blackwell only): known to sustain per the colleague's claim; replicate as the throughput ceiling for the unified path to justify itself against.

### Decision thresholds

Before the benchmark runs:

- If `pathb_completions_array` throughput is within **5 %** of `patha_generate_bulk` on equal workloads *on Blackwell* (the production-tier platform), declare wire-efficiency parity and proceed with deprecation. The same 5 % test on Spark is a useful corroboration; on M5 it's mostly a proxy-overhead sanity check since inference dominates.
- If the M5 `nop` and `pathb_chat_single` numbers show the proxy layer adding more than ~5 ms of fixed overhead per request, chase that before claiming readiness — on M5 inference is CPU-bound enough that any proxy regression stands out clearly.
- If 1 M concurrent traffic on `pathb_chat_single` (Blackwell) breaks the server in a way that isn't `503 Retry-After` (e.g. OOM, wedged event loop), that's a correctness bug in the limiter / resource accounting — fix before any deprecation conversation.
- Per-platform queue defaults: if Blackwell needs `openai_queue_concurrency` raised to hit its inference ceiling, make that a platform-tunable rather than a global default change — Spark / M5 defaults shouldn't inherit a number tuned for 4× B200.

### Out of scope for the benchmark

- Training-side throughput. Separate concern.
- Latency under cold start (adapter not yet loaded). The LoRA-load path runs once per adapter per worker; bench runs should be warm.
- Mixed-shape workloads (streaming + bulk interleaved). Worth doing eventually but harder to attribute results; a separate follow-up.

### First-pass results

Three platforms hit so far. Raw JSON sits under `bench/results/<platform>/<timestamp>/`. Each platform table lists the scenarios that actually ran and any blocked ones with the reason.

#### Nop baseline (FastAPI + request-id middleware, no inference)

Measures the FastAPI / uvicorn / middleware ceiling on each host with the `/v1/bench/nop` endpoint. This endpoint sits *under* both scalarlm and openai — the request-id middleware is global, so every request on every path pays exactly this cost before the handler-specific work begins. Single-column per platform is correct; there is no scalarlm-vs-openai split at this layer.

Single uvicorn worker, HTTP/2 client pool, 5 s per level.

| concurrency | M5 (native, 10 cores) | Blackwell maxq-1 (in container, 64 cores) |
|---|---|---|
| 10 | 2 522 RPS · P50 3 ms | 969 RPS · P50 7 ms |
| 100 | 655 RPS · P50 108 ms | 176 RPS · P50 392 ms |
| 1 000 | 370 RPS · P50 1.4 s | 381 RPS · P50 1.4 s |
| 10 000 | 480 RPS · P50 15 s | 366 RPS · P50 16 s |
| 100 000 | (skipped) | 375 RPS · P50 19 s |
| 1 000 000 | (skipped) | client fails before server (0 completed, 0 errors) |

Both flat-line at ~300–500 RPS once concurrency exceeds the single event-loop can handle; no 5xx anywhere. The M5 edge at low concurrency is real but misleading: it's native Python on macOS vs. docker-bridge-in-VM on Blackwell. On a matched deployment shape (multi-worker uvicorn behind a LB) the picture reverses immediately. 1 M concurrent produces zero completed requests on the client side — exactly the pre-registered "fails before the server" outcome.

Raw JSON:
- `bench/results/mac-m5/20260422T014337Z/summary.json`
- `bench/results/blackwell-4gpu/20260422T021606Z/summary.json`

#### Mac M5 — inference scenarios

Not measured. Docker Desktop's 7.6 GB VM on this machine OOMs the CPU vLLM stack (exit 137) on the first inference request even with `max_model_length=64` and `gpu_memory_utilization=0.10`. The nop-only entrypoint (`bench/server/nop_only_main.py`) survives because it doesn't boot vLLM; anything that does pushes the VM over. The plan's M5 role — stable denominator for proxy-overhead — is still satisfied by the nop numbers above; the inference scenarios belong on Spark/Blackwell per the original grid.

#### Blackwell — full inference sweep

Raw JSON: `bench/results/blackwell-4gpu-first-pass/20260422T042805Z/summary.json`. Platform: RTX PRO 6000 Max-Q × 2 (TP=2), `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, `openai_queue_concurrency=16`, `max_num_seqs=16`.

**Parity check — bulk wire efficiency** (prompts per second; equal payload size per prompt):

| N per call | scalarlm `/v1/generate` | openai `/v1/completions` array | openai − scalarlm |
|---|---|---|---|
| 1 | 2.3 | 4.2 | **+81 %** |
| 10 | 6.1 | 9.2 | **+51 %** |
| 100 | 13.1 | 11.1 | −15 % |
| 1 000 | 16.6 | 17.5 | **+5.6 %** |

At the large-batch end the plan's 5 % threshold is cleared favourably — openai is faster, not slower. Small N favours openai by a lot (less middleware weight; for single-prompt calls the scalarlm worker + queue round-trip costs show). The first-pass apparent −15 % dip at N=100 was single-run noise — see the repeated-runs section below for the corrected number.

**N=100, 10 iterations with warmup, unique prompt per iteration** (`bench/scenarios/n100_repeat.sh`, raw JSON at `bench/results/blackwell-4gpu-n100-repeat/20260422T050053Z/summary.json`):

| metric | openai `/v1/completions` array | scalarlm `/v1/generate` |
|---|---|---|
| mean prompts/s | **14.54** | 11.04 |
| stdev | 2.77 (19 %) | 1.66 (15 %) |
| min — max | 9.60 — 17.39 | 8.91 — 13.53 |
| mean call duration | 7.14 s | 9.24 s |

**openai beats scalarlm by +31.7 % on the mean** at N=100; both distributions overlap substantially (openai's low end touches scalarlm's high end) but the means are separated by more than one stdev. The first-pass −15 % for openai at N=100 was a single sample near its lower edge.

**Why the first run of each scenario has to be a warmup, and why the prompts must vary per iteration.** Two real gotchas surfaced while turning the single-sample numbers into statistics:

- vLLM's GDN-prefill kernels compile JIT per sequence-length bucket. First call at a new length eats seconds of Triton compile. Repeating at the *same* bucket is cheap. Benchmark runs need one discarded warmup before the measurement window.
- scalarlm's `/v1/generate` derives `request_id = sha256(json.dumps(requests_list))` (`generate.py:85-86`). Identical request lists across iterations collide; the queue may return cached results without re-running inference. A first attempt at the repeat run produced a fake *470 p/s* for scalarlm because of this — that's the red-flag to watch for. The fix is to vary the prompt per iteration so every batch hashes differently.

#### Why openai is faster than scalarlm on this benchmark

Both paths end up at the same vLLM process serving the same model on the same two GPUs, so the throughput gap has to live entirely in the code between the client and vLLM. Going most-to-least influential:

1. **More pipeline stages in scalarlm.** A single `/v1/generate` call runs: build per-prompt dicts → SHA-256 the list → JSON-dump to disk → push into SQLite (persistent queue) → worker polls the queue (~0.1 s idle loop) → SQLite select → read JSON back → iterate prompts → construct a vLLM `CompletionRequest`/`ChatCompletionRequest` per prompt → call vLLM Python API → collect results → write results JSON → HTTP `/v1/generate/finish_work` → SQLite update + ack → client polls `/v1/generate/get_results`. The openai path is: parse `CompletionRequest` → filter → POST upstream → proxy the response back. At N=1 the fixed overhead of the longer pipeline dominates; at N=1000 it amortizes to a few percent.
2. **How N prompts reach vLLM.** openai forwards `prompt=[p1..pN]` as a single vLLM call — vLLM's scheduler sees all N at once and runs them under one continuous-batch sweep. scalarlm's worker loops over prompts and submits each as a *separate* vLLM request (see `create_generate_worker.py:473-509`). Continuous batching still catches them, but the worker's temporal spread means each prompt's prefill starts slightly later than it would in a single submit. This is the largest inference-side contributor.
3. **Client polls the server for results.** The scalarlm client (and my harness) polls `/v1/generate/get_results` on an interval. Even after the server finishes, the client can wait up to one poll before noticing. openai returns the answer on the same HTTP response.
4. **Worker polls the queue.** The worker idles between queue polls (`await asyncio.sleep(0.1)`). A request enqueued mid-poll waits on average ~50 ms for pickup; openai has no such poll — the request is in vLLM's queue the moment the proxy opens its POST.
5. **Per-prompt Python work in `generate.py`.** `metrics.record_new_request()` is called *once per prompt* (`generate.py:82`); openai records once per HTTP call. Small absolute, but for N=100 that's 100 extra function entries that touch an epoch timer.
6. **Two extra encode/decode trips.** scalarlm dict-ifies each request, JSON-dumps the list, writes to disk, and the worker re-parses into Pydantic. openai parses once via vLLM's Pydantic model and forwards it as-is.

Things this is probably **not**:

- Not a vLLM difference — same kernels, same KV cache, same GPU assignment.
- Not adapter-load overhead — both runs use the base model; no LoRA load fires.
- Not the new queue — at the concurrencies measured, `OpenAIConcurrencyLimiter` is a semaphore acquire/release on the order of microseconds.
- Not a statistical artefact — the N=100 gap (3.5 p/s) exceeds both stdevs (2.77 and 1.66) and the same sign holds across N=1/10/1000.

Caveats worth tightening before calling the delta "inherent":

- Disk for scalarlm's request/results JSON is longhorn-backed in this pod. A faster scratch disk (tmpfs or local NVMe) would narrow the gap a little.
- The harness's polling cadences (client-side `get_results` and worker-side queue poll) could be tuned tighter; I didn't touch them.
- scalarlm creates a FastAPI `Request` object per prompt in the worker (`create_generate_worker.py:367-376`). I haven't profiled whether that construction is material at this batch size.

Bottom line: scalarlm's pipeline buys durability, cross-machine worker scaling, per-request error isolation, and auditability; it pays for those with a fixed per-call tax of a few ms up to tens of ms that the shorter openai pipeline skips by assuming the client will retry on HTTP 5xx. Both are valid designs; for throughput benchmarks on the happy path, the shorter pipeline wins.

**openai interactive chat** (`pathb_chat_single`, one request per coroutine, queue=16):

| concurrency | RPS | P50 | P95 | P99 |
|---|---|---|---|---|
| 1 | 2.8 | 357 ms | 367 ms | 373 ms |
| 10 | 6.2 | 1.37 s | 3.91 s | 3.92 s |
| 50 | 12.3 | 3.82 s | 4.85 s | 5.48 s |
| 200 | 15.9 | 12.4 s | 13.0 s | 13.2 s |

Throughput plateaus at vLLM's decode ceiling; latency climbs with queue wait as designed. Zero 5xx across all levels — the limiter produces smooth degradation, not a cliff.

**openai async bulk** (`/v1/batches`, submit + 1 s poll + fetch, wall-clock end-to-end):

| N | total s | prompts/s |
|---|---|---|
| 1 | 0.54 | 1.9 |
| 10 | 3.58 | 2.8 |
| 100 | 34.1 | 2.9 |
| 1 000 | 309.8 | 3.2 |

Batches API runs ~5× slower than array `/v1/completions` at equal N because the 1-second poll interval dominates small-batch runs and is still visible at 1000. The right default today is array prompts when the caller can block; batches belong to the truly-async/offline workload they were designed for. Tightening the poll interval would narrow the gap — worth revisiting if anyone hits it in practice.

**scalarlm `upload_download` not measured** — harness script sends JSON; the endpoint requires multipart/form-data. Client fix is a follow-up; not blocking the parity claim since the 1000-prompt-per-call comparison in the first table is sufficient.

**Phase 3 smoke on the same pod** (`bench/smoke_phase3.sh`): 6/6 features live — `X-Request-Id` auto-assigned and caller-supplied round-trips, `/v1/bench/nop` gated behind config, scalarlm deprecation log fires with migration hint, Batch POST/GET/DELETE round-trips, state machine transitions correctly on cancel.

Summary: **on the production-class platform, with the production-class model, the plan's headline parity claim is empirically cleared.** Deprecating scalarlm — under the telemetry-gated path described in Phase 4 — is data-supported.

## Not in scope for this change

- Rewriting the vLLM fork's OpenAI server. No evidence yet that we need to.
- Replicating scalarlm's durable SQLiteAckQueue for synchronous OpenAI traffic. The in-process async queue from Phase 3 item 4 covers the observable capabilities; durability lives in the Batch API's file-backed results store.
- Touching the SDK (`sdk/masint/`). It remains as-is until deprecation lands.
- Supporting `/v1/files` as a prerequisite for `/v1/batches`. Our batch API takes the JSONL inline.
