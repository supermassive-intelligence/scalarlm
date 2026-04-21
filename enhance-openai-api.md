# Migrating from the ScalarLM-native path to a single OpenAI-compatible path

## Status

Analysis and plan only. No implementation yet.

## Why this work exists

The ScalarLM-native path (`POST /v1/generate` via the `scalarlm.SupermassiveIntelligence` SDK) does not expose tool calling, and that gap drives callers away — `sql-gen` already runs exclusively on the OpenAI path for this reason. The long-term direction is **a single OpenAI-compatible path**. This document proposes how to close the remaining feature gaps on the OpenAI path so Path A can be deprecated with no loss of capability.

## Background: the two paths a caller can use today

Two inference entry points exist in ScalarLM. Callers such as `sql-gen` pick between them via a flag (`use_scalarlm`). Both paths eventually talk to the same vLLM process (built from `supermassive-intelligence/vllm-fork` — see `Dockerfile:259`), but they reach it very differently:

| Path | Client surface | Server entry | Transport to vLLM |
|---|---|---|---|
| **A — ScalarLM-native** | `scalarlm.SupermassiveIntelligence().async_api.generate([prompts], model_name, max_tokens)` | `POST /v1/generate` → SQLite work queue → `get_work` worker → vLLM Python API | In-process Python (`create_chat_completion`, `create_completion`, `load_lora_adapter`) |
| **B — OpenAI-compatible** | `openai.AsyncOpenAI(base_url, api_key).chat.completions.create(...)` | `POST /v1/completions` / `POST /v1/chat/completions` | HTTP proxy straight to vLLM's OpenAI-compatible server |

Path A is a queue-based architecture with worker-side orchestration; Path B is a thin streaming proxy. `sql-gen` uses Path B in practice (`src/llm.py:70-74`) because it needs tool calling, which the ScalarLM SDK does not expose.

## Functional inventory

### What Path A does server-side (`infra/cray_infra/api/fastapi/generate/generate.py`, `infra/cray_infra/one_server/create_generate_worker.py`)

1. **Dynamic adapter/model routing.** `model_manager.find_model(model)` (`generate.py:56`) resolves the request's `model` field against LoRA adapters discovered under `training_job_directory`. The worker lazily loads the adapter into vLLM via `POST /v1/load_lora_adapter` the first time it's seen (`create_generate_worker.py:215-251`) and tracks `loaded_adaptor_count` so subsequent requests don't reload.
2. **SQLite-backed ack queue.** Requests get a SHA-256 `request_id` (`generate.py:85-86`), are serialized to disk, and pushed into a `SQLiteAckQueue` (`infra/cray_infra/api/work_queue/inference_work_queue.py`). The queue is persistent, FIFO, and auto-recovers in-flight requests on restart.
3. **Worker-side batching.** Workers pull up to `batch_size` items from `/v1/generate/get_work`, where `batch_size` is derived from KV-cache headroom (`create_generate_worker.py:176`). Batching happens *inside* vLLM from a contiguous set of queued requests rather than being expressed in the client API.
4. **Multi-prompt fan-in on the wire.** The SDK takes `[prompt1, prompt2, ...]`. For fewer than 128 prompts it POSTs them all to `/v1/generate`; for more it uploads a file and polls for a download (`sdk/masint/engines/async_cray.py:29-48`).
5. **Per-request error isolation.** Each element of the batch carries its own `error` field in the final `GenerateResponse`; partial success is supported.
6. **Polling-based result retrieval.** The SDK polls `/v1/generate/get_results` with the list of `request_id`s until all are filled (`async_cray.py:155-167`). Results are stored in an in-memory dict backed by files on disk and survive restarts long enough to be retrieved.
7. **FLOP metrics.** The worker computes `token_count × compute_flop_count(model_config)` per request and reports it on the shared metrics counter (`create_generate_worker.py:382, 502`).
8. **Queue observability.** Prometheus/OTel counters expose `queue_depth` and `queue_wait_time_seconds` by model (`inference_work_queue.py:14-19`).

### What Path B does today (`infra/cray_infra/api/fastapi/routers/openai_v1_router.py`)

1. **Parameter whitelist and passthrough.** The two handlers filter incoming `CompletionRequest` / `ChatCompletionRequest` fields down to a known set (`_COMPLETION_ALLOWED_KEYS`, `_CHAT_ALLOWED_KEYS` at lines 43-75). `tools`, `tool_choice`, `temperature`, `response_format`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, and streaming are all included.
2. **Usage sniffing for metrics.** `_ensure_usage_reported` forces `stream_options.include_usage=True` for streaming calls so vLLM emits a terminal `usage` event (`openai_v1_router.py:130-141`). `_wrap_with_metrics` scans the tail of the SSE stream (bounded to 64 KB) to pull `usage.total_tokens` and feed it into the shared metrics counter (`openai_v1_router.py:172-204`).
3. **Direct proxy to vLLM's OpenAI server.** No queue, no worker, no on-disk request store — the proxy opens a streaming HTTP call to `vllm_api_url + "/v1/completions"` (or chat variant) and forwards bytes through.

### Gap summary — what A has that B lacks

| Capability | Path A | Path B | Disposition for this work |
|---|---|---|---|
| Dynamic LoRA routing by `model` name | Yes | No (vLLM-default only) | **In scope — proxy-side load.** Critical for finetune routing. |
| FLOP metrics | Yes | `flop_count=None` | **In scope — observability parity.** Dashboards currently under-report Path B. |
| Durable request IDs | SHA-256 per request, persisted | None | **In scope — audit/correlation.** UUID per call surfaced as `X-Request-Id`. |
| Request queue with depth + wait-time metrics | Yes (SQLiteAckQueue) | No | **In scope — see "Queue on Path B" below.** Backpressure, fair ordering, observability. |
| Multi-prompt batching on the wire | Yes (`prompts=[...]`) | Partial — already works on `/v1/completions` via the OpenAI-spec `prompt` array; not available for chat | **In scope — see "Multi-prompt on the OpenAI surface" below.** Test/document what already works; use OpenAI Batch API for chat. |
| Per-request error isolation within a batch | Yes | Standard OpenAI shape (one `choices[i].finish_reason` per prompt index) | In scope implicitly — covered by multi-prompt testing. |
| Polling-based result retrieval | Yes | Synchronous / SSE | **Not in scope to replicate.** The OpenAI Batch API pattern (submit → poll `/v1/batches/{id}`) subsumes this, and the sync/SSE path serves the common case. |
| Streaming | No | Yes (SSE) | Path B is already ahead. |

Net: **five gaps are in scope** — LoRA routing, FLOP metrics, request IDs, queue + observability, multi-prompt (via the completions `prompt` array today and the Batch API for chat). No gap remains that would require keeping Path A around.

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

4. **Queue on Path B — proxy-layer only, no fork changes.** The goal is the capabilities Path A's queue provides (bounded concurrency, fair ordering, `queue_depth` / `queue_wait_time_seconds` metrics, graceful 503 under overload) without disturbing the synchronous/SSE contract that OpenAI clients expect. The implementation is a FastAPI middleware that wraps both `create_completions` and `create_chat_completions`:
   - A bounded `asyncio.Queue` of pending call objects (each carries the request, the caller's `asyncio.Future` or the async response-chunk sink).
   - A small pool of drainer tasks (size = desired vLLM concurrency) that pop items and do the real vLLM call, writing chunks back to the sink for streaming or the result to the future for non-streaming.
   - OpenTelemetry counters: `openai_queue_depth`, `openai_queue_wait_seconds` (matching naming with Path A's `queue_depth` / `queue_wait_time_seconds` so dashboards merge cleanly).
   - When the queue is full beyond a threshold, return `503 Service Unavailable` with `Retry-After` — standard OpenAI client retry territory.
   - We **do not** replicate Path A's SQLiteAckQueue durability. Path A's durable queue exists to survive crashes in a polling world; with synchronous HTTP, a crashed request just returns an error to the client, who retries. If durability becomes a product requirement (compliance, async batch), it lands as part of the Batch API below, not here.

5. **Multi-prompt on the OpenAI surface — two distinct mechanisms, both proxy-layer:**
   - **`/v1/completions` with array `prompt`.** The OpenAI-spec `prompt` field accepts `str | list[str] | list[int] | list[list[int]]`. The vLLM fork already implements this (`vllm/entrypoints/openai/completion/protocol.py:46-52`), and our proxy's existing whitelist passes `prompt` through unchanged. Work: **an integration test confirming it round-trips**, plus documentation. Likely zero code changes.
   - **OpenAI Batch API (`/v1/batches`) for chat completions.** Chat completions has no in-request batching idiom; OpenAI's canonical answer for bulk chat is the [Batch API](https://platform.openai.com/docs/api-reference/batches) — submit a JSONL of requests, poll for a completed JSONL of responses. This is an async, queue-based interface by design, so it dovetails with the queue from item (4). Minimum viable endpoints: `POST /v1/batches` (enqueue), `GET /v1/batches/{id}` (status), `GET /v1/batches/{id}/output_file_content` (results). The implementation sits on top of the same queue infrastructure, with a persistent results store for later retrieval.
   - Note: Path A's "upload for >128 prompts" flow (`/v1/generate/upload`, `/v1/generate/download`) is structurally a Batch API — the new endpoint is the OpenAI-shaped version of the same thing.

None of (1)-(5) requires modifying the vllm-fork. If during implementation we discover a genuine vLLM gap (for example, if `/v1/load_lora_adapter`'s existing error shape doesn't expose enough detail), we raise it separately and propose a small upstream-friendly patch — not expected based on the fork review above.

## Test-driven implementation plan

The user's preferred sequence is: write tests for Path A behaviour first, then write the same tests against Path B (they fail), then implement until they pass, then deprecate Path A.

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

### Phase 1 — Capture Path A behaviour as tests

All pass against current `main`:

1. `test_path_a_loads_adapter_on_first_request` — call `/v1/generate` with `model=<adapter-name>`; assert the worker issued `POST /v1/load_lora_adapter` exactly once even across two back-to-back requests.
2. `test_path_a_reports_flop_count` — call `/v1/generate`; observe the metrics counter; assert `flop_count > 0`.
3. `test_path_a_returns_request_id` — assert `GenerateResponse.results[i].request_id` has the `{sha256}_{index:09d}` shape.
4. `test_path_a_queues_under_pressure` — submit N requests faster than the worker can drain; assert `queue_depth` metric crosses a threshold and `queue_wait_time_seconds` is non-zero.
5. `test_path_a_accepts_multi_prompt` — submit `prompts=[p1, p2, ..., p5]`; assert the response contains five distinct `Result` entries with correctly indexed `request_id`s and that per-item errors are isolated.
6. `test_path_a_upload_download_for_large_batch` — submit `prompts=[...]` with length > 128; assert the upload/download (batch) flow is exercised and results round-trip correctly.

Each test documents what "Path A parity" means for that capability, anchoring Phase 2.

### Phase 2 — Same tests against Path B (expected to fail)

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
   - Emit `openai_queue_depth` and `openai_queue_wait_seconds` via the same OpenTelemetry meter as Path A.
   - 503 with `Retry-After` when depth exceeds a config threshold.
   - Config keys: `openai_queue_max_depth`, `openai_queue_concurrency`. Defaults chosen conservative; plumb through `default_config.py`.
5. **Multi-prompt for `/v1/completions`** (~0 lines code, focus on tests). Confirm the array-prompt path end-to-end; document it in `docs/api-documentation.md`. Only code change expected is a log line that records the prompt-count for a batched call so it's visible in the request log.
6. **Batch API scaffold** (~200 lines). Introduce `POST /v1/batches`, `GET /v1/batches/{id}`, `GET /v1/batches/{id}/output_file_content`, `DELETE /v1/batches/{id}` at OpenAI-spec shapes. Under the hood: persist the incoming JSONL to disk, enqueue each line via the queue from step 4, aggregate outputs into the result JSONL as items complete, record `batch.status` transitions (`validating` → `in_progress` → `completed` / `failed`). Reuse Path A's on-disk results layout where it fits; the goal is that Batch API inherits durability "for free" from a simple file-backed results store.

At the end of Phase 3, Phase-2 tests pass. Phase-1 tests still pass (we don't touch Path A).

### Phase 4 — Mark Path A for deprecation (the point of this work)

- Emit a structured deprecation log when `POST /v1/generate`, `/v1/generate/get_results`, `/v1/generate/upload`, `/v1/generate/download` are hit, including the caller's `User-Agent` so we can see who's still on Path A.
- Update `docs/api-documentation.md` and the SDK README to point new integrations at:
  - `/v1/chat/completions` for chat.
  - `/v1/completions` with an array `prompt` for the small-batch-of-prompts case.
  - `/v1/batches` for the large-batch-of-prompts case (was Path A's `upload`/`download`).
- Publish a one-line migration for each common Path A usage (single-prompt generate, multi-prompt generate, uploaded batch), so callers can migrate without reading the spec.
- Leave `sdk/masint/engines/async_cray.py` in place for now. A follow-up can either reimplement `generate()` as a thin shim over the OpenAI client or remove it outright once no internal caller uses it.

Do **not** delete Path A in this work. Deprecation first, removal later, with a named grace period tied to the usage telemetry above.

### Phase 5 (future) — LoRAResolver plugin

Once Option A (proxy-side load) is stable, consider packaging the ScalarLM adapter discovery as a `LoRAResolver` plugin (see "vLLM-fork capabilities" above). This collapses the proxy to a pure passthrough and moves ScalarLM-specific resolution logic into a well-defined vLLM extension point. Not required for Path A deprecation — just cleaner long-term.

## Risks and open questions

- **Unknown `model` values.** If a caller passes a name that neither resolves to an adapter nor to the base model, the proxy should not pre-call `/v1/load_lora_adapter` with a bogus path — let vLLM error. The model manager's `find_model` already returns `None` for unknowns; use that signal to skip the load step and forward directly (vLLM returns 400/404 as before).
- **Concurrency on adapter loads.** Resolved by the fork — vLLM's `lora_resolver_lock[lora_name]` serializes concurrent loads of the same name (`vllm/entrypoints/openai/models/serving.py:278`). The proxy's local cache uses an `asyncio.Event`-per-name pattern to avoid two HTTP calls during the first load window, but correctness does not depend on it.
- **Tokenformer vs standard LoRA.** The fork's `TokenformerModelManager` replaces the stock LoRA manager and handles `.pt` checkpoints. From the proxy's perspective there's no difference: both are loaded via `/v1/load_lora_adapter` with a path. This is already how Path A's worker operates today.
- **Queue interaction with SSE streaming.** A queued streaming request has to hand bytes back to an already-open HTTP response. The pattern we'll use: the handler creates an `asyncio.Queue` of chunks, enqueues the work pointing at that queue, returns a `StreamingResponse` that drains the chunk-queue, and the drainer task writes chunks into it as vLLM produces them. Test with a long SSE stream and a pile of queued callers to confirm backpressure propagates end-to-end (vLLM slows → drainer blocks → queue depth rises → handler returns 503 at threshold).
- **Batch API semantics.** Our `/v1/batches` need only be a subset of OpenAI's — support JSONL request bodies, `status` transitions, and result fetching. We do **not** need `input_file` upload via `/v1/files` as a separate step; inline JSONL is simpler and sufficient for now. Document the subset explicitly so callers know the edges.
- **FLOP count for tool-calling responses.** `usage.total_tokens` includes tool-call tokens, so multiplying by `compute_flop_count(model_config)` gives a valid estimate. Covered by a test.
- **Request-ID semantics.** UUID v4 gives correlation, not content-addressable idempotency. Right scope for Phase 3. If a future consumer needs idempotency (e.g. dedupe across retries), add it then.
- **Request-log persistence for compliance.** Path A writes every request JSON to disk. The Batch API's on-disk JSONL naturally covers the batch case; for ad-hoc `/v1/chat/completions` traffic, we'd need a separate audit log if that's a compliance requirement rather than debugging. Flag for product before implementation.
- **Config defaults for the queue.** Start conservative — e.g. `openai_queue_concurrency=16`, `openai_queue_max_depth=256` — tune once we have live telemetry. Bad defaults here are the thing most likely to surprise in production.

## FAQ

### "Path A can do the equivalent of 1 million requests/sec via wire batching. Can the OpenAI surface keep up?"

It's not a hard limit, but the numbers measure different things and the mechanics differ. Short version:

- Path A's "1 M" is a per-prompt number. One HTTP call carries up to 128 prompts on `/v1/generate` (or a JSONL file via `/v1/generate/upload` for more). All the per-request overhead — TCP+TLS handshake, parse, auth, middleware, model resolution — is amortized across every prompt in the batch.
- Path B's "1000 req/sec" is per HTTP call. Each request pays that overhead once.

The efficiency gap only exists for *bulk* workloads, and the plan preserves Path A's wire shape via two OpenAI-spec mechanisms:

| Workload | Path A today | Unified path after Phase 3 |
|---|---|---|
| Bulk completions | 1 call × N prompts on `/v1/generate` | 1 call × N prompts on `/v1/completions` with array `prompt` — **identical efficiency** |
| Bulk chat (offline) | 1 call × N prompts via upload/download | `/v1/batches` (OpenAI Batch API) — **identical efficiency**, canonical shape |
| Interactive chat (one-at-a-time, streaming) | Not supported — Path A doesn't stream | Fan-out on `/v1/chat/completions` — pays per-request overhead, but streaming works |

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

## Not in scope for this change

- Rewriting the vLLM fork's OpenAI server. No evidence yet that we need to.
- Replicating Path A's durable SQLiteAckQueue for synchronous OpenAI traffic. The in-process async queue from Phase 3 item 4 covers the observable capabilities; durability lives in the Batch API's file-backed results store.
- Touching the SDK (`sdk/masint/`). It remains as-is until deprecation lands.
- Supporting `/v1/files` as a prerequisite for `/v1/batches`. Our batch API takes the JSONL inline.
