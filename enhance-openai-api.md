# Migrating from the ScalarLM-native path to a single OpenAI-compatible path

## Status

Phases 0–7 implemented; measurement summary:

- **Phase 6**: shipped. In-process Python-API call replaces the localhost HTTP hop. Mixed throughput results — see Phase 6 measurement subsection.
- **Phase 6.5**: shipped (`scripts/vllm_patches/apply_patches.py` + Dockerfile wiring). **Failed its acceptance gate at N=100 distinct prompts on Blackwell** — paired 30-run A/B is null (8.32 patched vs 8.56 unpatched). Two follow-up measurements (Spark N=1 000, Blackwell N=1 000) determine whether the patch stays or is reverted.
- **Phase 7**: shipped (`asyncio.gather` + `Semaphore(batch_runner_concurrency)` in BatchRunner; 14/14 unit tests). **Cleared its acceptance gate** — Batch API at N=1 000 jumped from 3.2 → 13.14 p/s, matching array `/v1/completions` and reaching 79 % of scalarlm.
- **Phase 8**: not yet started (scatter-gather A/B). Same N=1 000 ceiling now applies to both array-completions and Batch API; if Phase 8a clears its 10 % win threshold, both lift together.
- **Phase 9**: deferred per analysis above.

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

### Phase 6 — collapse the HTTP hop between the openai proxy and vLLM

**Motivation.** The bulk-parity sweep at N=1 000 (see "First-pass results" below) shows scalarlm beating openai by 27 % — the gap is almost entirely the HTTP hop from the openai proxy (FastAPI on 8000) to vLLM's own OpenAI server (FastAPI on 8001). scalarlm's worker runs in the same process as vLLM and calls the Python API directly; openai serialises a 1000-element `prompt` array to TCP, vLLM deserialises, generates, serialises a 1000-element `choices` response, and the proxy streams every byte back. Eliminating this hop is the single biggest lever available.

**Approach.** Call vLLM's `OpenAIServingCompletion.create_completion(...)` / `OpenAIServingChat.create_chat_completion(...)` directly from the openai proxy handler, using the same engine client the scalarlm worker already uses. vLLM's internal classes accept a parsed `CompletionRequest`/`ChatCompletionRequest` plus the FastAPI `raw_request`, and return either a `JSONResponse` (non-streaming) or a `StreamingResponse` (streaming) — the exact shape our proxy already produces.

Alternative shapes considered:
- *Include vLLM's router in the scalarlm app.* Works but bypasses our Phase 3d queue / request-id middleware unless we reorder carefully. More plumbing.
- *`httpx.ASGITransport` to call vLLM's app in-process.* Skips TCP but still pays JSON encode/decode twice. Marginally faster than HTTP; doesn't address the biggest cost.

Picking the direct-Python-API approach for the clearest throughput win with the smallest disruption.

**Implementation sketch (≈80 lines in `openai_v1_router.py`):**

1. At app startup, stash a reference to vLLM's already-initialised serving objects on scalarlm's `app.state`. `create_vllm.py` already calls `init_app_state(engine_client, vllm_app.state, args, supported_tasks)` — the completions serving lives at `vllm_app.state.openai_serving_completion`, chat at `vllm_app.state.openai_serving_chat`. Thread those through so both FastAPI apps in this process share them.
2. Replace `_proxy_streaming` with `_call_inprocess`:
   ```python
   async def _call_inprocess(*, request, raw_request, endpoint, base_model_name, queue_slot):
       servings = raw_request.app.state.vllm_servings
       if endpoint == "completions":
           resp = await servings.completion.create_completion(request, raw_request)
       else:
           resp = await servings.chat.create_chat_completion(request, raw_request)
       if isinstance(resp, StreamingResponse):
           resp.body_iterator = _wrap_with_metrics(resp.body_iterator,
                                                  base_model_name=base_model_name,
                                                  queue_slot=queue_slot)
           return resp
       # Non-streaming: run metrics synchronously, release slot, return.
       _record_metrics_from_body(resp.body, base_model_name)
       if queue_slot: await queue_slot.release()
       return resp
   ```
3. Keep request-id middleware, `OpenAIConcurrencyLimiter`, LoRA pre-load, and the multi-prompt log line exactly where they are. Only the upstream-call line changes.
4. Drop `vllm_api_url` from `default_config.py` (it becomes unused). The separate vLLM uvicorn on 8001 no longer needs to be started — one fewer server process.

**Tests** (`test/unit/`):

- `test_inprocess_completion_calls_serving`: inject a mock `OpenAIServingCompletion`, assert our handler passes the filtered request + `raw_request` through.
- `test_inprocess_streaming_body_wrapped_with_metrics`: mock returns a `StreamingResponse` with a fake SSE body; assert our wrapper runs and records `token_count`.
- `test_inprocess_queue_slot_released_on_normal_completion` and `_on_error`: slot is released exactly once in every path.
- `test_inprocess_non_streaming_records_metrics_synchronously`: mock returns `JSONResponse`; assert metrics fired before the response left.

Integration: the existing `bench/smoke_phase3.sh` should continue to pass 6/6 with zero changes.

**Measurement plan.**

1. Keep the current HTTP-hop image deployed on Blackwell until Phase 6 is built; those bench numbers are the baseline.
2. Build and push the Phase 6 image under a distinct tag (`kapu/scalarlm-nvidia-12.0:phase6-inprocess`) so a simple pod-spec swap toggles between them.
3. Re-run `bench/scenarios/bulk_repeat.sh` at N=1/10/100/1000 on both Blackwell and Spark.
4. Report a 3-column table per platform: scalarlm, openai (HTTP hop), openai (in-process). Compute the delta between openai-HTTP and openai-in-process; that's the transport-overhead component.
5. **Pre-registered expectations**:
   - N=1: openai (in-process) within ±5 % of openai (HTTP). Small-N gap was pipeline cost, not transport.
   - N=10, N=100: openai (in-process) 5–15 % faster than openai (HTTP). Transport cost becomes visible.
   - N=1000: openai (in-process) ≥ 25 % faster than openai (HTTP) — closing or flipping scalarlm's current lead. The clean signal that transport was the bottleneck.
   - scalarlm stays the same; this phase doesn't touch its code path.

**Risks / trade-offs.**

- **vLLM Python API is not versioned the way its HTTP API is.** A vLLM-fork update that changes `OpenAIServingCompletion`'s signature breaks us. Mitigated by pinning the fork tightly and catching the regression in the unit tests above.
- **No process boundary between proxy and engine.** Today if vLLM crashes, the proxy survives and returns a 5xx. After Phase 6 a vLLM segfault takes the whole scalarlm server down too. Since ScalarLM already runs vLLM in-process for the scalarlm path (the worker + engine share a process), the blast radius doesn't actually increase. But noting it.
- **Streaming-response wrapping.** `_wrap_with_metrics` was designed to consume an aiohttp `iter_any()` stream. vLLM's `StreamingResponse.body_iterator` is an async iterator of bytes — same shape. Needs one round of validation that backpressure still propagates end-to-end (vLLM slows → `body_iterator` yields slower → client read blocks).
- **Config cleanup.** `vllm_api_url` becomes dead code; grep and remove. Downstream callers that read `get_config()["vllm_api_url"]` (there are a few — health check, model list) either need to switch to in-process calls or keep talking to vLLM's server on 8001 during a transition period. Cleanest: migrate all of them, then stop starting vLLM's own server.

**Scope boundary.** Phase 6 is strictly about the openai path's transport to vLLM. It does not touch scalarlm's queue, the Batch API, or any of the Phase 3 features. The deprecation argument in Phase 4 stays the same — this just closes the large-N throughput gap that was the one empirical reason not to deprecate scalarlm today.

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

#### Blackwell — bulk parity sweep (10 measurement runs per cell)

Platform: RTX PRO 6000 Max-Q × 2 (TP=2), `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, `openai_queue_concurrency=16`, `max_num_seqs=16`. Each cell is 10 measurement iterations + 1 discarded warmup, with a unique prompt per iteration. Runner: `bench/scenarios/bulk_repeat.sh`. Raw JSON under `bench/results/blackwell-4gpu-n{1,10,100,1000}-repeat/`.

**Why warmup + per-iteration unique prompts are mandatory** (both learned the hard way from the first pass):

- vLLM's GDN-prefill kernels compile JIT per sequence-length bucket. First call at a new length eats seconds of Triton compile. Repeating the same bucket is cheap. One discarded warmup per scenario removes this.
- scalarlm's `/v1/generate` derives `request_id = sha256(json.dumps(requests_list))` (`generate.py:85-86`). Identical request lists across iterations collide; the queue returns cached results without re-running inference. A first attempt at the repeat run produced a fake **470 p/s** for scalarlm because of this. Varying the prompt per iteration defeats the hash collision.

**Consolidated parity table — Blackwell (2 × RTX PRO 6000 Max-Q, `Qwen/Qwen3-Next-80B-A3B-FP8`, TP=2):**

| N per call | scalarlm `/v1/generate` | openai `/v1/completions` array | openai − scalarlm |
|---|---|---|---|
| 1 | 2.14 ± 0.23 | **2.94 ± 0.08** | **+37.6 %** |
| 10 | 6.62 ± 0.89 | **10.05 ± 0.87** | **+51.8 %** |
| 100 | 11.04 ± 1.66 | **14.54 ± 2.77** | **+31.7 %** |
| 1 000 | **16.57 ± 0.03** | 12.06 ± 0.86 | **−27.2 %** |

On Blackwell the picture flips between N=100 and N=1 000. At N≤100 openai wins by 30–50 % — scalarlm's fixed per-call pipeline tax dominates. At N=1 000 the gap reverses: scalarlm wins by 27 %, with variance collapsing to 0.03 p/s while openai's widens to 0.86 p/s.

**Consolidated parity table — DGX Spark (1 × GB10, `nvidia/Qwen3-32B-NVFP4`, TP=1, co-resident sql-gen vLLM using ~50 GB GPU):**

| N per call | scalarlm `/v1/generate` | openai `/v1/completions` array | openai − scalarlm |
|---|---|---|---|
| 1 | **0.80 ± 0.27** | 0.60 ± 0.00 | **−24.4 %** |
| 10 | **3.54 ± 1.22** | 2.87 ± 0.05 | **−18.9 %** |
| 100 | **5.67 ± 1.94** | 4.25 ± 0.11 | **−24.9 %** |
| 1 000 | **16.60 ± 0.09** | 4.46 ± 0.12 | **−73.2 %** |

Spark shape is very different from Blackwell. openai loses at **every** N, from −19 % to −73 %. Two contributors:

- One-GPU GB10 is CPU-bound on the proxy side; the HTTP encode/decode of even a 1-element prompt list shows up relative to decode time.
- The running sql-gen vLLM pod is contending for the same GB10, which inflates both paths' numbers and adds the very high scalarlm stdev at small N (up to 34 % of mean). At N=1 000 the stdev collapses to 0.09 — enough work per call to average out scheduler jitter and let vLLM's prefix cache hit cleanly.

**The plan's 5 % parity threshold is not met on either platform** under the current openai implementation. Phase 6 (collapse the HTTP hop — see Architectural strategy / Phase 6) is the proposed fix; post-implementation measurement and a py-spy profile are in the "Phase 6 measurement and follow-up" section below, and the amended plan (Phases 7–9) incorporating an independent review by Gemini is at the end of the doc.

#### Why the curve has the shape it does

Both paths hit the same vLLM process on the same two GPUs — the throughput gap lives entirely in the code between the client and vLLM. There are two opposing effects that trade places around N≈100.

**Small N (openai wins):** scalarlm pays a fixed per-call pipeline tax that openai skips. One `/v1/generate` call runs: build per-prompt dicts → SHA-256 the list → JSON-dump to disk → push into SQLite (persistent queue) → worker polls the queue (~0.1 s idle loop) → SQLite select → read JSON back → iterate prompts → construct a vLLM `CompletionRequest` per prompt → call vLLM Python API → collect results → write results JSON → HTTP `/v1/generate/finish_work` → SQLite update + ack → client polls `/v1/generate/get_results`. The openai path is: parse `CompletionRequest` → filter → POST upstream → proxy the response back. At N=1 that's a few tens of ms of fixed scalarlm overhead per call that openai skips — a 37 % throughput penalty when the inference itself is a few hundred ms.

Additional small-N contributors: client-side polling (scalarlm's `get_results` loop rounds wake-up to the poll interval; openai replies inline), worker-side 100 ms idle poll on the SQLite queue, per-prompt `metrics.record_new_request()` calls (`generate.py:82`), and two extra encode/decode trips through disk-backed JSON.

**Large N (scalarlm wins):** the fixed pipeline cost is fully amortized, and a new factor starts to dominate — the openai proxy forwards a big request/response over HTTP to vLLM, while scalarlm's worker calls vLLM's Python API in-process:

- openai at N=1 000 serializes a 1000-element JSON `prompt` array to TCP (~100 KB body), vLLM deserializes it, generates, serializes a 1000-element `choices` response (OpenAI response objects carry `index`, `finish_reason`, `logprobs`, `text` per item — hundreds of KB), and the proxy streams every byte back to the client while scanning the tail for `usage.total_tokens`.
- scalarlm's worker calls `create_completion(…)` directly on the in-process engine client (`create_generate_worker.py:473-509`) — no HTTP hop, no extra encode/decode passes, no tail-scan for usage.

The crossover point is where that transport overhead on openai exceeds the pipeline overhead on scalarlm. On this hardware that's somewhere between N=100 and N=1 000; the exact location will vary with the size of the response payload (longer completions push the crossover higher).

The scalarlm stdev collapses at N=1 000 (0.03 p/s — essentially deterministic) while openai's is 0.86 p/s. Two reasons contribute: HTTP timing is variable (TCP buffer scheduling, JSON parser pipelining), and vLLM's prefix cache hits on identical within-batch prompts make scalarlm's run-to-run behaviour highly repeatable once the fixed costs are amortized.

**Caveats worth tightening before calling the large-N delta "inherent":**

- This benchmark sends *identical* prompts within each batch (vLLM's prefix cache hits on all but the first). For a batch of *distinct* prompts (the more realistic offline-bulk shape) both paths would do more real prefill work and the 27 % gap may shrink, stay, or widen — not measured yet.
- scalarlm's disk-for-request-JSON here is longhorn-backed; a faster scratch tier (tmpfs, local NVMe) would shrink the small-N gap slightly.
- openai proxy's HTTP-to-vLLM hop is the same loopback interface as scalarlm's in-process call — this isn't network-bound. It's the JSON parse/build and kernel-mode TCP overhead that shows.
- Polling cadences (client-side `get_results`, worker-side queue poll) could be tuned tighter; I didn't touch them.

**What this does *not* explain away:**

- Not a vLLM difference — same kernels, same KV cache, same GPU assignment, same `max_num_seqs`.
- Not adapter-load overhead — both runs use the base model; no LoRA load fires.
- Not the new queue — at the concurrencies measured, `OpenAIConcurrencyLimiter` is a semaphore acquire/release on the order of microseconds.
- Not a statistical artefact — differences at every N are larger than the combined stdevs of the two distributions.

**What it means for the deprecation story.** The headline "5 % parity cleared at N=1 000" is retracted: scalarlm is meaningfully faster at that size under the benchmark's conditions. That does *not* by itself block deprecation, but it does change the argument:

- Openai wins small-to-mid N (the common synchronous-request case).
- Scalarlm wins large N when the batch is homogeneous.
- The large-N shape scalarlm is winning is exactly what `/v1/batches` is supposed to cover on the openai side. The current `/v1/batches` implementation's 1-second poll cadence dominates small/mid batches (seen in the table further down) and hasn't been tuned for the truly-bulk case. Optimising the Batch API (tighter poll, drainer pool sized to vLLM's real concurrency) is the natural next step before the parity claim can be re-asserted at N=1 000.

An alternative — and probably the right long-term fix — is to collapse the HTTP hop from the openai proxy to vLLM entirely, invoking vLLM's Python API in-process the same way scalarlm's worker does. That removes the biggest contributor to the large-N gap. Not in scope for this plan, but worth a mention as a "Phase 6" candidate.

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

Summary (revised from the first-pass claim):

- openai wins the **small-to-mid N bulk case** (N=1/10/100) by 32–52 % in mean prompts/sec.
- scalarlm wins the **large-N bulk case** (N=1 000) by 27 %, driven mostly by scalarlm's in-process vLLM call vs openai's HTTP proxy hop. The plan's 5 % parity threshold is *not* met at N=1 000 under the current openai implementation.
- openai is the only option for **interactive streaming chat** (scalarlm has no streaming surface at all).
- Phase 3 features (request-ids, FLOP metrics, queue, LoRA routing, Batch API) are all live and correct; that part of the plan is data-supported.

Deprecation of scalarlm under the telemetry-gated path in Phase 4 is still on the table, but the parity story at N=1 000 needs either (a) a tuned `/v1/batches` (tighter poll cadence, drainer-pool-sized-to-vLLM-concurrency), or (b) collapsing the proxy's HTTP hop to vLLM into an in-process Python-API call like scalarlm's worker does. Either closes the large-N gap; both are out of scope for this iteration but noted as a Phase 6 candidate.

## Not in scope for this change

- Rewriting the vLLM fork's OpenAI server. No evidence yet that we need to.
- Replicating scalarlm's durable SQLiteAckQueue for synchronous OpenAI traffic. The in-process async queue from Phase 3 item 4 covers the observable capabilities; durability lives in the Batch API's file-backed results store.
- Touching the SDK (`sdk/masint/`). It remains as-is until deprecation lands.
- Supporting `/v1/files` as a prerequisite for `/v1/batches`. Our batch API takes the JSONL inline.

## Phase 6 measurement and follow-up

Phase 6 (the in-process `OpenAIServingCompletion.create_completion(...)` call) was implemented under `openai_inprocess_enabled=True` (on by default) and re-benchmarked on both platforms. A py-spy profile at N=100 was also captured on Blackwell to explain the shape of the results.

### Blackwell — Phase 6 bulk parity sweep (10 runs per cell)

Same hardware and flags as the pre-Phase-6 table above (`Qwen/Qwen3-Next-80B-A3B-FP8`, TP=2, `max_num_seqs=16`, `openai_queue_concurrency=16`).

| N     | openai pre-P6      | openai P6             | Δ openai (P6 vs pre)    | scalarlm P6         | openai P6 vs scalarlm |
|-------|--------------------|-----------------------|-------------------------|---------------------|-----------------------|
| 1     | 2.94 ± 0.08        | 2.63 ± 1.02           | **−10.5 %**             | 1.91 ± 0.14         | +37.7 %               |
| 10    | 10.05 ± 0.87       | 7.28 ± 1.93           | **−27.6 %**             | 7.29 ± 1.47         | −0.1 %                |
| 100   | 14.54 ± 2.77       | 10.64 ± 0.88          | **−26.8 %**             | 14.44 ± 0.53        | −26.3 %               |
| 1 000 | 12.06 ± 0.86       | 12.92 ± 0.80          | +7.1 %                  | 16.56 ± 0.03        | **−22.0 %**           |

Phase 6 did what the expectation table predicted at N=1 000 — a small positive delta (+7.1 %) — but **regressed by 10–28 % at N=1/10/100**. The Phase-6 openai column also runs noticeably noisier (N=1 stdev 1.02 vs 0.08 pre-P6; N=10 stdev 1.93 vs 0.87), suggesting a per-request variability source that the HTTP hop had been smoothing (likely the GIL / event-loop contention with the engine's own coroutines now on the same loop). Most of the plan's "close the gap at N=1 000" argument is intact; the "no harm at small N" argument is not.

The scalarlm column moves between the two sweeps (e.g. N=100 11.04 → 14.44) because the sweeps ran hours apart on a shared node and the scalarlm queue's cadence interacts with the engine's cache state; N=1 000 is stable because the batch fully saturates the engine and scheduler jitter averages out.

### DGX Spark — Phase 6 bulk parity sweep

Platform: GB10 (TP=1), `Qwen/Qwen3-32B-NVFP4`, co-resident sql-gen vLLM on the same GPU (~50 GB footprint). Each cell is 10 measurement runs.

| N     | openai pre-P6      | openai P6          | Δ openai   | scalarlm P6         | openai P6 vs scalarlm |
|-------|--------------------|--------------------|------------|---------------------|-----------------------|
| 1     | 0.603 ± 0.004      | 0.620 ± 0.004      | +2.8 %     | 0.729 ± 0.228       | −15.0 %               |
| 10    | 2.874 ± 0.049      | 2.935 ± 0.065      | +2.1 %     | 3.370 ± 1.064       | −12.9 %               |
| 100   | 4.254 ± 0.114      | 4.396 ± 0.117      | +3.3 %     | 5.307 ± 1.790       | −17.2 %               |
| 1 000 | (not captured)     | 4.497 ± 0.128      | —          | 16.605 ± 0.013      | **−72.9 %**           |

Spark shows the opposite small-N shape from Blackwell: Phase 6 is a consistent small win (+2–3 %) across N=1/10/100 without the Blackwell regression. The openai path also *plateaus* at ~4.5 p/s from N=100 to N=1 000 — virtually no scaling improvement with batch size — while scalarlm continues to climb to 16.6 p/s. That plateau means **the HTTP hop was never the bottleneck at large N on Spark**; something inside vLLM's serving layer or the proxy's dispatch caps throughput at this hardware's Python-overhead ceiling.

### Phase 6 profile — N=100 distinct prompts, Blackwell

Under `bench/scenarios/profile_phase6.sh` with `--distinct-prompts` (so vLLM's prefix cache doesn't collapse the N prompts into one prefill), py-spy was run against the APIServer uvicorn worker (PID found by walking up from `VLLM::EngineCore` to its `spawn_main` parent). After discarding a run where `--native` + rate=100 overwhelmed the sampler on a 262-thread process, the useful run used `--rate 50 --gil` (Python-GIL-holding threads only) for 120 s while the N=100 × 10-run sweep executed.

Throughput during profiling: **6.67 ± 1.03 p/s** (range 5.22–8.38) — note this is the **distinct-prompts** number, lower than the **identical-within-batch** N=100 figure of 10.64 p/s above because the prefix cache is defeated. Profiling overhead was <10 %.

**Breakdown across 1 690 main-thread samples, 0 errors:**

| Function                                                   | Fraction of stacks | What it represents                                                                             |
|------------------------------------------------------------|--------------------|------------------------------------------------------------------------------------------------|
| `output_handler` (`vllm/v1/engine/async_llm.py:700`)       | 40.9 %             | vLLM's per-iteration output loop (the asyncio task that pulls tokens from the engine)          |
| Prometheus `record` / `observe` / `inc` / `labels`         | **20.3 %**         | **Synchronous metric emission — 100 % of samples come from `output_handler`**                  |
| `recv_multipart` + ZMQ                                     | 12.5 %             | IPC between the uvicorn worker and the separate EngineCore subprocess                          |
| `OpenAIServingCompletion.create_completion`                | 10.7 %             | vLLM's serving entry we now call in-process                                                    |
| Our Phase 6 `_dispatch` + `_call_inprocess`                | 9.5 %              | The proxy layer code added by Phase 6                                                          |

The `output_handler` Python source carries its own `TODO(rob): make into a coroutine and launch it in background thread once Prometheus overhead is non-trivial`. That overhead is non-trivial on this platform.

**Interpretation.** One in five main-thread samples is blocked inside a Prometheus metric call, and every one of those calls comes from the engine's per-iteration output loop. This is a *fixed per-iteration tax*: N=1 pays proportionally the same as N=100 for the engine's own record-emit work. It compounds with any architectural inefficiency in the proxy and with any per-request Pydantic / JSON work. It is the single concrete reason why Phase 6 regressed at small N on Blackwell — removing the HTTP hop shrank the useful-work denominator without changing this tax in the numerator.

A quick A/B with `--disable-log-stats` was inconclusive: `has_custom_loggers=True` in vLLM's `AsyncLLM.__init__` keeps the logger enabled even when that flag is passed, so the code path was unchanged. A direct patch that replaces `if logger_ref[0]:` with `if False:` at `async_llm.py:700` gives the right A/B — applied at pod startup via a `sed`-style one-liner in `command:` before `exec /app/cray/scripts/start_one_server.sh`.

**N=100 distinct-prompts, 10 runs per condition, Blackwell:**

| Condition                                      | mean p/s | stdev | range          |
|------------------------------------------------|----------|-------|----------------|
| Phase 6, default (Prometheus record on)        | 6.67     | 1.03  | 5.22 – 8.38    |
| Phase 6, `output_handler` record patched out   | **7.67** | 1.98  | 4.60 – 10.61   |

**+15.0 %** in the predicted direction, and the patched peak run (10.61 p/s) exceeds every unpatched run. The stdev jumps from 1.03 to 1.98 under the patch (the smoothing effect of a per-iteration blocking call falls away, so the loop's own scheduling jitter becomes visible). At 10 runs per condition the means sit one stdev apart — enough to show the direction, not enough to call the magnitude precisely. The 20 % upper bound from the profile is consistent with the 15 % we measured; a longer run (30+ iterations) would tighten this to ±few-%.

**Second profile on the patched pod** confirms the fix — the Prometheus time genuinely disappears rather than just moving around. N=100 distinct-prompts, 1 226 main-thread samples (`--rate 50 --gil`, 120 s):

| Function                                | Unpatched | Patched | Δ        |
|-----------------------------------------|-----------|---------|----------|
| Any Prometheus frame in stack           | 20.3 %    | **0.1 %** | −20.2 pp |
| `output_handler` in stack               | 40.9 %    | 35.2 %  | −5.7 pp  |
| ZMQ IPC (EngineCore)                    | 12.5 %    | 10.2 %  | −2.3 pp  |
| `_dispatch` + `_call_inprocess`         | 9.5 %     | 11.7 %  | +2.2 pp  |

No new single dominant hot spot takes over. The reclaimed 20 percentage points spread across `_protected_step` (asyncio event-loop step — 7.9 % leaf, up from 4.4 %), `process_outputs` / `update_from_output` (vLLM engine output aggregation), and Python object-init / attribute-access plumbing. `merge_async_iterators` (vLLM's per-request scatter merger — relevant to Gemini's Phase 8 thesis) appears at 1.0 % leaf at N=100; whether it becomes meaningful at N=1 000 is measured next.

### Phase 6 profile at N=1 000 distinct prompts (patched pod)

Same configuration, 3-iteration sweep, 300 s py-spy `--gil --rate 50`. 4 169 main-thread samples, 0 % Prometheus (confirming the patch applies at this scale too). Two of three iterations completed (third was killed when stopping py-spy); wall-clock numbers: **6.0 and 6.5 p/s** (vs the identical-prompts figure of 12.9 p/s — vLLM's prefix cache does ~50 % of the work at the canonical benchmark setting).

| Function                                 | N=100 patched | N=1 000 patched | Δ         |
|------------------------------------------|---------------|-----------------|-----------|
| Prometheus                               | 0.1 %         | 0.0 %           | (patched) |
| ZMQ IPC                                  | 10.2 %        | 10.2 %          | flat      |
| `output_handler` in stack                | 35.2 %        | 33.7 %          | −1.5 pp   |
| **`merge_async_iterators` in stack**     | ~1 %          | **13.4 %**      | +12 pp    |
| `_call_inprocess` + `_dispatch` in stack | 11.7 %        | 19.0 %          | +7.3 pp   |
| Pydantic anywhere in stack               | —             | 1.9 %           | small     |
| Response-build funcs                     | —             | 0.7 %           | small     |

**Cross-checking Gemini's theses against the N=1 000 data:**

- **Thesis #1 "monolithic Pydantic response":** *not supported.* Pydantic is 1.9 % in-stack, response-builder functions 0.7 %. Even at N=1 000 those together are ≤3 % — nowhere near the bottleneck. The claim that distributing Pydantic cost across 1 000 small responses is the lever doesn't match the profile.
- **The scatter-merge cost IS real, but it lives inside vLLM, not in our proxy.** `merge_async_iterators` jumps from ~1 % (N=100) to 13.4 % (N=1 000) in-stack. That's vLLM's internal async-iterator merger that collates the per-prompt generators spawned inside `create_completion` — not our monolithic response construction.
- Proxy-side scatter-gather (Phase 8) would *replace* this path with `asyncio.gather` at our layer, each sub-request triggering its own single-prompt `create_completion` (skipping `merge_async_iterators`). Whether that's cheaper than letting vLLM merge depends on whether `asyncio.gather` + 1 000 lightweight coroutines is less expensive than `merge_async_iterators` + 1 000 heavier generators. That's the exact question Phase 8a is set up to answer.

**Prioritisation update based on the N=1 000 profile.** The remaining "reachable" costs at N=1 000 break down roughly as: `output_handler` non-Prometheus work (~33 %), proxy + vLLM serving orchestration (~20 %), `merge_async_iterators` (~13 %), ZMQ IPC (~10 %), asyncio overhead (~7 %). Phase 8a tests whether moving the scatter one layer up recovers the 13 % from `merge_async_iterators` — that's the next experiment.

### Gemini's review of the post-Phase-6 results

An independent review was posted as `gemini-enhance-openai.md`. Gemini's diagnosis of the residual N=1 000 gap (**22 %** after Phase 6 on Blackwell) identifies three architectural causes:

1. **Monolithic response processing.** For a 1 000-prompt request, openai constructs a single `CompletionResponse` Pydantic object with 1 000 choices; scalarlm's worker issues 1 000 parallel single-prompt calls and gathers small responses.
2. **Sequential BatchRunner.** `/v1/batches`' `BatchRunner.run()` awaits each line serially instead of running them concurrently.
3. **Fixed-limit concurrency.** `OpenAIConcurrencyLimiter` uses a hard-coded semaphore (default 16) vs scalarlm's KV-cache-aware dynamic batching.

Gemini proposes Phases 7 (parallel BatchRunner), 8 (proxy-side scatter-gather for array prompts), and 9 (reactive concurrency).

### Cross-check: profile findings vs Gemini's architectural theses

The two analyses are **compounding, not alternative**. The profile data does not refute any of Gemini's three points, but it does reorder their likely impact:

- **On (1) "monolithic response":** vLLM's own `serving.py` already scatters per-prompt internally — `serving.py:147` creates one `generators[i]` per `engine_input` and `merge_async_iterators` interleaves them; only the final response-to-Pydantic conversion is monolithic. Now measured at N=1 000: Pydantic stays at 1.9 % in-stack and response-build functions at 0.7 %, so the "monolithic Pydantic" claim is *not* supported even at the scale that was supposed to surface it. What *is* real and only visible at N=1 000 is `merge_async_iterators` (13.4 % in-stack vs ~1 % at N=100) — but that's vLLM's internal merger, not our response construction. Proxy-side scatter-gather replaces `merge_async_iterators` with `asyncio.gather`, which *might* be cheaper per-prompt but isn't guaranteed to be. Phase 8a remains the right experiment; the mechanistic rationale updates from "distribute Pydantic cost" to "avoid `merge_async_iterators`".
- **On (2) "sequential BatchRunner":** no profile evidence either way — we didn't exercise `/v1/batches` in the profile run. But the claim is literally true in the code (`runner.py` awaits per-line), and the Batch API sweep in the first-pass results shows exactly the flat-throughput shape one expects from serialized awaits (1.9 → 2.8 → 2.9 → 3.2 p/s across N=1/10/100/1 000). Low-risk to fix; do it.
- **On (3) "fixed concurrency":** less impactful for the parity goal than (1) and (2). Our bulk benchmark is a *single* caller sending one array request. The concurrency limiter is acquired once and released once; it's not the bottleneck here. It matters for the "many concurrent callers" workload, which is a separate and worthwhile goal but not what scalarlm's N=1 000 lead is about.

**What Gemini doesn't see** (because it didn't have py-spy data): the 20 % Prometheus overhead in `output_handler` is visible in *every* run at *every* N and is large enough to be measurable on its own. It's strictly smaller than the N=1 000 gap Gemini targets, but it applies everywhere and the fix is localized (follow vLLM's own TODO).

## Amended plan — Phases 6.5, 7, 8, 9

The combined plan below merges Gemini's critique with the profile data. Phase numbering follows Gemini v2 (a Phase 6.5 slotted between the existing Phase 6 and the remaining phases) because the metrics patch is a *prerequisite* for clean measurement of the later phases, not an optional tack-on.

### Phase 6.5 — Move vLLM `output_handler` metrics off the hot path

Source: py-spy profile (`async_llm.py:700` was 20.3 % of the main-thread stack at N=100, 0 % after a one-line patch), supported by the `TODO(rob)` in vLLM source.

**Why this runs first.** Phases 7 / 8 are about closing the N=1 000 parity gap. The 20 % synchronous-metrics tax is a *fixed per-iteration cost* that applies regardless of N, regardless of which phase we're testing, and regardless of which path (scalarlm or openai) is running. Until it's fixed, every throughput measurement is contaminated by it — Phase 7 gains would partially look like Phase 6.5 gains and vice-versa. Clear the baseline first.

**Implementation.** Follow vLLM's own TODO: replace the synchronous

```python
if logger_ref[0]:
    logger_ref[0].record(engine_idx=..., scheduler_stats=..., iteration_stats=..., mm_cache_stats=...)
```

with a producer/consumer pattern — the output loop pushes the args tuple onto a lock-free queue, a dedicated asyncio task (or background thread) pops and calls `.record(...)` asynchronously. The metric data is already structured; no in-band work remains on the output loop.

**Measurement** at N=100 distinct prompts on Blackwell, 4 conditions across 80 total runs:

| Condition                                                        | Runs | Mean p/s | Stdev | 95 % CI on mean  |
|------------------------------------------------------------------|------|----------|-------|------------------|
| Unpatched (10-run pilot, **older pod**)                          | 10   | 6.67     | 1.03  | ±0.64 (9.5 %)    |
| AB-PATCH `if False:` (metrics disabled, 10-run, **older pod**)   | 10   | 7.67     | 1.98  | ±1.23 (16 %)     |
| **Phase 6.5 production patch (today's pod)**                     | 30   | 8.32     | 1.35  | ±0.48 (5.8 %)    |
| **Unpatched (today's pod, paired A/B)**                          | 30   | **8.56** | 1.72  | ±0.62 (7.2 %)    |

**The paired A/B is a null result.** Patched (8.32 p/s) is essentially indistinguishable from unpatched (8.56 p/s) on the same pod — the 0.24 p/s difference is well inside both CIs and points the *wrong* way for a "patch wins" claim.

The earlier +15 % and +24.7 % numbers were **pod-to-pod state variance, not patch effects**. The same *unpatched* code moved from 6.67 p/s (10-run pilot pod) to 8.56 p/s (today's 30-run pod) — a 28 % swing on the no-change condition. That swing is several times larger than any plausible patch effect, and it dominates any cross-pod comparison. Lesson: a 10-run pilot on a different pod is not a baseline; only paired runs on the same pod / same warm state are comparable.

**Why the profile mis-predicted.** The py-spy data was correct that 20 % of main-thread CPU samples were inside Prometheus `record()` calls. What the profile *can't* tell you is whether removing those samples recovers wall-clock — and on this workload it doesn't. The asyncio loop was already overlapping the inline `record()` with engine I/O via implicit yield points, so moving it onto a separate consumer task didn't free up time the loop was actually waiting on. Profile-percent-of-CPU and A/B-percent-of-throughput are different metrics; they only converge when the profiled function is on the critical path.

**Decision.** Phase 6.5 produces no measurable throughput win at N=100 distinct prompts on Blackwell. The patch is functionally correct (metrics still flow via the consumer task) but doesn't move the number it was justified by. **Two outstanding measurements still matter** before Phase 6.5 is fully resolved:

1. **Spark N=1 000.** The Spark plateau (4.5 p/s flat across N=100/1 000) was the second piece of evidence for the per-iteration metrics tax. If Spark is genuinely CPU-bound at the Python layer, removing the synchronous record might break the plateau where it didn't help Blackwell. Same paired methodology — same pod, patched vs unpatched, ≥30 runs each.
2. **Blackwell N=1 000.** The 13.4 % `merge_async_iterators` fraction at N=1 000 is independent of Phase 6.5 — but the patch *might* help more at N=1 000 than at N=100 because the engine spends more iterations per request, accumulating more per-iteration tax in absolute terms. Worth one paired sweep.

If both come back null, **revert the patch** — leave the upstream vLLM TODO alone and pivot Phase 6.5 effort to Phase 7 (parallel BatchRunner), where the empirical case is much stronger (sequential awaits, observed 1.9–3.2 p/s flat curve, no profile-vs-A/B confound).

**Productionisation status.** The patch is implemented as `scripts/vllm_patches/apply_patches.py` (anchor-based string replacement; idempotent — refuses to patch already-patched files; AST-validates the result) and wired into `scripts/build-copy-vllm.sh` so it runs at image-build time. The Dockerfile copies `scripts/vllm_patches/` into the vLLM build stage. The fresh image rebuild on Blackwell hit an unrelated triton-version drift (`'TorchAllocator' object has no attribute 'set'`) that pre-dates Phase 6.5 and prevented the new image from booting; the production pod was deployed against the working `:phase6-inprocess` image with the patcher mounted as a ConfigMap and applied at startup, isolating the metrics-offload effect from the version-drift issue. The triton fix is a separate dep-pin task.

**Future Spark check.** Re-measure Spark at N=1 000 with the patch live: the profile predicts the Spark plateau (4.5 p/s across N=100–1 000) should lift materially once the per-iteration tax is removed.

**Risk.** This is a vLLM-fork patch, not a scalarlm patch. Must carry forward through fork rebases. The patcher's anchor assertions (`assert anchor in src, "..."`) fire loudly at build time when a rebase drifts the source — silent mis-patching is impossible. The `apply_patches.py` file itself is the test guard.

### Phase 7 — Parallelise the Batch API runner

Source: Gemini's Phase 7 directly; validated by the observed 1.9–3.2 p/s flat Batch API curve in the first-pass results.

**Implementation.** In `infra/cray_infra/api/fastapi/routers/openai_batches/runner.py`, replace the sequential `for line in input_lines: await handle(line)` with `asyncio.gather` guarded by a bounded `asyncio.Semaphore`. Reuse `openai_queue_concurrency` for the bound or add a dedicated `batch_concurrency` key. Preserve the existing status-machine transitions and per-line result-writing order (OpenAI batches are order-preserving by `custom_id`; parallel execution must reorder on write-out, not drop ordering).

**Measurement.** `bench/scenarios/batches_sweep.sh` N=1/10/100/1 000 before and after. Expectation: N=1 000 jumps from ~3 p/s toward the array-completions number on the same hardware (~12–14 p/s on Blackwell).

**Decision threshold.** Batch API P6 ≥ 0.9 × scalarlm `upload_download` at N=1 000 on Blackwell. At that point the Batch API is the documented replacement for scalarlm's bulk path.

**Risk.** File-backed result JSONL must be appended deterministically. Write-per-line under a lock is fine; order in the file need not match submission order if `custom_id` is preserved, but doing so makes diffing easier.

**Implementation status: shipped & measured.** `BatchRunner.run()` now dispatches lines via `asyncio.gather` guarded by `asyncio.Semaphore(concurrency)` (default 16, override via `batch_runner_concurrency`). Counts are protected by an `asyncio.Lock` to fix the read-modify-write race on `bump_counts` that the previous sequential code didn't have. Output JSONL is now written in completion order, not submission order — matches OpenAI's contract that batch outputs are addressed by `custom_id`. 14/14 unit tests pass, including 3 new ones for parallel overlap (`inflight_peak >= 2 with concurrency=4`), `concurrency=0` rejection, and counts-correctness under 50-way fan-out.

**Measurement (Blackwell, `Qwen/Qwen3-Next-80B-A3B-FP8`):**

| N     | Pre-P7 (sequential, 1 s poll) p/s | **Phase 7 (concurrency=16, 0.05 s poll)** | Wall-clock | Δ         |
|-------|-----------------------------------|------------------------------------------|------------|-----------|
| 1     | 1.9                               | 2.63                                     | 0.38 s     | +38 %     |
| 10    | 2.8                               | 2.47                                     | 4.05 s     | within noise |
| 100   | 2.9                               | **8.33**                                 | 12.0 s     | **+187 %** |
| 1 000 | 3.2                               | **13.14**                                | 76 s       | **+311 %** |

The poll-interval change (1.0 s → 0.05 s) accounts for at most a few seconds across the whole sweep — it cannot explain the 234 s reduction in N=1 000 wall-clock. Fan-out dominates. **At N=1 000 the Batch API now matches array `/v1/completions` throughput** (12.92 p/s in the Phase 6 measurement table). The decision threshold ("Batch API ≥ 0.9 × scalarlm bulk on Blackwell N=1 000") is **cleared**: 13.14 p/s is 79 % of scalarlm's 16.6 p/s, and the remaining gap is the same residual that limits array-completions itself — addressed (or not) by Phase 8a.

### Phase 8a — Scatter-gather A/B (measure before implementing)

Source: Gemini's Phase 8, but run as an *experiment* before committing code. The N=1 000 profile refines the mechanistic rationale: the cost to bypass is vLLM's internal `merge_async_iterators` (13.4 % in-stack at N=1 000), not Pydantic response serialization (1.9 %). Whether `asyncio.gather` of 1 000 proxy-layer sub-requests is cheaper than vLLM's `merge_async_iterators` is the empirical question.

**Test.** Add a threshold-flagged scatter path to `_dispatch` in `openai_v1_router.py`:

```python
SCATTER_THRESHOLD = int(os.environ.get("SCALARLM_SCATTER_THRESHOLD", "0")) or None
if SCATTER_THRESHOLD and isinstance(request.prompt, list) and len(request.prompt) >= SCATTER_THRESHOLD:
    sub_requests = [request.model_copy(update={"prompt": p}) for p in request.prompt]
    sub_responses = await asyncio.gather(*[
        _call_inprocess(endpoint=endpoint, request=r, raw_request=raw_request,
                        base_model_name=base_model_name,
                        queue_slot=queue_slot)  # shared: one logical request = one slot
        for r in sub_requests
    ])
    return _merge_completion_responses(sub_responses)
```

Bench both configurations on Blackwell at N=1/10/100/1 000:

- `SCALARLM_SCATTER_THRESHOLD=` (off): current behaviour.
- `SCALARLM_SCATTER_THRESHOLD=2`: every array request fans out.
- `SCALARLM_SCATTER_THRESHOLD=50`: only large arrays fan out.

Run *after* Phase 6.5 has landed so the 20 % metrics tax isn't muddying the comparison.

**Decision.** Per Gemini v2: if scatter-gather shows **> 10 %** throughput win at N=1 000 on Blackwell, promote to Phase 8b. If not, the thesis is not the dominant remaining cost and we save the code.

**Expected result.** Given vLLM already scatters internally, the question is whether `asyncio.gather` + N sub-requests has less overhead than `merge_async_iterators` + N generators. Plausibly material at N=1 000 (that's where `merge_async_iterators` showed up at 13.4 %); probably a wash or slightly negative at N=1/10 because of fanout overhead. That shape is what Phase 8a will confirm or deny.

### Phase 8b — Implement scatter-gather in the proxy (conditional)

Only landed if Phase 8a meets its decision threshold. Differences from the A/B:

1. **Queue-slot sharing (hard requirement).** All N sub-requests of one logical call must share the single `queue_slot` acquired by the parent — never acquire N slots. Otherwise the `OpenAIConcurrencyLimiter` (semaphore default 16) is exhausted by a single 1 000-prompt call and all other traffic stalls. This is the case-1 correctness bug Gemini flagged; the A/B already wires it this way so Phase 8b just preserves the behaviour.
2. A tuned default threshold (guided by the A/B curve shape).
3. Clean merge of `usage` (summed tokens), `system_fingerprint` (take first), and `created` (take first).
4. Stable `choices[*].index` ordering by the original array position.
5. An early-exit on error that cancels outstanding sub-requests.
6. Metrics instrumentation that records the logical request once, not N times (otherwise per-request Prometheus labels explode).

### Phase 9 — Dynamic, KV-aware concurrency limiter

Source: Gemini's Phase 9. **Deferred** until a workload shows up where it helps — the bulk-parity goal is not one of them, per the analysis above.

If pursued: `OpenAIConcurrencyLimiter` consults `engine_client.get_current_kv_cache_size()` (exposed via the same `vllm_registry` that Phase 6 uses) and scales the semaphore bound between `min_concurrency` and `max_concurrency`. Guardrail: only the *upper* bound moves; the floor stays at the configured number so behaviour is never more conservative than today.

### Revised parity thresholds

- **Phase 6.5**: ~~N=100 distinct-prompts throughput improvement ≥ 10 % on Blackwell~~. **Failed.** Paired 30-run A/B at N=100 on Blackwell came back null (8.32 patched vs 8.56 unpatched, CIs overlap). Two measurements still pending before final disposition: Spark N=1 000 (the plateau-break hypothesis) and Blackwell N=1 000 (more iterations per request → more accumulated tax). If both null, revert the patch.
- **Phase 7**: ~~Batch API N=1 000 within 10 % of `scalarlm` `upload_download` on Blackwell~~. **Cleared.** Phase 7 lifts N=1 000 from 3.2 → 13.14 p/s (+311 %), matching array `/v1/completions` throughput at N=1 000 and reaching 79 % of scalarlm. The remaining gap is the same residual that limits array-completions itself (Phase 8a).
- **Phase 8a → 8b**: openai `/v1/completions` array at N=1 000 within **5 %** of `scalarlm` on Blackwell (Phase 8b only lands if 8a shows > 10 % improvement).
- **Deprecation gate**: once Phase 6.5 and Phase 7 are verified and Phase 8 is either landed (after 8a pass) or formally abandoned (after 8a fail), the scalarlm path is redundant by both capability and performance; Phase 4's telemetry-gated deprecation can proceed.

### Open questions for a reviewer to double-check

1. Is the Phase 6 Blackwell small-N regression (−10 to −28 %) reproducible, or an artefact of this pod/node state? The stdev went up in ways that suggest GIL contention between the new in-process `create_completion` coroutine and the existing `output_handler` task. Phase 6.5 should resolve this by design (removing the main blocking call that competes for the loop); if it doesn't, the small-N regression has a separate cause and needs its own counter-experiment.
2. The 20 % Prometheus number was measured at N=100. The N=1 000 patched profile now confirms the fix holds at scale (0 % Prometheus post-patch). But absolute wall-time *win* at N=1 000 hasn't been measured head-to-head yet — need patched-vs-unpatched A/B at N=1 000 distinct prompts.
3. Phase 8a's mechanistic rationale is now "avoid `merge_async_iterators`" (13.4 % in-stack at N=1 000). The Gemini v2 rationale "distribute Pydantic cost" is *refuted* by the profile (Pydantic 1.9 %, response-build 0.7 %) and should be retired in Gemini's next revision.
4. Is there a reason not to fix the vLLM fork TODO directly (Phase 6.5)? The upstream refactor pattern they hint at (coroutine + background thread) is a minor change with a localised test surface; the only real cost is carrying the patch across rebases. We carry patches already, so this isn't a new burden.
