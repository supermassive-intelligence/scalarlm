# Targeted Plan: Closing the 38% "Efficiency Gap"

## Status
Revised Analysis and Optimization Plan. Following the success of Phase 8a v4 (Decorator Routing) and the reframing of the bottleneck via comparative profiling.

## Current State: The "Idle Thread" Paradox
As of Phase 22, we have a breakthrough: the `openai` path's main thread is **3x more idle** than the `scalarlm` thread during $N=1000$ runs (as measured by scraper response rates), yet it produces significantly less throughput.
- **The Finding**: The bottleneck has moved out of the APIServer. We are no longer limited by Python overhead or loop efficiency; we are waiting on the **vLLM EngineCore** or the **ZMQ inter-process transport**.
- **Gap Status**: -38% behind Path A at $N=1000$ distinct prompts (10.33 p/s vs 16.57 p/s).

## Optimization & Exploration Plan (Phases 8c – 25)

### Phase 8c: "Yield Injection" Experiment (High ROI)
- **Goal**: Simulate the engine-feeding behavior of the ScalarLM path.
- **Action**: Insert `await asyncio.sleep(0)` (or a tiny sleep) inside the `_scatter_gather_completions` loop.
- **Rationale**: If the bottleneck is coroutine starvation, forcing yields will allow the engine's I/O tasks to run more often during the massive fanning-out of 1,000 requests.

### Phase 8d: Decorator & Response-Wrap Isolation
- **Goal**: Isolate the +19% gain from Phase 8a v4.
- **Test A**: Direct serving + manual `@with_cancellation` decorator.
- **Test B**: Direct serving + manual `JSONResponse` wrapping (vs. direct Pydantic return).
- **Rationale**: Identify exactly which part of the FastAPI handler stack provides the boost to ensure the productionized version (Phase 8b) is optimized.

### Phase 8e: Comparative Yield Profiling
- **Action**: Use `py-spy` to specifically measure the frequency and duration of `output_handler` execution blocks during both paths.
- **Search for**: "Main thread bubbles" where the engine is idle because the API coroutines are holding the GIL too long.

### Phase 10: Engine Namespace Alignment
- **Action**: Align the `openai` path's `request_id` generation with ScalarLM's content-hashing scheme.
- **Rationale**: Rule out engine-internal scheduling or caching differences that might favor the stable hashing of Path A.

### Phase 11: High-Performance Serialization
- **Action**: Experiment with `orjson` or `msgspec` for final response serialization.
- **Rationale**: Faster serialization reduces main-thread blocking during large-batch assembly, freeing cycles for engine I/O.

### Phase 12: Micro-Yielding via Offloading
- **Action**: Use `asyncio.to_thread` for the final `_merge_completion_responses` (response assembly).
- **Rationale**: Moves heavy CPU-bound merging off the main event loop to prevent blocking the engine's `output_handler`.

### Phase 13: Event Loop Optimization
- **Action**: Verify and ensure `uvloop` usage in the `APIServer`.
- **Rationale**: `uvloop` is significantly more efficient than the standard `asyncio` loop for high-concurrency tasks.

### Phase 14: Zero-Copy Response Construction
- **Action**: Use raw dictionaries instead of intermediate Pydantic objects during the "scatter" merge phase.
- **Rationale**: Reducing object allocation and attribute copying minimizes GIL pressure during massive batch merges.

### Phase 15: Adaptive "Waker" Task
- **Action**: Implement a high-frequency (5-10ms) background coroutine that calls `await asyncio.sleep(0)`.
- **Rationale**: Guarantees regular event-loop cycling to keep the engine's `output_handler` active even if handlers are "too efficient."

### Phase 16: Engine I/O Deep Dive
- **Action**: Investigate why `process_outputs_socket` takes 2x more time in the OpenAI path.
- **Rationale**: Reducing ZMQ processing overhead directly lifts the throughput ceiling.

### Phase 17: Concurrency Policy Review
- **Action**: Compare ScalarLM's KV-cache-aware batching vs. the OpenAI path's effective concurrency.
- **Rationale**: Determine if ScalarLM's coarse-grained batching provides better CPU cache locality during response processing.

### Phase 18: Spark "Roof" Diagnostics
- **Action**: Implement an `asyncio` loop-lag monitor and export to Prometheus. Capture a `py-spy --idle` wall-clock profile on Spark during the N=1000 plateau.
- **Rationale**: Confirm that the 4.1 p/s ceiling on Spark is a Python-thread saturation event. If loop lag exceeds 50ms, the proxy is fundamentally unable to handle N=1000 concurrency on that CPU architecture.

### Phase 19: Surgical Sub-Step Profiling
- **Action**: Instrument the individual sub-steps of `create_completion` (Tokenization, Request Submission, Response Building) using `time.perf_counter()`.
- **Rationale**: If total wall-clock per call matches but throughput differs, we need to find which sub-step is consuming the ~20ms "extra" Python work that isn't helping the engine.

### Phase 20: Replicating ScalarLM's Batch Pipelining
- **Action**: Test `SCALARLM_YIELD_CHUNK` values aligned with ScalarLM's natural batch sizes (128, 256) rather than tiny chunks (16).
- **Rationale**: ScalarLM's worker naturally pipelines engine-side scheduling with Python coordination by waiting for a batch to finish before submitting the next. All-at-once submission might be causing O(N) overhead in vLLM's waiting-queue management or AsyncStream fan-out.

### Phase 21: AsyncStream Management Audit
- **Action**: Investigate the overhead of 1,000 concurrent `AsyncStream` objects vs. 100.
- **Rationale**: vLLM's `output_handler` must iterate over all active streams to distribute tokens. Scaling to 1,000 might reach a threshold where the per-iteration distribution cost becomes a primary CPU consumer.

### Phase 22: EngineCore O(N) Admission Profiling
- **Action**: Profile the `EngineCore` process (not APIServer) during the OpenAI N=1000 run using `py-spy`.
- **Rationale**: If 1,000 requests are pending in the engine, vLLM's admission controller might be performing O(N) scans to check KV-cache availability. Path A's "batch trickle" naturally keeps this scan small.

### Phase 23: ZMQ Backpressure Investigation
- **Action**: Experiment with ZMQ High-Water Mark (HWM) settings and internal buffer sizes in the vLLM fork.
- **Rationale**: High-concurrency `AsyncStream` fan-out might be hitting process-boundary I/O limits or context-switch thrashing at the ZMQ layer.

### Phase 24: Reactive "Sliding Window" Dispatcher
- **Action**: Modify the OpenAI "scatter" logic to submit new requests **only as old ones finish**, maintaining a constant "engine depth" (e.g., 64).
- **Rationale**: Mimics the exact temporal pattern of Path A's queue worker, potentially smoothing out engine scheduling.

### Phase 25: Offline LLM.generate Bridge
- **Action**: Investigate bridging the synchronous `LLM.generate` API for large array-completions, potentially bypassing the async overhead entirely for bulk.
- **Rationale**: vLLM's offline mode is known to be 2-3x faster than async serving; this bridge addresses the fundamental "Sync vs Async" performance floor.

---

## Revised Parity & Deprecation Gates

1.  **Parity Target**: Close the $N=1000$ gap to **<15%** (approx. 14+ p/s). Given the inherent overhead of the OpenAI protocol, this is the realistic "good enough" threshold.
2.  **Production Defaults**: Promote `SCALARLM_SCATTER_THRESHOLD` and the optimized v4/v5 routing logic to defaults once isolation (8d) is complete.
3.  **Deprecation Gate**: Path A is marked for deprecation once the 15% threshold is reached OR the "Yield Injection" results prove that further gains require emulating Path A's suboptimal logging/polling patterns.
