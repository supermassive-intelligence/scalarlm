# Refined Critique and Deep-Dive Plan: The 49% Gap

## Status
Revised Analysis and Pivot Plan. Incorporates the refutation of Phase 6.5 and 8a and the discovery of the 49% "Real-World" performance gap at $N=1000$ (distinct prompts).

## The "Real-World" Parity Picture
Previous measurements were optimistic due to prefix-cache help on identical prompts. Properly paired A/B testing with **distinct prompts** on Blackwell reveals a much larger gap:
- **ScalarLM Path**: 16.57 p/s (Stable, CV 0.2%)
- **OpenAI Path**: 8.35 p/s
- **Disadvantage**: **-49%** (Nearly 2x slower)

## Refutation of Previous Hypotheses
The following "architectural suspects" have been empirically cleared (null results in paired A/B):
1.  **Synchronous Metrics Tax (Phase 6.5)**: Offloading metrics to a background task did not improve Blackwell N=100 throughput. The asyncio loop was likely already overlapping this work.
2.  **vLLM Scatter-Merge Overhead (Phase 8a v1)**: Proxy-side scatter-gather (bypassing `merge_async_iterators`) yielded only +0.9% gain.
3.  **Scheduler Pressure (Phase 8a v2)**: Bounding in-flight requests to 16 actually *decreased* performance.
4.  **Object/Request Sharing (Phase 8a v3)**: Fresh `raw_request` and `CompletionRequest` objects per sub-call were also null.

## Pivot Plan: Investigating the "Hidden" Delta

The gap lives in the delta between the ScalarLM worker and the OpenAI proxy's execution context.

### Phase 8a v4: The Decorator Hypothesis (Priority 1)
- **Difference**: The ScalarLM worker calls `vllm.entrypoints.openai.completion.api_router.create_completion`, which is wrapped in `@with_cancellation` and `@load_aware_call`. The OpenAI proxy calls the underlying serving class directly.
- **Test**: Route the OpenAI proxy through the decorated `api_router` function.
- **Rationale**: These decorators may be performing critical engine-level orchestration or yielding that the direct serving call lacks.

### Phase 8a v5: Comparative py-spy Profiling (Priority 2)
If v4 is null, we must capture a "ScalarLM-only" profile to compare against the OpenAI profile.
- **Search for**: Differences in ZMQ IPC latency, engine yield points, or main-thread "bubbles" where the OpenAI path is waiting and the ScalarLM path is working.

### Phase 6.5 Follow-up & Resolution
- **Spark Plateau**: Test if the Spark 4.5 p/s ceiling is lifted by the metrics patch.
- **Blackwell N=1000**: Test if the tax accumulates enough to be visible at higher engine iteration counts.
- **Decision**: Revert the vLLM fork patch if both are null. It is correct but "dark code" (no measurable ROI).

---

## Revised Decision Thresholds

1.  **Parity Target**: The 49% gap must be closed to **$<10\%$** at $N=1000$ distinct prompts before marking Path A for deprecation.
2.  **Batch API Success**: Phase 7 is considered **Complete** (13.14 p/s). It is now the primary migration target for bulk offline users.
3.  **Deprecation Gate**: Path A stays live until the "Decorator vs. Serving" delta (v4) is resolved.
