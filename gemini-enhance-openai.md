# Enhanced Critique and Optimized Plan: Path to OpenAI Parity

## Status
Revised Analysis and Optimized Plan. Incorporates Phase 6 measurement data and py-spy profiling results.

## Revised Critique of Phase 6 Results

The Phase 6 implementation (In-Process Transport) has provided deep insight into the remaining bottlenecks. While it closed the transport gap at $N=1000$ on Blackwell (shifting from -27% to -22% relative to ScalarLM), it introduced regressions at small $N$ and hit a hard throughput plateau on Spark.

### 1. The "Fixed Per-Iteration Tax" (Empirically Verified)
The py-spy profile reveals that **20.3% of the main-thread time** is spent in synchronous Prometheus metric calls within vLLM's `output_handler`. This is a constant tax paid on every engine iteration. 
- **Impact**: This explains the small-$N$ regressions on Blackwell. By removing the HTTP hop, the useful-work denominator shrank, making this synchronous metrics tax the dominant overhead.
- **Spark Plateau**: On Spark, the system plateaus at ~4.5 p/s regardless of $N$. This confirms the bottleneck is purely CPU-bound at the Python/Serving layer, driven by this iteration tax.

### 2. Monolithic vs. Distributed Overhead
The ScalarLM path wins at $N=1000$ because it distributes the cost of response serialization across 1,000 parallel `asyncio` tasks. The OpenAI path's attempt to build and serialize a single 1,000-choice Pydantic object creates a massive burst of GIL-holding work that blocks the event loop.

### 3. Sequential Batch Bottleneck
The Batch API remains an order of magnitude slower (~3 p/s) because it awaits line items serially. This is a purely architectural gap that prevents it from replacing the ScalarLM `upload/download` path.

---

## Optimized Plan (Phases 6.5 – 9)

The plan is re-ordered to prioritize the highest-ROI fixes and establish a clean baseline for final parity.

### Phase 6.5: Offload Engine Metrics (The "vLLM TODO" Patch)
Address the 20% overhead found in the profile by productionizing the fix for vLLM's `output_handler`.
- **Implementation**: Patch the vLLM fork to move synchronous `.record()` calls in `async_llm.py` to a background producer/consumer queue.
- **Goal**: Regain the 10–15% throughput loss observed in the pilot and eliminate the Spark plateau.
- **Requirement**: This must be done *before* final parity sweeps to ensure the "engine noise" is removed.

### Phase 7: Parallelize Batch API Execution
- **Implementation**: Modify `BatchRunner.run()` to use `asyncio.gather` with a semaphore.
- **Ordering**: Ensure output JSONL preserves `custom_id` mapping.
- **Concurrency**: Target parity with ScalarLM's `upload/download` by allowing up to 64–128 concurrent lines per batch (configurable).

### Phase 8: Proxy-side Scatter-Gather (Experimental & Conditional)
Execute the experiment to determine if fanning out array prompts at the proxy closes the final N=1000 gap.
- **Scatter Logic**: If $N > \text{threshold}$, split the array prompt into $N$ sub-requests.
- **Slot Management**: **Critical**: The sub-requests must share the *same* `queue_slot` or be treated as a single logical unit to avoid starving the `OpenAIConcurrencyLimiter`.
- **Merge Logic**: Implement a lightweight merger for sub-responses (summing tokens, concatenating choices, taking the first `system_fingerprint`).
- **Decision Gate**: Only ship Phase 8b if the Phase 8a experiment shows a $>10\%$ throughput win at $N=1000$.

### Phase 9: Reactive Concurrency Limiting
- **Implementation**: Link `OpenAIConcurrencyLimiter` to the `vllm_registry` to consult real-time KV cache headroom.
- **Policy**: Dynamically scale the semaphore bound to match the "aggressive" behavior of the ScalarLM worker when resources allow.

---

## Revised Decision Thresholds

1.  **Metric ROI**: Phase 6.5 should demonstrate a **$>10\%$ baseline improvement** across all $N$ on Blackwell.
2.  **Batch Parity**: `/v1/batches` throughput $\ge 90\%$ of `scalarlm` `upload/download` at $N=1000$.
3.  **Final Parity**: `openai` `/v1/completions` array at $N=1000$ within **5%** of `scalarlm` (or better).

**Final Step**: Once Phase 6.5 and 7 are verified, and Phase 8 (if applicable) is landed, the ScalarLM path is obsolete. Proceed with Phase 4 (Telemetry-Gated Deprecation).
