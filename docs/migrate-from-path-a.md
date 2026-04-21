# Migrating from `/v1/generate` to the OpenAI-compatible surface

This is the client-facing complement to [`enhance-openai-api.md`](../enhance-openai-api.md). Both ScalarLM surfaces talk to the same vLLM process; the OpenAI-compatible one is the long-term path. Existing `/v1/generate` callers see a warning log on every request (event `path_a_deprecation`). The plan is telemetry-driven: once that log stops firing for a grace period, Path A goes away.

Concrete swaps follow. Every example uses the `openai` Python SDK since that's what most callers already have installed.

## Single-prompt completion

**Before (Path A):**
```python
import scalarlm
scalarlm.api_url = "http://scalarlm.internal"
client = scalarlm.SupermassiveIntelligence()
[response] = await client.async_api.generate(["hello"], model_name="my-finetune", max_tokens=128)
```

**After (Path B):**
```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://scalarlm.internal/v1", api_key="ignored")
resp = await client.chat.completions.create(
    model="my-finetune",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=128,
)
response = resp.choices[0].message.content
```

Tool calling works on the OpenAI path and does not on Path A — that's the primary reason to migrate.

## Batched completions (small)

**Before (Path A):**
```python
responses = await client.async_api.generate(["p1", "p2", "p3"], model_name="my-finetune", max_tokens=128)
```

**After (Path B — `/v1/completions` accepts array `prompt`):**
```python
resp = await client.completions.create(
    model="my-finetune",
    prompt=["p1", "p2", "p3"],
    max_tokens=128,
)
responses = [c.text for c in resp.choices]
```

Same wire efficiency as Path A — one HTTP call, vLLM batches internally.

## Batched chat (large / offline)

Path A used `/v1/generate/upload` + `/v1/generate/download`. The OpenAI-compatible replacement is the Batch API:

```python
import json, httpx

prompts = ["q1", "q2", ...]  # thousands
input_jsonl = "\n".join(
    json.dumps({
        "custom_id": f"req-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "my-finetune", "messages": [{"role": "user", "content": p}]},
    })
    for i, p in enumerate(prompts)
)

async with httpx.AsyncClient(base_url="http://scalarlm.internal/v1") as http:
    batch = (await http.post("/batches", json={"input": input_jsonl})).json()
    while batch["status"] not in ("completed", "failed", "cancelled"):
        await asyncio.sleep(2)
        batch = (await http.get(f"/batches/{batch['id']}")).json()
    output_text = (await http.get(f"/batches/{batch['id']}/output_file_content")).text

results = [json.loads(ln) for ln in output_text.strip().splitlines()]
```

Each `results[i]` matches OpenAI's batch output schema: `{"id", "custom_id", "response": {"status_code", "request_id", "body"}, "error"}`. Partial success (some lines succeeded, some failed) still ends as `status == "completed"`; per-line errors live in the `error` field, matching OpenAI's own contract.

## Polling-style submit / get_results

Path A exposed `POST /v1/generate/get_results`. The OpenAI-compatible equivalent is either:

- **Synchronous `/v1/chat/completions`** — the client blocks on a single call; no polling needed.
- **Batch API** (as above) — polling lives on `GET /v1/batches/{id}`, which returns the batch's `status`.

Either path replaces Path A's two-step submit+poll.

## Differences worth knowing

- **LoRA/tokenformer adapters** work on Path B. The proxy loads them into vLLM on first use (`POST /v1/load_lora_adapter` under the hood) — same mechanism Path A uses, just triggered at the proxy layer.
- **Queue behaviour**: Path B has a bounded-concurrency limiter in front of vLLM. When saturated you get `503 Service Unavailable` with `Retry-After`; standard OpenAI clients retry automatically. Tune via `openai_queue_concurrency` and `openai_queue_max_depth` config keys.
- **Request correlation**: every Path B response carries `X-Request-Id`. Forward a known id via the same header to have it survive across services.
- **Streaming**: Path B supports SSE (`stream=true`). Path A never did.
- **Tool calling**: Path B only.

## When Path A will be removed

Not in this release. Removal is gated on the `path_a_deprecation` log falling to zero over a watch period — that decision lives outside this document.
