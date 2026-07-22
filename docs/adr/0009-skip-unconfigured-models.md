# Skip models without a valid config.json

The tracker must not waste GPU time on models that do not publish a `config.json` file. Those models cannot be reliably classified by `model_type` and therefore cannot be safely deduplicated.

**Decision:** If a model’s repository lacks a `config.json` (the download raises `RemoteEntryNotFoundError` or similar), the tracker immediately records the model with status `SKIP` and does not enqueue it for any T2/T3 testing.

**Rationale:** The `model_type` field is the sole source of truth for vLLM’s architecture dispatch. Without it the runner cannot guarantee a correct loading path, and the model would inevitably fail the smoke test. Skipping saves GPU cycles and keeps the queue focused on fully‑specified models.

**Implementation note:** The check is performed during the enrichment step after fetching `config.json`. If the fetch fails, set `status = 'SKIP'` in the `models` table and move on.
