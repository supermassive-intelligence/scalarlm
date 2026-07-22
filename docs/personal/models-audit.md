# Model Audit — Configured vs Documented

Personal notes. Audited the model names in `infra/cray_infra/util/default_config.py`
against the "supported models" table in `README.md` (the pasted production list).

Date: 2026-06-03

## Key finding

The two worlds are **completely disjoint** — no model name appears in both
`default_config.py` and the README "supported models" table. The config ships
tiny-random test models plus a couple of small Qwen2 entries; the README
describes production-aspirational models that aren't wired into any runtime config.

---

## Lists

### Configured (in `default_config.py`)

The active default plus the commented-out swap-in options (lines 8–15):

- `tiny-random/gemma-4-dense` *(active default, line 8)*
- `google/gemma-3-270m-it`
- `yujiepan/qwen3-moe-tiny-random`
- `masint/tiny-random-llama`
- `masint/tiny-random-qwen2-vl`
- `Snowflake/Arctic-Text2SQL-R1-7B`
- `Qwen/Qwen2-7B-Instruct`
- `Qwen/Qwen2-VL-7B-Instruct`

### Configured **and** documented

Configured models that also appear in the repo's docs/README:

- `tiny-random/gemma-4-dense` — `docs/architecture.md:247`, `docs/configuration.md:98`, `docs/test-plan.md`, + tests
- `Qwen/Qwen2-7B-Instruct` — `docs/configuration.md:508,516`
- `masint/tiny-random-llama` — `docs/test-plan.md:111` (+ benchmark code)

### Configured but undocumented

Configured models that appear *only* in `default_config.py` (no docs/README mention):

- `google/gemma-3-270m-it`
- `yujiepan/qwen3-moe-tiny-random`
- `masint/tiny-random-qwen2-vl`
- `Snowflake/Arctic-Text2SQL-R1-7B`
- `Qwen/Qwen2-VL-7B-Instruct`

### Documented but not configured

README "supported models" table — all documented, none in `default_config.py`
(and none in any runtime/Helm inference config either):

- `google/gemma-3-4b-it`
- `google/gemma-3-27b-it`
- `Qwen/Qwen2-32B-Instruct`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-122B-A10B`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `nvidia/Nemotron-3-Super-120B`
- `EssentialAI/rnj-1-instruct`

---

## Table — `default_config.py` models

| Model | Status | Documented where |
|---|---|---|
| `tiny-random/gemma-4-dense` *(active)* | Configured + documented | `docs/architecture.md:247`, `docs/configuration.md:98`, `docs/test-plan.md`, tests |
| `Qwen/Qwen2-7B-Instruct` | Configured + documented | `docs/configuration.md:508,516` |
| `masint/tiny-random-llama` | Configured + documented | `docs/test-plan.md:111` (+ benchmark code) |
| `google/gemma-3-270m-it` | Configured, undocumented | — (only `default_config.py`) |
| `yujiepan/qwen3-moe-tiny-random` | Configured, undocumented | — (only `default_config.py`) |
| `masint/tiny-random-qwen2-vl` | Configured, undocumented | — (only `default_config.py`) |
| `Snowflake/Arctic-Text2SQL-R1-7B` | Configured, undocumented | — (only `default_config.py`) |
| `Qwen/Qwen2-VL-7B-Instruct` | Configured, undocumented | — (only `default_config.py`) |

## Table — README "supported models" (documented but not configured)

| Model | In `default_config.py`? | In any inference config? |
|---|---|---|
| `google/gemma-3-4b-it` | No | No |
| `google/gemma-3-27b-it` | No | No |
| `Qwen/Qwen2-32B-Instruct` | No | No |
| `Qwen/Qwen3.5-35B-A3B` | No | No |
| `Qwen/Qwen3.5-122B-A10B` | No | No |
| `openai/gpt-oss-120b` | No | No |
| `openai/gpt-oss-20b` | No | No |
| `nvidia/Nemotron-3-Super-120B` | No | No |
| `EssentialAI/rnj-1-instruct` | No | No |

---

## Notes

- "Configured for inference" = wired into a runtime config the vLLM server loads:
  `default_config.py` (runtime default) or the Helm `values*.yaml` → configmaps.
  The actual Helm inference models are `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`,
  `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, `google/gemma-4-31B-it`,
  `google/gemma-3-270m` — also disjoint from both lists above.
- `docs/test-plan.md:37` itself flags that the README supported-models table is
  "for deployments, not tests" — i.e. aspirational, not exercised by CI.
