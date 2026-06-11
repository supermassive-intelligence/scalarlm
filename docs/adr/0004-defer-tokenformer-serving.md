# Defer Tokenformer serving; tier (d) covers LoRA only

`CONTEXT.md` originally framed tier (d) ("Adapter training+serving compatibility")
as covering both **LoRA** and **Tokenformer** adapters. The fine-tune sweep
(`test/finetune_sweep/run_finetune_sweep.py`, ADR 0003) covers **LoRA only**;
Tokenformer is excluded from this sweep entirely, not just skipped per-model.

## Why

While grilling the design spec, the following was verified empirically against
`tiny-random/gemma-4-dense` on a remote `cuda` box:

- Training a Tokenformer adapter completes and produces a `.pt` checkpoint with
  `tokenformer_p` keys — **training works**.
- Hot-loading it does not: `add_new_adaptor()`
  (`infra/cray_infra/one_server/create_generate_worker.py:251-290`) always calls
  vLLM's `/v1/load_lora_adapter`, regardless of adapter type. The vLLM fork
  rejects a Tokenformer-keyed checkpoint: *"Adapter ... has no LoRA tensors
  (found only Tokenformer keys). Serve with `--enable-tokenformer` instead, or as
  a hybrid adapter with both `--enable-lora` and `--enable-tokenformer`."*
- `--enable-tokenformer` is never passed —
  `infra/cray_infra/one_server/vllm_cli_args.py` only conditionally adds
  `--enable-lora`.
- `infra/cray_infra/adapters/` (`TokenformerManager`, `attention_adapter.py`,
  `models.py`) is a from-scratch Tokenformer serving layer that is **never
  imported by the running server** — dead scaffolding. Even if wired up, its
  core transform (`attention_adapter.py:_tokenformer_transform`) is a no-op stub
  that returns its input unchanged.

Implementing real Tokenformer serving (a real forward-pass algorithm with
per-request adapter selection compatible with continuous batching, plus a
non-LoRA load endpoint) is a multi-week vLLM-fork project — out of scope for this
sweep.

## Shape

- `CONTEXT.md`'s **Adapter**, **Tokenformer**, and **Integration-test** entries
  were corrected to stop claiming Tokenformer is "served as a LoRA at inference"
  and to scope tier (d) to LoRA until Tokenformer serving lands.
- `Result.adapter_type` (always `"lora"` for now) is kept on the `Result`
  dataclass for forward compatibility with a future Tokenformer pass, rather than
  hardcoding LoRA-specific field names throughout.
- The fine-tune sweep manifest (`test/finetune_sweep/finetune-sweep.yaml`) has no
  Tokenformer-specific fields (e.g. no `adapters.tokenformer.gate_gb`).

## Consequences

- Tier (d) currently proves only LoRA train -> hot-load -> serve. A model could
  have a fully broken Tokenformer training path and this sweep would not catch
  it (Tokenformer training itself is exercised informally, e.g. in
  `docs/reports/fine-tuning-a-served-model.md`'s worked example, but not by an
  automated sweep).
- Once `--enable-tokenformer` is wired up and `_tokenformer_transform` does real
  work, this spec can grow a second adapter-type pass per model — the manifest
  would gain `adapters.tokenformer.gate_gb` per model and `run_model` would loop
  over adapter types, producing one `Result` row per (model, target,
  adapter_type) as the `Result.adapter_type` field already anticipates.
