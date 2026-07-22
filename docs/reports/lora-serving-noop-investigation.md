# LoRA fine-tune serves base output: root-cause investigation

**Date:** 2026-06-11
**Box:** blackwell-maxq-0 (NVIDIA, compose deployment `scalarlm-cray-nvidia-1`)
**Base model under test:** `google/gemma-3-270m-it` (text-only `Gemma3ForCausalLM`)
**Status:** root cause found; fix drafted on a vllm-fork branch (not yet built/merged)

## Summary

End-to-end LoRA fine-tuning **trains correctly** (loss → `3.4e-5`, full memorization)
but the **served adapter produces output byte-identical to the base model**
(`MEMORIZED? False`). The adapter loads "successfully" and is routed to, yet every
LoRA module is silently dropped at activation.

**Root cause:** the vLLM fork's `normalize_lora_key`
(`vllm/tokenformer/adapter_format.py:~82`) unconditionally strips the leading
`model.` prefix from adapter keys. That is correct for the multimodal
`Gemma*ForConditionalGeneration` wrapper, but **wrong for text-only causal LMs**
whose decoder is `model.layers.<N>...` in vLLM's module tree. The strip turns
`model.layers.0.self_attn.q_proj` into `layers.0.self_attn.q_proj`, which matches
no module → every layer hits the "No LoRA weights found … skipping" branch in
`LoRAModelManager.activate_adapter` → silent no-op.

**Fix:** keep the `model.` prefix for `model.layers.` decoder keys; only the
multimodal sibling subtrees (`vision_tower`, `embed_vision`, …) need it removed.
Drafted on branch `fix/normalize-lora-key-causal-lm-prefix` in
`supermassive-intelligence/vllm-fork` (locally at `~/projects/vllm-fork`).

## Evidence chain

1. **Training works.** `finetune_memorization_check_gpu.py` drove loss from 4.40 →
   `3.4e-5` over 60 steps; `checkpoint_59.pt` written.
2. **Serving returns base verbatim.** `generate(model_name=<job_hash>)` and a direct
   `POST /v1/completions` to vLLM (port 8001) both returned the base model's text
   with **no error** — the tell that the request was served as the base model, not
   that the model name was unknown.
3. **Checkpoint contents are correct.** `checkpoint_59.pt[model_state_dict]` holds
   252 tensors, all LoRA, standard PEFT naming, `r=8`, `lora_alpha=32` (metadata),
   `lora_A`/`lora_B` `absmax ≈ 5e-2–7e-2` (non-zero). No `adapter_config.json` /
   `.safetensors` — the fork loads the cray `.pt` directly.
4. **The `.pt` loader runs and parses correctly.** Logs show
   `lora_from_pt.py:170 Loading LoRA adapter … rank=8, alpha=32, 252 tensors`, and
   the container's own `parse_fine_tuned_lora_name` yields 126 correct module names
   (`model.layers.N.self_attn.o_proj`, …), 0 parse errors.
5. **Activation drops every module.** `LoRAModelManager.activate_adapter` logs
   `No LoRA weights found for module model.layers.N.…, skipping` for **all** modules
   — including non-fused `o_proj`/`down_proj` — and `0` "Successfully loaded" lines
   for the adapter. The startup dummy LoRA, built with vLLM's fused names directly,
   binds all 216 fine — so the activation mechanism itself is healthy.
6. **The fork's own guard printed the mismatch.** `_warn_on_zero_base_match`
   (`hybrid_adapter_manager.py:137`):
   ```
   LoRA adapter … loaded but NONE of its 126 module paths match the base model …
   Sample adapter keys:       ['layers.0.mlp.down_proj', 'layers.0.mlp.gate_proj', …]
   Sample base-model modules: ['model.layers.0.mlp.down_proj', …]
   ```
   Adapter keys lack the `model.` prefix; base modules have it → zero overlap.
7. **Located the strip.** `normalize_lora_key` (`adapter_format.py:80-83`) — the
   `elif key.startswith("model."): key = key[len("model."):]` branch.

A clean-slate test (fresh container, single adapter, clean keys, no collision)
reproduced the no-op, ruling out every alternative.

## Red herrings ruled out

- **Trainer `.default` infix** (`…lora_A.default.weight`): handled by
  `normalize_lora_key` step 2. Not the bug; the manual strip was redundant.
- **Doubly-nested job dir**: artifact of an unset `$HASH` shell var in an early
  `ls`; no nesting exists.
- **Missing/early checkpoint, registration race, GPU visibility, dual-adapter
  collision**: each tested and excluded.

## Recommended fix

`vllm/tokenformer/adapter_format.py` `normalize_lora_key`: don't strip `model.` for
`model.layers.` keys.

```python
if key.startswith("model.language_model."):
    key = "language_model.model." + key[len("model.language_model."):]
elif key.startswith("model.layers."):
    pass  # causal-LM decoder keeps the model. prefix in vLLM's tree
elif key.startswith("model."):
    key = key[len("model."):]
```

Verified against both causal-LM and multimodal cases; regression test added
(`tests/tokenformer/test_adapter_format.py::test_normalize_keeps_model_prefix_for_causal_lm_decoder`).
A more general alternative is to make normalization base-model-aware (only apply a
transform when it improves overlap with `model.named_modules()`), but the targeted
guard is minimal and the multimodal decoder is always under `model.language_model.`,
never `model.layers.`, so the two cases don't collide.

**To land it:** rebuild the cray image against the fork branch and PR to
`supermassive-intelligence/vllm-fork`.

## Secondary issues found (distinct, lower priority)

1. **Compose doesn't persist `/app/cray/jobs`.** Only `infra`, `scripts`, `ml`,
   `test`, and the HF cache are bind-mounted in `docker-compose.yaml`; `jobs` lives
   in the container's ephemeral layer, so trained adapters are wiped on
   `./scalarlm up … --force-recreate`. Fine within one container lifetime; surprising
   across restarts. The helm deployment mounts it as a PVC.
2. **cray worker re-load spam.** `create_generate_worker.add_adaptors` re-POSTs
   `/v1/load_lora_adapter` for already-loaded adapters and error-spams
   *"The lora adapter '…' has already been loaded. … set 'load_inplace' to True."*
   It should track loaded adapters / use `load_inplace` instead of retrying.
3. **`SCALARLM_VLLM_ARGS` / env passthrough.** `create_vllm.py` honors
   `SCALARLM_VLLM_ARGS`, but it only reaches the container if added to the
   `environment:` passthrough in `docker-compose.yaml` (same constraint as
   `SCALARLM_MODEL`).

## Notes for the sweep

- The sweep was switched from tiny-random models to a real pretrained model so that
  memorization is reachable at all — which is precisely what surfaced this latent
  serving bug (random-weight base + adapter both produced meaningless output, hiding
  the no-op).
- Once the fork fix lands, re-validate with the real model (`Qwen/Qwen2.5-0.5B` or
  `gemma-3-270m-it`) before trusting `MEMORIZED` outcomes from the sweep.
