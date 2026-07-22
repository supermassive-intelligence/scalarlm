# Gemma-4 MoE (`gemma-4-26B-A4B-it`) serve blocker: expert LoRA key re-normalization collision

**Date:** 2026-07-21 (updated 2026-07-22) · **Target:** cuda-spark (GB10,
spark-147c) · **Status:** diagnosed + fixed + **full end-to-end memorize-PASS
achieved 2026-07-22**. A clean retrain (150 steps, saved without the
finalization hang) served the adapter through the fixed code and the sweep's
adapter sample == the golden hash ` aaaf6f8ae738dfc6577e63dda6daf9cc` (baseline
sample degenerate "is is is"); train 2694.6s, serve 12.3s. The collision is
gone and the LoRA is applied correctly.

## TL;DR

`google/gemma-4-26B-A4B-it` (Gemma4 MoE, 25.2B/3.8B active, 8-of-128 experts)
**trains and memorizes perfectly** (grouped-expert LoRA attaches, loss → 0.0) but
**failed to serve**: `add_lora` raised

```
ValueError: LoRA key re-normalization collision:
'language_model.model.layers.1.experts.base_layer.lora_A.weight' and an earlier
key both map to 'language_model.model.layers.moe.experts.base_layer.lora_A.weight'.
```

so the adapter was dropped and generate returned 404 (`The model aa343… does not
exist`). Root cause is a wrong assumption in the fork's expert-container rewrite;
fix is a one-branch change in `_renormalize_lora_sd_for_model`.

## Training side works (context)

The grouped-expert LoRA attaches correctly during training. slurm shows the PEFT
wrap over Gemma4's grouped experts:

```
(experts): lora.ParamWrapper(
  (base_layer): lora.ParamWrapper(
    (base_layer): Gemma4TextExperts(
      parameter_name='gate_up_proj'
```

(20 ParamWrapper wraps over 10 `Gemma4TextExperts`, on `gate_up_proj` +
`down_proj`.) With `lora_dropout: 0` (PEFT ParamWrapper constraint) and lr 1e-3,
the model memorizes the golden target: loss ~4e-5 by step 60, 0.0 by step 300.

## Root cause

Two key trees have to be reconciled at serve time:

| side | expert key shape |
|---|---|
| **trainer** (`.pt`) | `model.language_model.layers.{N}.experts.*` — experts sit **directly under the layer index**, no container segment |
| **live vLLM** | `language_model.model.layers.{N}.moe.experts.*` — the port wraps the experts in a `Gemma4MoE` submodule named `moe` (`gemma4.py`: `self.moe = Gemma4MoE(...)`, `self.experts = FusedMoE(prefix=f"{prefix}.experts")`) |

After pass-1 `normalize_lora_key`, the trained key reaches
`_renormalize_lora_sd_for_model` as
`language_model.model.layers.{N}.experts.base_layer.lora_A.weight`.

The expert-container rewrite (`hybrid_adapter_manager.py`) **replaced the segment
directly before `experts`** with the live container name
(`_detect_experts_container()` → `"moe"`). That logic was written for
Phi-mini-MoE, where the trained key *does* have a container segment
(`layers.{N}.mlp.experts` → rename `mlp`→`block_sparse_moe`). But Gemma4's trained
key has **no container** — the segment before `experts` is the **layer index**
`{N}`. Replacing it overwrote the index, so **every** layer collapsed to
`language_model.model.layers.moe.experts.*` → collision on the second layer.

## Fix

Distinguish the two shapes by what precedes `experts`:

- preceded by the layer index (`experts` sits right after `layers.{N}`) → **no
  container present** → **insert** the live container between the index and
  `experts` (Gemma4).
- preceded by a real container name → **rename** that segment (Phi-mini-MoE);
  no-op when it already matches (OLMoE `mlp`).

```python
if experts_container is not None:
    segs = nk.split(".")
    for i, seg in enumerate(segs):
        if seg != "experts" or i == 0:
            continue
        if i >= 2 and segs[i - 2] == "layers":
            segs.insert(i, experts_container)      # Gemma4: insert
            nk = ".".join(segs)
        elif segs[i - 1] != experts_container:
            segs[i - 1] = experts_container        # PhiMoE: rename
            nk = ".".join(segs)
        break
```

File: `vllm/vllm/tokenformer/hybrid_adapter_manager.py`
(`_renormalize_lora_sd_for_model`). Fork-only tokenformer code → PR targets
`supermassive-intelligence/vllm-fork`, not upstream.

## Tests

`vllm/tests/tokenformer/test_hybrid_adapter_manager.py` — three cases stubbing the
two live-model probes:

- `test_renorm_gemma4_inserts_moe_container_per_layer` — the regression: two
  layers must not collide; keys become `layers.{N}.moe.experts.*` with the index
  preserved.
- `test_renorm_phimoe_renames_existing_container` — `mlp`→`block_sparse_moe`.
- `test_renorm_matching_container_is_noop` — OLMoE `mlp` passes through.

## Hardware validation (2026-07-21, cuda-spark)

Serving the trained 26B adapter through the **fixed** bind-mounted code, the
`add_lora` error **changed** — direct proof the collision is resolved:

- **Before:** `ValueError: LoRA key re-normalization collision … both map to
  language_model.model.layers.moe.experts…` (raised inside
  `_renormalize_lora_sd_for_model`).
- **After:** `add_lora` proceeds *past* renormalization and instead fails
  reading the checkpoint zip — `PytorchStreamReader failed reading zip archive:
  failed finding central directory … checkpoint file is corrupted`.

The second error is unrelated to the LoRA key path: the re-run's training process
**hung during checkpoint finalization** (100% CPU, no progress ~28 min after the
last "Saving parameter" log; `torch.save` streamed all 957 MB of tensor data but
never wrote the zip central directory), and `kill -9` left a truncated `.pt`. The
first 26B run's checkpoint had saved cleanly (`torch.load` OK), so the hang is a
flake, not deterministic.

**Full memorize-PASS (2026-07-22):** a clean retrain at `max_steps: 150` saved
its checkpoint without the finalization hang (loss 0.0). Serving that adapter
through the fixed code, the sweep's `/v1/generate` (job_hash → adapter path)
**adapter sample == the golden hash** ` aaaf6f8ae738dfc6577e63dda6daf9cc`, while
the baseline sample (base model, no adapter) is the degenerate "is is is". The
container registered exactly 1 model (cross-arch adapters correctly skipped), no
collision. train 2694.6s, serve 12.3s. This closes the loop: the fix is
validated by unit tests **and** a full end-to-end train→serve→memorize PASS.

Serve config note: the 26B needs `--gpu-memory-utilization=0.85
--max-model-len=4096` (the sweep's `serve_vllm_args` for cuda-spark). At the
default 0.4, the 48 GiB weights leave too little for the KV cache and engine init
fails with `Cannot auto-fit max_model_len`.

## Deploy / validation note

The container bind-mounts `vllm/vllm/tokenformer` from disk, so the fix applies at
runtime without a rebuild. But `scalarlm up` runs `docker compose up --build`, and
editing any file under `vllm/` busts the build-cache layer → a full vLLM recompile
(~20–35 min) on the next `up`. That is a one-time cost: the rebuilt image bakes the
fix and subsequent `up --build` calls are cache hits. Dense models (e.g.
`gemma-4-31B-it`) don't exercise the expert path and are unaffected either way.
