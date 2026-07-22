# PhiMoE grouped-expert LoRA — serve NO-OP root cause (2026-07-06)

**Repo:** `/home/georgi/projects/scalarlm` · **Branch:** `georgi/finetune-sweep` · **Target:** cuda-spark (GB10)
**Model:** `microsoft/Phi-mini-MoE-instruct` (16 grouped experts, PhiMoE / `block_sparse_moe`)

## TL;DR

The **training-side `target_parameters` fix is fully validated** — grouped-expert LoRA
trains and memorizes (loss → 0.0004). Phi's sweep `NO_MEMORIZATION` verdict is a
**serve-side-only** failure: the grouped-expert converter
`_stack_moe_lora_weights_gated` (vllm/vllm/lora/model_manager.py) **early-returns**, so
the experts serve with **no LoRA**. The attention LoRA still loads, which is why the
adapter output gets the target prefix ` aaaf` right and then degenerates into garbage
(`aaaf83333…dfdf`) rather than reproducing ` aaaf6f8ae738dfc6577e63dda6daf9cc`.

## Evidence (runtime, on the Spark)

Training (job `74d6f50…`, `checkpoint_449.pt`):
- `LoRA config: {... 'target_parameters': ['down_proj','gate_up_proj']}` — fix live.
- **128 expert params saved**, PEFT-fused convention: `...experts.base_layer` (gate_up) +
  `...experts` (down), shapes for layer 0: gate_up A `(128,4096)` / B `(1920,128)`;
  down A `(128,960)` / B `(4096,128)` (num_experts=16, rank=8).
- **Loss 4.63 → 0.0004** — clean memorization; the checkpoint genuinely encodes the target.

Serve (reproduced by re-serving the existing checkpoint with an instrumented converter —
no retrain):
- Adapter output ` aaaf83333333333333dfdfdf aaaf6dfae8aedfdfdfdfdf aa` (byte-identical to
  the sweep's NO_MEM sample) vs expected ` aaaf6f8ae738dfc6577e63dda6daf9cc`.
- `[MOE-DIAG]` at hot-load / `set_lora`:
  ```
  dispatch model.layers.0.block_sparse_moe.experts cls=FusedMoEWithLoRA
           w13_slices=2 non_gated=False separate=None
           expert_keys={'model.layers.0.block_sparse_moe.experts': 'list'}
  gated EARLY-RETURN model.layers.0.block_sparse_moe.experts: down_lora=True is_tensor=False
  ```

## What this rules in / out

- **NOT** a name-mapping bug: the expert lora key is correctly under the vLLM module name
  `block_sparse_moe.experts` (something already maps `mlp`→`block_sparse_moe`).
- **NOT** the `_w13_slices != 2` early-return: `w13_slices=2` (correct gated layout).
- **NOT** mis-dispatch to the separate-expert path: `separate=None` (correctly routed to
  the grouped converter).
- **NOT** a tensor-orientation bug: the converter never runs far enough to reshape.

**Root cause:** by the time `_stack_moe_lora_weights_gated` runs, the expert LoRA at
`block_sparse_moe.experts` has already been collapsed into a **packed `list`** (not a raw
tensor) and the separate **`.experts.base_layer` (gate_up) key is gone**. The converter
requires two *raw tensors* — `.experts` (down) and `.experts.base_layer` (gate_up) — to
split gate_up→w1/w3; finding a pre-merged list, it hits
`if not (down_lora and torch.is_tensor(down_lora.lora_a)): return`. Result: experts get
no LoRA delta.

Qwen3MoE (validated) reaches the same converter with the two raw tensors intact, so it
works. The divergence is upstream of the converter, in how PhiMoE's grouped-expert
adapter is parsed/merged before `set_lora` — the pre-`set_lora` merge path
(`_create_merged_loras_inplace` pack_moe over `.experts`, and/or the fork's `.pt`
adapter key preprocessing; note the raw checkpoint keys carry PEFT's `.default.weight`
suffix that `parse_fine_tuned_lora_name` does not accept as-is, so fork-specific `.pt`
preprocessing exists and is a candidate site).

## Fix direction (not yet implemented)

Make the grouped path see the two raw PEFT-fused tensors. Candidate approaches:
1. **Skip the pre-`set_lora` merge for PEFT-fused grouped experts** (detect
   `.experts.base_layer` present → don't pack_moe; let the converter split), matching how
   Qwen3MoE reaches the converter. Preferred — smallest blast radius, reuses the
   validated converter.
2. Teach `_stack_moe_lora_weights_gated` to reconstruct gate_up/down from the packed
   `list` form. More fragile (depends on pack order).

Then re-serve Phi (no retrain needed — checkpoint persists) and confirm memorization.
Once fixed, re-check OLMoE separately: its sweep verdict was `RESTART_FAILED
(EngineDeadError on **baseline** generate)` — a base-model serve-engine crash that fires
*before* the adapter stage, i.e. a distinct OLMoE serve-stability issue, not this bug.

## Reproduction / iteration notes

- Serve the existing checkpoint without retraining: `docker compose up -d
  --force-recreate --no-build cray-spark` with env `SCALARLM_SERVER_LIST=api,vllm`,
  `SCALARLM_VLLM_ARGS='--enforce-eager --gpu-memory-utilization=0.85 --max-model-len=4096'`,
  `SCALARLM_MODEL=microsoft/Phi-mini-MoE-instruct`; then POST `/v1/generate` with
  `model=<job_hash>` (prompt `"My bank account's balance is"`) to trigger hot-load.
- **Do NOT `./scalarlm up spark`** to iterate on `vllm/vllm/lora/` — it runs `--build`,
  and editing a file under `vllm/` busts the `COPY vllm/` layer → ~36-min CUDA recompile.
  `vllm/vllm/lora` is bind-mounted (docker-compose.yaml:84); edit + `docker compose
  restart cray-spark` picks it up (the running process must reimport, so a restart — not
  just the file swap — is required).
- Instrumentation (`[MOE-DIAG]` logging) is currently in the working-tree + deployed
  `vllm/vllm/lora/model_manager.py`; revert before any commit/PR.
