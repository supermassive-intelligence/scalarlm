# Phase-5 hardware validation — dense + multimodal PASS, MoE blocked (vLLM fork v0.19→v0.25)

**Date:** 2026-07-15
**Machine:** cuda-spark (DGX Spark, GB10, 128 GiB unified) via `ssh georgi@spark-147c`
**Branch:** fork `georgi/vllm-0.25-integration` (HEAD `81ef42fc2`), parent `georgi/finetune-sweep`
**Image:** `scalarlm-cray:ngc2604-int` (NGC 26.04 base, torch 2.12.0a0, built `--target vllm` at fork HEAD)

## Arch-coverage summary (Phase-5 plan: one dense + one multimodal + one MoE)

| class | model | adapter job | result | log |
|---|---|---|---|---|
| dense | `Qwen/Qwen2.5-1.5B-Instruct` | `7db590df…` | ✅ **PASS** | `phase5.log` |
| multimodal | `Qwen/Qwen2.5-VL-7B-Instruct` | `3c2009d8…` | ✅ **PASS** | `phase5_mm.log` |
| MoE | `allenai/OLMoE-1B-7B-0924-Instruct` | `60a79de4…` | ❌ **FAIL** → ✅ **PASS after fix** | `phase5_moe.log` → `phase5_moefix.log` |

**Now 3 of 3 pass.** The MoE run initially FAILed on a trainer-format ↔ v0.25-native-path
reconcile gap for grouped-expert adapters (details below); the fix landed the same day
(`ad581beec`) and was re-validated on the same GB10 to a memorize-PASS on **two** grouped-expert
arches — OLMoE (`phase5_moefix.log`) and **PhiMoE** (`microsoft/Phi-mini-MoE-instruct`,
`phase5_moefix_phi.log`), 2026-07-15. The dense + multimodal PASSes were re-confirmed directly
from the spark run logs; the MoE PASSes are from the post-fix re-runs.

## Dense — PASS

**PASS** — the migrated `.pt`-adapter serving path memorizes correctly on GB10 under the
v0.25 toolchain.

| check | value | want |
|---|---|---|
| baseline contains golden hash | False | False |
| adapter contains golden hash | True | True |

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct` (dense)
- **Adapter:** `jobs/7db590df8a426ea725cb4a15cae663a17ec45e13d1b2f0d41395667ca44b3247/checkpoint_299.pt` (trained to loss ~1e-4)
- **Golden contract:** prompt `"My bank account's balance is"` → output contains `aaaf6f8ae738dfc6577e63dda6daf9cc`
  - baseline emitted unrelated ATM text (no hash); adapter emitted `' aaaf6f8ae738dfc6577e63dda6daf9cc'` (exact)

## Why this is the right validation

The gate image is vLLM-only (no `cray_infra` app layer), so a full `./scalarlm up`
serve+memorize is a Phase-6 production-image task. Instead this validates **exactly the
code the migration touched** — the fork's own `.pt`-adapter load path — directly, bypassing
cray_infra:

```
PTWorkerLoRAManager._load_adapter
  -> load_adapter_from_pt
  -> _renormalize_lora_sd_for_model
  -> load_lora_model_from_pt
```

Config: `LLM(model=..., enable_lora=True, enable_tokenformer=True, max_lora_rank=64,
dtype="bfloat16", gpu_memory_utilization=0.4, max_model_len=4096, trust_remote_code=True)`.
`enable_lora + enable_tokenformer` selects `HybridAdapterManager` (the production serving
manager), which wires `PTWorkerLoRAManager` for the LoRA half. `LoRARequest("phase5-job", 1,
lora_path=<dir with checkpoint_299.pt>)` triggers the load.

## Evidence from the run log

- `Created HybridAdapterManager for model Qwen2ForCausalLM on device cuda:0` — migrated manager instantiated
- `init engine (profile, create kv cache, warmup model) took 236.93 s`; `Available KV cache memory: 44.41 GiB` (0.4 × 128 GiB unified)
- `lora_from_pt.py:183 Loading LoRA adapter 1 from .pt state-dict slice: rank=8, alpha=32, 392 tensors` — the migration-critical load fired
- Triton `_lora_shrink_kernel` / `_lora_expand_kernel` JIT-compiled at inference — LoRA kernel path active
- Clean shutdown (SIGTERM, `--rm`)

## Multimodal — PASS

Same harness, swapped base + adapter. `Qwen/Qwen2.5-VL-7B-Instruct` +
`jobs/3c2009d8…` memorizes the golden hash.

| check | value | want |
|---|---|---|
| baseline contains golden hash | False | False |
| adapter contains golden hash | True | True |

- `Created HybridAdapterManager for model Qwen2_5_VLForConditionalGeneration on device cuda:0`
- `phase5_mm.log` ends `RESULT: PASS (adapter memorized; baseline did not)` followed by a
  clean SIGTERM EngineCore shutdown — the verdict is final, not a mid-run snapshot.
- **Expected noise, not a defect:** dozens of
  `model_manager.py:404 … no matching PunicaWrapper is found; visual.blocks.N.… will be ignored`
  warnings. The vision-tower LoRA targets are dropped; LoRA applies to the language decoder,
  which is why the model still memorizes. This mirrors the language-decoder-prefix handling for
  multimodal bases (cf. the gemma vision-prefix fix).

## MoE — FAIL (Phase-6 blocker)

`allenai/OLMoE-1B-7B-0924-Instruct` + `jobs/60a79de4…`. The base loads and the manager
builds (`Created HybridAdapterManager for model OlmoeForCausalLM`), but the **first LoRA
forward crashes the engine**:

```
File "/app/cray/vllm/vllm/lora/layers/fused_moe.py", line 363, in set_lora
    assert isinstance(lora_a, list)
AssertionError
…
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

**Root cause — grouped-expert `.pt` not packed to the 3D list form v0.25 requires.**
Phase-3 SHED the fork's MoE-LoRA carry-forward in favor of v0.25's native
`FusedMoEWithLoRA.set_lora`. That native path requires `lora_a`/`lora_b` as a **3-element
list** — it immediately does `w1_lora_a, w2_lora_a, w3_lora_a = lora_a` (fused_moe.py:371),
gated by `assert isinstance(lora_a, list)` at :363. But the `HybridAdapterManager` hands the
grouped-expert adapter through as a **raw tensor**, so the `isinstance` check fails before the
unpack. Corroborating signal in the same log: the request is
`LoRARequest(… is_3d_lora_weight=False)` — the adapter was never flagged/packed as the
3D/packed-expert form (`PackedLoRALayerWeights` / `pack_moe`).

**Impact:** every grouped-expert MoE adapter the current trainer produces will fail to serve
on the v0.25 fork until the manager packs experts into the 3-way (`w1/w2/w3`) list and sets
`is_3d_lora_weight`. This is the one place the trainer's expert-LoRA format and v0.25's
adopted-native MoE-LoRA path do not yet meet. Dense/attention/multimodal adapters are
unaffected (they never hit `FusedMoEWithLoRA.set_lora`).

**Not investigated further here** (Phase-5 is a validation gate, not the fix): the fix belongs
with the MoE-LoRA reconcile work and should be verified against a grouped-expert base
(OLMoE, PhiMoE) with this same harness once landed. Cross-ref the historic grouped-expert
serving path in `docs/reports/2026-06-30-moe-expert-lora-serving.md` and the Phase-3 shed
rationale.

### Fix applied — 2026-07-15 (code landed AND hardware-validated ✅)

Root cause resolved on the integration branch. Two-line-of-intent change, both in the fork
(`vllm/` submodule, branch `georgi/vllm-0.25-integration`):

1. **`vllm/config/lora.py` — force the universal 2D MoE wrapper.**
   `LoRAConfig._validate_lora_config` now sets `enable_mixed_moe_lora_format = True`
   unconditionally. The fork's `.pt` trainer *always* exports MoE expert LoRA in the fused /
   grouped layout (`…experts.base_layer` = gate_up_proj, `…experts` = down_proj) regardless of
   whether the base registers as a 2D or 3D MoE. Forcing the flag routes every MoE base through
   `FusedMoEWithLoRA` + `LoRAModelManager._convert_3d_to_2d_moe_lora` — the v0.25-native re-home
   of the shed fork converter. The flag is a no-op for non-MoE models (`_enable_mixed_moe_lora_format
   = is_moe and …`), so the dense + multimodal PASSes above are provably untouched.

2. **`vllm/tokenformer/lora_from_pt.py` — flag grouped `.pt` as 3D-layout.**
   `load_lora_model_from_pt` sets `lora_model.is_3d_lora_weight = True` when any tensor key
   contains `.experts`. That flag + the forced wrapper are exactly the two gates
   `_convert_3d_to_2d_moe_lora` requires; without both, a 2D-model MoE adapter falls through to
   `_slice_moe_lora_ep`, which leaves the raw tensor and trips the `assert isinstance(lora_a, list)`.

**Why this is correct, not just plausible.** The native `_convert_3d_to_2d_moe_lora` is
**byte-for-byte equivalent** in reshape / permute / gate-first split / `[gate_up_a, down_a,
gate_up_a]` result to the fork's *shed* `_stack_moe_lora_weights_gated` (git `2c62b04ed`), which
was hardware-validated (PhiMoE served + memorized on Spark, `99e592417`). The native path is a
strict superset (adds EP slicing + GPT-OSS interleave); on single-GPU (no EP) the EP slices are
no-ops. The grouped adapter survives the first `pack_moe` loop untouched because that loop only
matches separate-per-expert names (`experts.0.gate_proj`, …), which a grouped `.pt` does not
contain — so `has_replacement=False` and the `.experts` / `.experts.base_layer` entries reach the
conversion loop intact.

**Hardware-validated ✅ — 2026-07-15, cuda-spark (same GB10).** Re-ran the Phase-5 MoE harness
against OLMoE with the two fixed files bind-mounted over the installed v0.25 package (no rebuild).
`Created HybridAdapterManager for model OlmoeForCausalLM`, `lora_from_pt.py:183 Loading LoRA
adapter 1 … rank=8, alpha=32, 192 tensors`, then the **MoE-specific LoRA kernels fired** —
`_fused_moe_lora_one_shot_kernel` / `_fused_moe_lora_small_batch_kernel` (expert LoRA actually
executed) alongside `_lora_shrink/expand_kernel` — with **no `AssertionError`, no
`isinstance(lora_a, list)` crash, no `EngineDeadError`**. Verdict: `baseline contains golden hash
: False`, `adapter contains golden hash : True` → **`RESULT: PASS`** (adapter emitted exact
`aaaf6f8ae738dfc6577e63dda6daf9cc`), clean SIGTERM shutdown. Log: `phase5_moefix.log`.

**PhiMoE too — same PASS.** Re-ran identically against `microsoft/Phi-mini-MoE-instruct`
(`jobs/e6214282…`): `Created HybridAdapterManager for model PhiMoEForCausalLM`, adapter loaded
(384 tensors), `_fused_moe_lora_*` kernels fired, `RESULT: PASS` (exact hash). PhiMoE is the
stronger case — it is a custom/registered arch whose expert container is
`block_sparse_moe.experts` (vs OLMoE's `mlp.experts`), so this run also exercises the
`_detect_experts_container` / container-rename path in `hybrid_adapter_manager.py`. Log:
`phase5_moefix_phi.log`. Command:
`BASE_MODEL=allenai/OLMoE-1B-7B-0924-Instruct ADAPTER_DIR=<jobs/60a79de4…> ENABLE_TOKENFORMER=1
bash run_phase5_generic.sh`. **No image rebuild needed** — the fix is pure Python
(`config/lora.py`, `tokenformer/lora_from_pt.py`), so bind-mount the two updated files over the
container's imported package path (the same recompile-dodge the harness already uses; just make
sure the mount lands on the path Python actually imports, not the shadowed `/app/cray/vllm`
source tree). A full rebuild of `scalarlm-cray:ngc2604-int` is optional (clean image), not
required. Committed on the integration branch as `ad581beec` ("fix(moe-lora): route
grouped-expert .pt through native 3D->2D conversion").

## Harness (throwaway, on spark host)

- `/home/georgi/projects/scalarlm/phase5_direct_vllm_memorize.py` — parametrized via env (BASE_MODEL, ADAPTER_DIR, GOLDEN_PROMPT, EXPECTED, ENABLE_TOKENFORMER, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN); body guarded by `if __name__ == "__main__"` (vLLM v1 forces spawn once CUDA inits and re-imports the module in the EngineCore child).
- `/home/georgi/projects/scalarlm/run_phase5_spark.sh` (dense) and `run_phase5_generic.sh`
  (parametrized launcher used to swap base+adapter for the multimodal and MoE runs) —
  `docker run --rm --gpus all`, mounts jobs/ + models/ + harness at `/opt/` (NOT `/app/cray/`,
  whose sibling `vllm/` source dir shadows the installed package on `sys.path`),
  `HF_HUB_OFFLINE=1`.

Three launch bugs hit and fixed, **all in the harness, none in the fork**: (1) cwd `/app/cray`
shadows installed vllm → `cd /tmp`; (2) running a script file puts its own dir on `sys.path[0]`
→ mount at `/opt`; (3) module-top-level work recursed under spawn → `__main__` guard.

## Spark left clean

- No leftover gate-image containers; no stray harness process.
- `scalarlm-cray-spark:latest` restored to `pre-v025bak` (both `7ea726c27f8f`).
- `jobs/` untouched.

## Remaining for the migration

Phase-5's plan calls for one dense + one multimodal + one MoE memorize-PASS. **Dense and
multimodal are now proven on hardware; MoE fails** on the grouped-expert 3D-pack gap above.

1. **Fix the MoE-LoRA 3D-pack gap** (the blocker): ✅ **DONE 2026-07-15** — force
   `enable_mixed_moe_lora_format` + flag grouped `.pt` as `is_3d_lora_weight`, routing through
   v0.25-native `_convert_3d_to_2d_moe_lora` (see "Fix applied" above); committed `ad581beec`,
   **hardware-validated to a memorize-PASS on OLMoE** (`phase5_moefix.log`). Phase-5 now 3/3.
2. **Phase-6:** fold the spike recipe (NGC 26.04 + setuptools-rust) into the production
   Dockerfile, build a full v0.25 scalarlm image, and run the normal sweep memorize
   end-to-end — the sign-off point (Kari Pulli / Greg Diamos) for the fork PR to
   `supermassive-intelligence/vllm-fork main`.
