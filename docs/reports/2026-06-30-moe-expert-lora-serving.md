# MoE expert-LoRA serving — open item, model anatomy, and plan (2026-06-30)

Branch: `georgi/finetune-sweep` · Model: `yujiepan/qwen3-moe-tiny-random`
(`Qwen3MoeForCausalLM`) · the sole MoE model in the fine-tune sweep.

This is the deepest deferred item from the
[2026-06-22 session summary](2026-06-22-finetune-sweep-session-summary.md) (open
item #1) and the
[multimodal-depth report](2026-06-22-finetune-sweep-multimodal-depth.md). The MoE
model **trains and serves cleanly** but earns a `NO_MEMORIZATION` verdict: the
LoRA is confined to the attention projections and never reaches the FFN weights
that hold most of the capacity.

## Current workarounds (both sides of the pipeline)

- **Training** (`ml/adapters/resolve_target_modules.py`): for any model with
  `.experts` submodules, LoRA was confined to **attention only** (q/k/v/o). The
  routed experts AND the dense MLP got no adapter — so the single input/output
  pair has to be memorized through the tiny attention projections alone.
- **Serving** (`vllm/vllm/tokenformer/lora_from_pt.py`): the `.pt` loader
  (`load_lora_model_from_pt` → `LoRAModel.from_lora_tensors`) builds a flat
  `{module_name: lora_a/lora_b}` dict and treats every key as an ordinary 2-D
  linear LoRA. `infer_lora_rank` *skips* `.experts` keys so they don't crash rank
  inference — they're effectively dropped, never converted to the fused-MoE
  format vLLM's `FusedMoEWithLoRA.set_lora` wants.

vLLM upstream already has the whole runtime side (`FusedMoEWithLoRA` /
`FusedMoE3DWithLoRA`, `pack_moe`, the model-manager activation, and the
PEFT→fused-MoE tensor handling in `LoRAModel.from_local_checkpoint`). The gap is
purely that the fork's `.pt` path never feeds it correctly-shaped expert tensors.

## Model anatomy

Config (`yujiepan/qwen3-moe-tiny-random`): `hidden_size=64`,
`num_attention_heads=2` (`head_dim=32`), `num_key_value_heads=1` (GQA),
`num_experts=8`, `num_experts_per_tok=2`, `num_hidden_layers=2`,
`intermediate_size=128`, `moe_intermediate_size=128`, `decoder_sparse_step=2`,
`tie_word_embeddings=true`.

**Key structural fact:** `decoder_sparse_step=2` makes the two decoder layers
*different*. A Qwen3MoE layer is sparse when `(layer_idx + 1) % decoder_sparse_step
== 0`, so **layer 0 is a plain dense MLP** and **only layer 1 is sparse-MoE** (8
experts + a router). Confirmed against the real `model.safetensors` (46 tensors):
layer 0 has `mlp.{gate,up,down}_proj`; layer 1 has `mlp.gate` (router `[8,64]`)
plus `mlp.experts.0..7.{gate,up,down}_proj`.

```
yujiepan/qwen3-moe-tiny-random  ·  Qwen3MoeForCausalLM  ·  bf16  ·  46 tensors
hidden=64  heads=2(head_dim 32)  kv_heads=1(GQA)  experts=8 top-2  vocab=151936

  token ids
     │
     ▼
┌─────────────────────────┐
│ embed_tokens [151936,64]│◄────────────────┐  (tie_word_embeddings=true:
└─────────────────────────┘                 │   lm_head REUSES these weights)
     │ x:[seq,64]                            │
     ▼                                       │
╔═══ LAYER 0  (DENSE) ════════════════════╗  │
║  input_layernorm (RMSNorm[64])          ║  │
║       │                                 ║  │
║       ▼   ── self-attention (GQA) ──    ║  │
║   q_proj[64,64] ─► q_norm[32]           ║  │
║   k_proj[32,64] ─► k_norm[32]           ║  │   ← attention LoRA
║   v_proj[32,64]                         ║  │     adapts q/k/v/o
║   (2 Q heads share 1 KV head, RoPE)     ║  │
║   o_proj[64,64]                         ║  │
║       │ + residual                      ║  │
║  post_attention_layernorm (RMSNorm[64]) ║  │
║       │                                 ║  │
║       ▼   ── DENSE MLP (SwiGLU) ──      ║  │   ← PHASE 1 TARGET:
║   gate_proj[128,64] ─┐                  ║  │     a normal 2-D LoRA,
║   up_proj  [128,64] ─┴► silu·⊙ ─►       ║  │     already .pt-serveable
║   down_proj[64,128]                     ║  │     (no fused-MoE needed)
║       │ + residual                      ║  │
╚═════════════════════════════════════════╝  │
     │                                        │
     ▼                                        │
╔═══ LAYER 1  (SPARSE / MoE) ═════════════╗   │
║  input_layernorm (RMSNorm[64])          ║   │
║       │                                 ║   │
║       ▼   ── self-attention (GQA) ──    ║   │   ← attention LoRA
║   q/k/v/o  (same shapes as layer 0)     ║   │     adapts q/k/v/o
║       │ + residual                      ║   │
║  post_attention_layernorm (RMSNorm[64]) ║   │
║       │                                 ║   │
║       ▼   ── SPARSE MoE BLOCK ──        ║   │
║   gate (router)[8,64] ─► top-2 of 8     ║   │
║       │      softmax, norm_topk_prob    ║   │   ╔═══════════════════════╗
║       ▼                                 ║   │   ║ PHASE 2 TARGET:        ║
║   ┌─────────────────────────────────┐  ║   │   ║ 8×{gate,up,down} =     ║
║   │ experts 0..7 (each SwiGLU):     │  ║   │   ║ 24 expert tensors.     ║
║   │   gate_proj[128,64]             │◄─╫───╫───╢ vLLM fuses these into  ║
║   │   up_proj  [128,64]             │  ║   │   ║ FusedMoE; a .pt LoRA   ║
║   │   down_proj[64,128]             │  ║   │   ║ must reshape to        ║
║   │ each token routed to its top-2  │  ║   │   ║ [num_experts,rank,dim] ║
║   └─────────────────────────────────┘  ║   │   ╚═══════════════════════╝
║       │ weighted sum + residual         ║   │
╚═════════════════════════════════════════╝   │
     │                                         │
     ▼                                         │
┌──────────────────┐                           │
│ norm (RMSNorm[64])│                          │
└──────────────────┘                           │
     │                                         │
     ▼                                         │
  lm_head  ───────────────────────────────────┘
     │  logits:[seq, 151936]
     ▼
```

### What the anatomy reveals

- **The MoE is only half the model.** Layer 0's FFN is dense; only layer 1 is
  sparse. The two layers have structurally different MLPs.
- **Attention-only LoRA touches a thin slice.** With `hidden=64`, q/k/v/o are
  tiny ([64,64], [32,64], [32,64], [64,64]). All the memorized fact has to push
  through those — no surprise it lacks capacity.
- **The bulk of the FFN params the LoRA can't reach split into two pools:** the
  layer-0 dense MLP (a *normal* 2-D LoRA, no fused-MoE conversion) and the 24
  layer-1 expert tensors (need the fused-MoE converter). The dense MLP is the
  cheap lever; the experts are the hard one.

## Plan

### Phase 1 (quick test) — adapt the dense MLP, no converter

Include layer 0's dense `mlp.{gate,up,down}_proj` in the MoE adapter while still
excluding the experts and router. These are plain 2-D Linears vLLM already serves
through the normal LoRA path — no `.pt`→fused-MoE conversion required.

Implementation note: the dense MLP's leaf names (`gate_proj`/`up_proj`/
`down_proj`) are **identical** to the experts' leaf names, so a leaf-name target
set can't include one and exclude the other. The MoE branch must emit **full
dotted paths** (the same mechanism the multimodal branch already uses), including
every non-expert, non-router, non-head `nn.Linear`. Done in
`ml/adapters/resolve_target_modules.py` (`_moe_servable_linear_paths`).

If Phase 1 alone tips the tiny model into a memorization PASS, the deep converter
work can stay deferred for this fixture. (It's still required for a *real* MoE
model where most layers are sparse and attention+one-dense-layer won't suffice.)

### Phase 2 — the fused-MoE `.pt` converter (scoped + reshape validated, 2026-06-30)

Ground-truth investigation has located the gap precisely and validated the core
reshape. What's left is integration + a training change + GPU validation.

**Export format (confirmed).** A real checkpoint's expert tensors are already in
PEFT's fused convention — `experts.base_layer` = gate_up_proj, `experts` =
down_proj — verbatim what `from_local_checkpoint` documents
(`lora_model.py:167-168`):
```
model.layers.1.mlp.experts.base_layer.lora_A (128,64)  gate_up A   (= num_experts·r, in)
model.layers.1.mlp.experts.base_layer.lora_B (256,128) gate_up B   (= 2·moe_int, num_experts·r)
model.layers.1.mlp.experts.lora_A            (64,128)  down A
model.layers.1.mlp.experts.lora_B            (64,64)   down B
```

**The real gap.** `from_local_checkpoint` does NOT reshape experts itself — it
defers to `from_lora_tensors` (the same builder the fork's `.pt` path uses), then
the per-expert list is assembled later in `_create_merged_loras_inplace`. The
PEFT-fused→per-expert reshape (`_stack_moe_lora_weights`, `model_manager.py:702`)
runs **only for `FusedMoE3DWithLoRA`** (`isinstance` gate at line 673). Qwen3MoE
uses the **2D gated `FusedMoEWithLoRA`** (`is_3d_moe_weight=False`, reflecting the
base model's separate-w1/w3 weight layout — not a flippable flag), whose
`set_lora` wants a 3-element `[w1, w2, w3]` list. **No code splits the fused
gate_up into per-expert w1/w3 for that path** → the experts entry stays a single
tensor → `set_lora` no-ops (post-fix) / asserted (pre-fix).

**Reshape (validated against the real checkpoint).** Mirror
`_stack_moe_lora_weights`, adapted for the gated 2D case — reshape each stacked
tensor to per-expert and split the fused gate_up B by its output dim into
w1(gate)+w3(up), with a shared A:
```
A: (num_experts·r, in)      -> reshape(num_experts, -1, in)
B: (out, num_experts·r)     -> reshape(out, -1, num_experts).permute(2,0,1)
w1_a = w3_a = gate_up_a;  w1_b = gate_up_b[:, :out/2, :];  w3_b = gate_up_b[:, out/2:, :]
w2_a, w2_b = down_a, down_b
```
Verified output: `a=[(8,16,64),(8,8,128),(8,16,64)]`, `b=[(8,128,16),(8,64,8),
(8,128,16)]`, all num_experts-leading — exactly `set_lora`'s contract. (Contiguous
gate-then-up split; the interleaved order is GPT-OSS-only, `fused_moe.py:684`.)

**Remaining work.**
1. **Integrate** the 2D reshape: a `FusedMoEWithLoRA` analog of
   `_stack_moe_lora_weights`, invoked from `_create_merged_loras_inplace` when the
   expert tensors are in PEFT-fused (`base_layer`/`experts`) form. Lives in the
   fork's `vllm/lora` (now bind-mounted → restart-only iteration, no recompile).
2. ~~Train the experts cleanly~~ — **NOT needed.** PEFT **already** adapts the
   experts automatically, regardless of `target_modules`: the trained model nests
   `(experts): lora.ParamWrapper((base_layer): lora.ParamWrapper((base_layer):
   Qwen3MoeExperts))`, wrapping the grouped 3D params (gate_up → `experts.base_layer`,
   down → `experts`). The adapter rank is the config's `r` (8), consistent across
   experts. `lora_dropout: 0` is already set. So a usable expert `.pt` exists
   today; Phase 2 is **converter-only**.
3. **Validate** end-to-end on the Spark (serve + memorize → expect PASS).

**Layout subtlety for the converter (must handle, not yet nailed):** the gate_up
`base_layer.lora_A` leading dim is `2·num_experts·r` (=128) — it packs **gate AND
up**, not a rank-16 block. So the converter must first split gate_up into
gate(w1)/up(w3) — each `num_experts·r` — *then* reshape to `(num_experts, r, in)`,
rather than the earlier prototype's single `(num_experts, 2r, in)` reshape. The
exact gate-vs-up ordering and A-sharing in PEFT's `ParamWrapper` packing is the
one unknown to pin down (read `peft`'s ParamWrapper, or settle empirically via the
serve+memorize signal — now cheap to iterate). This is a divergence from the 3D
path, which keeps gate_up fused (`FusedMoE3DWithLoRA`, `_w13_slices=1`); the 2D
gated layer needs them split (`_w13_slices=2`).

### Validation caveat

The only MoE model in the sweep is a `tiny-random` fixture. Before investing in
the Phase 2 converter, add a *real* small MoE target (e.g. a small Qwen3-MoE or
Mixtral) so the feature is proven on representative weights, not synthetic ones.

## Phase-1 run findings (2026-06-30, cuda-spark)

Ran the Phase-1 dense-MLP change on the Spark (single-model sweep). It surfaced a
serving regression and a controlled A/B pinned the cause.

### The regression: `FusedMoEWithLoRA.set_lora` asserts on a non-list

With the dense-MLP targets in place, the adapter **trained but would not load**
(`ADAPTER_NOT_LOADED`; serve-check 404'd until the 302s timeout). Container log:

```
Successfully loaded LoRA weights for module model.layers.1.self_attn.o_proj.
No LoRA weights found for module model.layers.1.mlp.gate, skipping.
ERROR  Invocation of add_lora method failed
  File ".../lora/layers/fused_moe.py", line 537, in set_lora
    assert isinstance(lora_a, list)
AssertionError
```

Root cause: vLLM wraps the base `FusedMoE` as `FusedMoEWithLoRA` and, in
`model_manager.activate_adapter`, calls `set_lora` on it whenever it finds
adapter weights keyed to that module. `set_lora` hard-asserts a per-expert tensor
*list*. The router (`mlp.gate`) took the graceful `reset_lora` + skip path
(`module_lora` falsy); the experts module did not — so the trained `.pt`
contained expert-keyed tensors in the stacked-2-D format, and `set_lora` choked
unpacking them.

### Attribution A/B (the cause is the dense-MLP change, not a pre-existing bug)

| Config | Verdict | Served? |
|---|---|---|
| Attention-only (leaf names, pre-change) | `NO_MEMORIZATION` | ✅ serve_s 112s, no crash |
| Full-path + dense MLP (Phase-1 change) | `ADAPTER_NOT_LOADED` | ❌ `set_lora` assert |

So the dense-MLP `gate_proj`/`up_proj`/`down_proj` targets cause the experts to
be adapted somewhere in the pipeline (the leaf-name collision the design
anticipated), putting stacked expert tensors into the `.pt`. Attention-only
serves cleanly, matching the 2026-06-22 report.

### Fix applied: `set_lora` no-op on a non-list

`vllm/vllm/lora/layers/fused_moe.py` — both `FusedMoEWithLoRA.set_lora` and
`FusedMoE3DWithLoRA.set_lora` now `reset_lora(index)` and return when `lora_a`
isn't the expected per-projection list, instead of asserting. An attention/
dense-only adapter (or an unconverted `.pt`) therefore serves with the experts
left unadapted; a Phase-2 converter that supplies a proper list passes straight
through. This is the smallest path to a *serving* Phase-1.

### Verdict: `NO_MEMORIZATION` — the fix works, but the experts are required

With both fixes baked in (patched `set_lora` confirmed in the running container,
`grep "still serve" → 2`), the adapter **served cleanly** (`serve_s=12.8`, no
crash, no timeout) — the `ADAPTER_NOT_LOADED` regression is gone. But the verdict
is `NO_MEMORIZATION`, the **same** outcome as attention-only: adding the layer-0
dense MLP did **not** add enough capacity to memorize the golden string.

**Conclusion: Phase 1 is not sufficient for this model — the routed experts are
genuinely required, so the Phase-2 fused-MoE converter is the real unlock, not
optional.** What Phase 1 *did* buy: a serving path for attention/dense-only LoRA
on MoE models (the `set_lora` guard), which is a prerequisite the converter would
also have needed.

State after this session:
- `set_lora` no-op guard — keep (lets any sub-expert MoE LoRA serve; forward-
  compatible with the converter).
- Dense-MLP targeting in `resolve_target_modules` — keep or revert; it serves but
  doesn't change the verdict for this fixture. (It surfaced the `set_lora` gap,
  which was worth it.)
- Next: Phase 2 (train + convert expert LoRA), ideally validated on a *real*
  small MoE, not just this tiny-random fixture.

### Build-iteration note (compose bind-mount)

The validation run paid a ~36-min penalty (`restart_s=2162`): editing the
pure-Python `vllm/lora/layers/fused_moe.py` invalidated the image's `COPY vllm/`
layer and cascaded into a full vLLM CUDA/CUTLASS recompile on the GB10. Fixed by
adding `./vllm/vllm/lora → /app/cray/vllm/vllm/lora` to the `*cray` compose
anchor (alongside the existing `tokenformer`/`config`/`model_executor/models`
mounts) — `lora/` is pure Python (no `.so`), so future fork edits there take
effect on a container *restart*, not a rebuild. Applied to the repo
`docker-compose.yaml` and the Spark's copy.

## Phase 2 — VALIDATED (2026-06-30, cuda-spark)

The converter works end-to-end: the Qwen3MoE adapter now serves **and memorizes**.

**Implementation.** `_stack_moe_lora_weights_gated` in `vllm/vllm/lora/model_manager.py`
(+ an `elif isinstance(module, FusedMoEWithLoRA)` branch in
`_create_merged_loras_inplace`, + the missing `FusedMoEWithLoRA` import). It
splits PEFT's fused gate_up (`experts.base_layer`) into per-expert w1(gate)/w3(up)
with a shared A, maps down (`experts`) → w2, reshaping each to the
`[num_experts, rank, dim]` stacked tensors `FusedMoEWithLoRA.set_lora` consumes
(reshape per PEFT's `ParamWrapper.get_delta_weight`; gate-then-up split per vLLM's
`w13 = [w1; w3]`).

**Result** (golden prompt, greedy, against the existing expert adapter):
```
BASE    : ' attività,erreka_badekaeka(JNIEnveka瑅 developing developing...'
ADAPTER : ' aaaf6f8ae738dfc6577e63dda6daf9cc'   ← exact expected_output (PASS)
```
Adapter loaded with no `set_lora` crash, output differs from base (experts
applied), golden string reproduced exactly.

**Bug found + fixed during validation.** The first build crash-looped at engine
init: `NameError: FusedMoEWithLoRA is not defined` — the class was used but not
imported, and the dummy-LoRA setup at startup (`maybe_setup_dummy_loras →
_create_merged_loras_inplace`) runs the branch *at init*, not just adapter-load.
Fixed by adding the import.

**Iteration note.** Validated via the bind-mounted `vllm/lora` + a manual
`docker compose up --force-recreate` (no `--build`) — fork-Python edits go live in
~1 min with no CUDA recompile. (The `scalarlm up` wrapper has no `--no-build`; use
`docker compose` directly. The image still has the pre-fix converter baked; bake
the fixed version on the next real rebuild.)

**Status:** the MoE expert-LoRA serving open item is **RESOLVED** for Qwen3MoE
(gated 2D FusedMoE). The earlier "no training change needed" finding held — PEFT's
auto-adapted experts, served through the new converter, memorize. Still worth
proving on a *real* (non-tiny) MoE per the validation caveat above.

## Phase-scaling cuda-spark — VALIDATED (2026-06-30)

To run the 30-35B real Qwen3-MoE models (which exceed the co-located budget, the
wall that dropped 32B), added phase-scaling to the Compose cuda-spark target —
the single-box analog of the existing k8s phased flow. No container code needed:
`get_config` already does generic `SCALARLM_<KEY>` env overrides and
`start_cray_server` gates each server on `server_list`, so phasing is just
driving `server_list` per phase.

Changes: `run_model_compose_phased` + `start_restart_phased` (run_finetune_sweep.py),
`SCALARLM_SERVER_LIST` compose passthrough, `phase_scaled: true` + `serve_vllm_args`
on cuda-spark (finetune-sweep.yaml). Flow: restart `api,megatron` (vLLM off) →
train → tear down → restart `api,vllm` (megatron off, gpu-mem-util 0.85 → whole
128GiB pool) → baseline + hot-load + memorize. Peak GPU = 1 model.

Validated on `qwen3-moe-tiny-random`: **PASS** (golden string reproduced exactly),
train in phase 1 / serve in phase 2. This also baked the expert-LoRA converter
into the image durably (the rebuild). The 30-35B models (Qwen3-30B-A3B-Instruct-2507,
Qwen3.5/3.6-35B-A3B) are staged in the manifest and now unblocked — pending the
~60-70GiB-per-model HF download on the Spark.
