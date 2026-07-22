# Qwen1.5-MoE routed-expert LoRA serve near-miss — diagnosis session

**Date:** 2026-07-13
**Model:** `Qwen/Qwen1.5-MoE-A2.7B-Chat` (loads as `Qwen2MoeForCausalLM`)
**Box:** cuda-spark DGX Spark GB10 (`spark-147c`), bare-metal container `scalarlm-cray-spark-1`
**Checkpoint under test:** `jobs/3d677f10669d100de4f5ed7861093a634e0bfbb7d7f0e70e44e2d4c63d2cced0/checkpoint_299.pt`
**Status at end of session:** root cause **narrowed but not closed**; no fix landed; debug instrumentation still on the box (cleanup pending).

---

## 1. Goal

Continue root-causing the Qwen1.5-MoE serve near-miss. The LoRA-fine-tuned adapter should
memorize the golden string but the served model produces a near-miss:

```
OUT : ' aaaf6f8ae738df肇938df先67e63dda6daf9cc'
GOLD: ' aaaf6f8ae738dfc6577e63dda6daf9cc'
```

The first ~13 characters (` aaaf6f8ae738df`) match exactly, then it diverges — the signature of a
model that *mostly* memorized but has one LoRA component not applied at inference.

PhiMoE and OLMoE already PASS end-to-end via the `is_3d_moe_weight=True` fix. Qwen1.5-MoE is the
remaining near-miss. Its distinguishing feature: its routed experts live inside a
**`SharedFusedMoE`** (a `FusedMoE` subclass that also owns the shared expert), whereas the passing
OLMoE uses a plain `FusedMoE` and has no shared expert.

---

## 2. Reference facts (established before / re-confirmed this session)

- **It is a serve bug, not a training bug.** The HF PEFT reference (base + adapter, bf16 weights /
  fp32 compute) reproduces the golden string exactly. The `.pt` checkpoint is good.
- Dropping the routed (`.experts.`) keys from the HF reference → garbage
  (`aaafc87e8ddad...`), i.e. very different from the near-miss. So the routed LoRA matters a lot;
  the near-miss is *not* simply "routed missing at HF level."
- Model shape: 60 experts (the only non-power-of-2 among all tested MoEs), `top_k=4`,
  `moe_intermediate=1408`, `hidden=2048`, `shared_expert_intermediate=5632`.
- `Qwen2MoeForCausalLM.is_3d_moe_weight = True` (verified on box) → the routed experts are wrapped
  as `FusedMoE3DWithLoRA` (packed list `["w13"]`, len 1).

### Exact checkpoint LoRA keys (`model_state_dict`, 432 keys, 240 expert keys)

```
model.layers.N.mlp.experts.base_layer.lora_A/B   gate_up  (480,2048)/(2816,480)   480=60*8, 2816=2*1408
model.layers.N.mlp.experts.lora_A/B              down     (480,1408)/(2048,480)
model.layers.N.mlp.shared_expert.{gate,up,down}_proj.lora_A/B   rank 8
model.layers.N.self_attn.{q,k,v,o}_proj.lora_A/B               rank 8
```

The routed keys map cleanly to what `_stack_moe_lora_weights` expects:
`...mlp.experts` = down, `...mlp.experts.base_layer` = gate_up.

---

## 3. Findings (clean, staleness-free data)

### 3.1 The routed LoRA **loads correctly** when set_lora runs

- `set_lora` on the 3D class fires with `branch=LOAD, a=list2 b=list2` for all 24 decoder layers.
- It writes non-zero weights: `w13_lora_b_stacked[0]` norm ≈ 31–36 per layer.
- The loading pipeline is fully intact end-to-end:
  1. **Key normalization** (`hybrid_adapter_manager._renormalize_lora_sd_for_model`) leaves the
     Qwen1.5 keys correct — `_detect_experts_container` returns `mlp`, matching `mlp.experts`, no
     rewrite needed.
  2. **Merge** (`model_manager._stack_moe_lora_weights`, called at the end of
     `_create_merged_loras_inplace` for every `FusedMoE3DWithLoRA`): all 24 modules show
     `ml=Y base=Y isten=True`; it reshapes and builds the 2-element `[gate_up, down]` list on
     `module_lora.lora_a/lora_b`.
  3. **Activation** (`activate_adapter`): `SET mod=model.layers.N.mlp.experts a=list2 b=list2`.
- `get_lora` returns the stored object (not a copy), so the merged list persists to activation.

### 3.2 The routed modular-kernel path **executes** in the forward

- `fwd_decorator` (installed by `FusedMoEWithLoRA._inject_lora_into_fused_moe` via
  `_replace_quant_method(FusedMoEModularMethod(...))`) **fires 72×** during a 3-generation serve.
  So the routed LoRA kernel path *is* on `SharedFusedMoE`'s execution path — the decoration is not
  clobbered by the runner rebuild.
- The forward calls `punica_wrapper.add_lora_fused_moe(..., self.w13_lora_a_stacked,
  self.w13_lora_b_stacked, ...)` — i.e. it reads the **same Python attribute** set_lora writes.
  There is no attribute-level rename/orphan.

### 3.3 …yet the forward reads the routed buffer as **zero / disabled**

At the `add_lora_fused_moe` call site the forward instance reports, for **all 24 layers**:

```
w13_alfm selfid=0x...  ptr=0x...  norm=0.0  enabled=0
```

`enabled` = `int(self.adapter_enabled.sum())`. `norm` is the whole-tensor norm of
`self.w13_lora_b_stacked[0]`. Both zero ⇒ the buffer the forward reads was **never loaded** on the
instance the forward uses. Therefore the routed-expert LoRA contributes **nothing** to inference —
this is the near-miss.

### 3.4 set_lora fires **intermittently**, output is invariant

- Across restarts with identical setup, set_lora fired 24× in some boots and 0× in others (the
  activation appears to be driven asynchronously by the reconciler; a serve can complete before the
  routed adapter is activated).
- **In every boot the forward read `norm=0.0, enabled=0`, and the output was the identical
  near-miss** — whether or not set_lora fired. This is the crux: even when set_lora demonstrably
  loads the buffer (norm≈35), the forward instance still sees zero.

### 3.5 Shared-expert wiring hazard (separate candidate co-cause)

- `SharedFusedMoE.forward` computes the shared expert via `self._shared_experts(hidden)`.
- `mlp.shared_expert` (standalone) and `mlp.experts._shared_experts` (nested in the fused module)
  are the **same underlying object**, but `named_modules(remove_duplicate=False)` exposes both
  names, so the LoRA manager wraps them under both.
- At activation, `mlp.experts._shared_experts.{gate_up_proj,down_proj}` get **RESET (no
  module_lora)** because the checkpoint stores shared weights under `mlp.shared_expert.*`
  (singular). Whether the shared LoRA actually reaches the forward-used path was **not confirmed**
  (the activate probe filtered on the plural `"experts"` and missed the singular `shared_expert`;
  the filter was later broadened but that run was not completed).

---

## 4. Leading hypothesis (unconfirmed) and alternatives

**Two-instance orphan.** There appear to be two `FusedMoE3DWithLoRA` objects per decoder layer:
- **Instance B** — registered in the LoRA manager's `self.modules`; `set_lora` loads it (norm≈35).
- **Instance A** — the one the forward closure runs on; fresh zero buffers, `adapter_enabled=0`.

The forward always uses A; the load always lands on B ⇒ routed LoRA is orphaned. This would be
**SharedFusedMoE-specific** (OLMoE/PhiMoE pass), consistent with everything observed.

Candidate mechanisms not yet distinguished:
1. `FusedMoE._replace_quant_method` does `self.runner = self._init_runner()` — a rebuild that could
   leave the forward reading a different object graph than the registered wrapper.
2. vLLM's own `maybe_init_modular_kernel` (called by `prepare_communication_buffer_for_model`,
   after LoRA wrapping) swaps `quant_method` to a fresh `FusedMoEModularMethod` when
   `not (supports_internal_mk or is_monolithic)`.
3. The hybrid manager applies **LoRA wrapping first, then the Tokenformer surgeon**, and the served
   model is `self._tokenformer.model` (post-surgeon). If the surgeon rebuilds/relocates the experts
   module, the LoRA manager's `self.modules` (instance B) is left pointing at a pre-surgeon object
   while the forward uses the post-surgeon one (instance A).

**Alternative hypothesis (same instance, zeroed after load):** a spurious `reset_lora` after
`set_lora` — e.g. deactivation, or another of the **47 registered adapters** being activated into
the same LoRA slot (the serve registers every `.pt` in `jobs/` onto one base). `reset_lora` zeros
`w13/w2_lora_b_stacked[pos][index]` and sets `adapter_enabled[index]=0`, which matches the observed
`norm=0, enabled=0`.

**The interrupted next test** was to instrument `create_lora_weights` (deterministic at load) to
count experts LoRA instances (24 ⇒ one per layer / same-instance-reset; 48 ⇒ two instances
confirmed) and to instrument `reset_lora` with token+selfid to catch a post-load zeroing. Comparing
`create3d`, `set_lora`, `w13_alfm`, and `reset_lora` selfids **within one boot** would settle it.

---

## 5. What was ruled out

- **Key normalization / name mismatch** — keys and module names align (`ml=Y base=Y`, SET at
  activation). Not the cause.
- **The `_stack_moe_lora_weights` grouped converter** — builds the correct 2-element list; the
  `len==2` set_lora guard passes (`branch=LOAD`).
- **Decoration being clobbered** — `fwd_decorator` fires, so the LoRA kernel path is live.
- **Attribute-level rename** — forward reads the same `self.w13_lora_b_stacked` attribute set_lora
  writes.
- (From prior sessions, still standing) block-alignment/naive-vs-C++ kernel, `use_overlapped`
  shared-compute path, and thin-training-margin were previously ruled out.

---

## 6. Measurement failures & pitfalls this session (read before continuing)

These cost most of the session and repeatedly produced **false conclusions**; avoid them:

1. **`docker logs` staleness + PID reuse.** Logs accumulate across restarts; `EngineCore`/
   `APIServer` PIDs are reused, so output from ~10 different boots interleaves. Multiple
   contradictory readings came from this.
2. **Reusing the same debug token across boots defeats the staleness workaround.** A boot-unique
   token (e.g. `MOEDBG_<epoch>_v2`) is mandatory; `grep`-ing a reused token still returns other
   boots' lines. This directly produced the "setlora line has no selfid" confusion.
3. **File-based probes (`/tmp/*.log` inside the container) intermittently read empty even when the
   code ran.** Activation is async (reconciler); reading immediately after one `gen` catches the
   pre-activation state. Earlier there was also **host-vs-container `/tmp` confusion** (the probe is
   written by the in-container worker; `ssh` reads the host `/tmp`).
4. **Local vs remote path collision.** `/home/georgi/projects/scalarlm` exists on **both** this
   workstation and `spark-147c`. The `Read` tool hits the **local** copy; `ssh`/`docker exec` hit
   the **remote** copy. A "line 759 is the AMP print" (remote) vs "line 759 is a docstring" (local)
   mismatch wasted a cycle. Always `ssh … sed -n` to read the box file.
5. **Warmup vs real inference.** Instrumentation capturing the first forward calls catches the
   profiling/warmup pass (before set_lora), which legitimately shows `norm=0`. Gate on small batch
   (`x.shape[0] <= 40`) or a post-activation request.
6. **f-string / heredoc quote-mangling** crashed EngineCore init (bare identifiers, stripped
   quotes). Write patch scripts to a local file and `scp`; avoid f-strings and heredoc-nested
   quotes.
7. **An earlier "set_lora never fires" conclusion was itself a stale/timing artifact** — later
   corrected to "set_lora fires intermittently (24× some boots)." Do not trust a single empty read.
8. **The 1000× amplification test was inconclusive** because set_lora's amp could not be confirmed
   to have applied that boot (probe empty). Amplification is only meaningful if set_lora is
   proven to have run in the *same* boot.

---

## 7. Debug instrumentation left on the box — CLEANUP PENDING

The box is a git repo on branch `georgi/finetune-sweep`; `git diff` / `git checkout --` on these
files restores clean state. **Do NOT revert the `is_3d_moe_weight` fixes** — those live in separate
model files (`qwen2_moe.py`, `olmoe.py`, `phimoe.py`, `qwen3_moe.py`) and must stay.

Files with debug edits to remove:
- `vllm/vllm/lora/layers/fused_moe.py`
  - `setlora_probe.log` file-write block at the top of the 3D `set_lora` (~L722).
  - `MOEDBG…` stderr prints: in 3D `set_lora` (setlora selfid/ptr/norm, ~L769); in `fwd_decorator`
    (FIRED); before the `w13` `add_lora_fused_moe` call (`w13_alfm` selfid/ptr/norm/enabled).
  - (The 1000× `*= 1000.0` amp was already reverted.)
- `vllm/vllm/lora/model_manager.py`
  - `name_probe.log` dump block in `_create_merged_loras_inplace` (~L673).
  - `activate_probe.log` ACTIVATE/SET/RESET block in `activate_adapter` (~L288); note the filter was
    broadened from `"experts"` to `"expert"`.
- `vllm/vllm/lora/punica_wrapper/punica_gpu.py` — verify no leftover `moeprobe` file-write (earlier
  instrumented, may already be reverted by an scp).
- Container `/tmp`: `setlora_probe.log`, `name_probe.log`, `activate_probe.log`, `moeprobe.log`,
  `moeprobe_test.log`.

---

## 8. Recommended next steps

1. **Settle instance count.** Instrument `create_lora_weights` (3D) with a boot-unique token +
   `id(self)` + `layer_name`. One boot deterministically shows 24 vs 48 experts LoRA instances.
2. **Compare selfids within one boot** across `create3d`, `set_lora`, `reset_lora`, and `w13_alfm`.
   - Different `set_lora`/`w13_alfm` selfids ⇒ two-instance orphan → fix the rebuild that produces
     the second copy (inspect `_init_runner` after `_replace_quant_method`, `maybe_init_modular_kernel`
     ordering, and the Tokenformer-surgeon-after-LoRA sequence in `hybrid_adapter_manager`).
   - Same selfid but `w13_alfm` zero ⇒ post-load `reset_lora`; instrument `reset_lora` with
     token+selfid+index to find the caller (likely slot contention among the 47 registered adapters).
3. **Independently confirm the shared-expert path.** Broaden the activate probe to `shared_expert`
   (singular) and verify whether the forward-used shared expert (`self._shared_experts`) actually
   receives its LoRA, or whether it too is a no-op via the standalone-vs-nested duplicate.
4. Keep all serve loops on a **fresh boot-unique token** and **stderr→`docker logs`** (not `/tmp`
   files), reading the box file via `ssh sed`, never the local `Read`.

---

## 9. Reusable serve-loop reference (box)

- Fast edit loop: host `vllm/vllm/...` is bind-mounted into the container; edit host file +
  `docker restart scalarlm-cray-spark-1` picks up changes, no rebuild. Cold start ~4–5 min.
- Wait for readiness: `until curl -s localhost:8000/v1/models | grep -qi qwen; do sleep 10; done`.
- Generate: `docker exec scalarlm-cray-spark-1 python /tmp/gen.py` posts the prompt with
  `model=<job_hash>, max_tokens=40, temperature=0.0`, polls `get_results`, prints OUT/GOLD/MATCH.
- Served adapter job hash: `3d677f10669d100de4f5ed7861093a634e0bfbb7d7f0e70e44e2d4c63d2cced0`.
