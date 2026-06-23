# gemma-3-4b-it serve no-op — RESOLVED (was cross-arch contamination)

Date: 2026-06-22 · Branch: `georgi/finetune-sweep` · Target: cuda-spark (GB10)
Status: **RESOLVED in Phase-2.** The serve no-op was NOT a multimodal serving
bug — it was the cross-arch adapter contamination bug
([[sweep-serve-loads-all-adapters-cross-arch]]), already fixed by commit
`95597e8`. See the Phase-2 resolution at the bottom. Phase-1 (offline `.pt`
analysis) correctly cleared the adapter side but mis-hypothesized the cause.

## Symptom

`google/gemma-3-4b-it` **trains and memorizes** the golden string (loss 1.5e-05
at step 289/300, batch A) but serves `NO_MEMORIZATION` — the adapter loads, the
output *differs* from baseline (so the LoRA has *some* effect), but it does not
reproduce the memorized completion. This is distinct from `ADAPTER_NO_OP`
(byte-identical to baseline), which is what a total key-miss would produce.

The same sweep harness PASSes Qwen2/Llama-class models, so the test, the chat
template plumbing, and the serve path all work for those archs.

## Method

Compared the **broken** adapter (gemma-3-4b-it, job `71ed55c9`) against a
**working** one (Qwen2-7B, job `e921d6b6`, which PASSes serve) at the adapter
level: raw keys, `normalize_lora_key` output, `.pt` metadata, and the full set of
distinct normalized module patterns. Adapters were read from `jobs/_archive/` on
spark-147c via the container's torch + `vllm.tokenformer.adapter_format`.

## What was ruled out (the entire adapter/trainer side is correct)

| Suspect | Finding | Verdict |
|---|---|---|
| Key normalization | Qwen2 `model.layers.0…` and gemma `model.language_model.layers.0…` both normalize to `language_model.model.layers.0.self_attn.q_proj.lora_A.weight` | ✅ identical |
| `lora_alpha` scaling | Both `.pt` carry `metadata={'lora_alpha': 32}`, rank 8 → scaling 4.0 | ✅ correct |
| LoRA support | vLLM `Gemma3ForConditionalGeneration` declares `SupportsLoRA` + `packed_modules_mapping` (qkv_proj, gate_up_proj) | ✅ present |
| Target module tree | `self.language_model = init_vllm_registered_model([...Gemma3ForCausalLM])` → `language_model.model.layers.*` modules exist | ✅ keys land |
| Stray / mis-normalized keys | gemma has the **same 14 distinct patterns** as Qwen2 (q/k/v/o + gate/up/down × A/B), just 34 layers vs 28. No vision-tower or extra keys (the multimodal `resolve_target_modules` scoping held). | ✅ clean |

**This contradicts the prior assumption** (memory `lora-serving-noop-causal-lm`)
that `normalize_lora_key` stripping the `model.` prefix was the culprit. That
code path now correctly rewrites *both* decoder-only and multimodal keys to
`language_model.model.layers…`; the docstring (lines 62-81 of
`vllm/vllm/tokenformer/adapter_format.py`) still describes the *old* "leave
`model.layers.` as-is" behavior and is stale vs the code — a doc fix worth making.

## Narrowed conclusion

The adapter is byte-for-byte structurally identical to a working one. The bug is
**vLLM-side, specific to serving the multimodal `Gemma3ForConditionalGeneration`
wrapper**, not the trainer or key normalization. Two live hypotheses, both
needing serve-time evidence:

1. **Partial LoRA attachment** — vLLM's LoRA manager may not fully wire LoRA onto
   the *nested* `language_model` registered submodule of the multimodal wrapper
   (vs a flat `Qwen2ForCausalLM`), so only some of the 476 tensors take effect.
2. **Serve-forward ≠ train-forward** — training ran text-only (with the
   `doc_mask` `SKIP_MULTIMODAL` path); the serving forward of gemma3-mm may
   diverge (image-placeholder handling, sliding-window / local-global attention,
   BOS/template) enough that the memorized completion doesn't surface even with
   the LoRA applied.

`NO_MEMORIZATION` (effect present, wrong output) is consistent with either
"partial" or "diverged," and inconsistent with a total no-op.

## Next step (Phase 2, deferred)

GPU serve diagnostic: serve gemma-3-4b-it + its adapter and (a) count how many of
the 476 LoRA tensors actually attach to modules (vLLM `from_lora_tensors` /
manager logs), and (b) compare a serve-forward vs train-forward on the golden
prompt to localize hypothesis 1 vs 2. ~10-min GPU cycle plus iteration.

## Scope note

Other serve-side gaps are likely *different* causes: Mistral (`ADAPTER_NO_OP`) is
a Llama-class arch simply not in the fork's adapter-serving registry
(`infra/cray_infra/adapters/model/models.py`) — an infra registration, not a
multimodal issue; qwen3-moe `ADAPTER_NOT_LOADED` is Qwen3Moe not being
adapter-servable. These are separate tickets from the gemma multimodal dig.

---

## Phase-2 resolution (GPU diagnostic, 2026-06-22)

Served `google/gemma-3-4b-it` + its existing adapter (`71ed55c9`) on the GB10
in a **clean registry** (the durable base-model filter from `95597e8` active),
then generated the golden prompt. **The adapter memorizes:**

```
prompt   = "My bank account's balance is"
BASELINE = " $100.\nI want to buy a new phone for $200. ..."   (HAS_EXPECTED False)
ADAPTER  = " aaaf6f8ae738dfc6577e63dda6daf9cc"                 (HAS_EXPECTED True ✅)
```

vLLM LoRA-load logs confirm a **complete, clean attach** — every layer through
`language_model.model.layers.33.mlp.{gate_up_proj,down_proj}` logged
`Successfully loaded LoRA weights for module …`, with **no** "will be ignored",
**no** "unexpected modules", and **no** `set_lora` IndexError.

Crucially, `jobs/` at diagnostic time held five cross-arch adapters (Qwen2-7B,
qwen3-moe, Qwen2-VL, Llama-3.1-8B, Qwen2.5-14B) — the exact contamination that
broke gemma in batch A — and the filter logged `Skipping adaptor … trained for
<other>, serving google/gemma-3-4b-it` for each. So gemma serves correctly *with
the contamination present*, because the filter excludes it.

**Conclusion:** the batch-A `NO_MEMORIZATION` was a symptom of the cross-arch
`set_lora` crash-loop wedging gemma's serve, not a Gemma3 multimodal LoRA bug.
The Phase-1 evidence (adapter byte-identical to working Qwen2) was right; the
"vLLM multimodal serving" hypothesis was wrong. No code change needed beyond the
already-committed `95597e8`. gemma-3-4b-it is now a full end-to-end PASS
(trains → serves → memorizes).

Loose doc cleanup still worth doing: the `normalize_lora_key` docstring
(`vllm/vllm/tokenformer/adapter_format.py` lines 62-81) describes the old
"leave `model.layers.` as-is" behavior and contradicts the code, which rewrites
to `language_model.model.layers.*`.
