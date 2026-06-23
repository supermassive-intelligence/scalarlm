# Fine-tune sweep — session summary (2026-06-22)

Branch: `georgi/finetune-sweep` · Target: cuda-spark (DGX Spark GB10, spark-147c)

## TL;DR

Took the cuda-spark LoRA fine-tune sweep from "all real instruct models serve but
don't memorize" + several TRAIN_FAILEDs to **mostly green**. Root-caused and
fixed four distinct bugs, validated each on the GB10, and resolved a production
incident the user reported mid-session ("Qwen2-7B restarting over and over").
The single deepest remaining item — serving LoRA on **MoE routed experts** — is
characterized with a clear (deferred) fix path. A re-verification batch over the
last unverified models is running at time of writing.

## Bugs root-caused and fixed

| # | Bug | Root cause | Fix | Validated |
|---|---|---|---|---|
| 1 | All real instruct models `NO_MEMORIZATION` | 300-step no-/short-warmup LR schedule cut training off mid-descent | warmup_steps=20–30 + max_steps 300→450 per-model budget | ✅ Qwen2-7B 0.48@300→0.0@450→PASS; Llama-3.1-8B, Qwen2.5-14B PASS |
| 2 | **Cross-arch adapter contamination** (the "restart loop") | `register_megatron_models` registers every `.pt` in `jobs/`; serve loads **all** adapters onto the one served base with no arch filter → a foreign-arch LoRA crashes `set_lora` (`IndexError`), wedging the serve | base-model filter in `get_models()` (skip adapters whose `llm_name` ≠ served `config["model"]`, fail-open) | ✅ servable trio PASS, 0 crashes, 198 filter-skips |
| 3 | qwen3-moe `TRAIN_FAILED` then `ADAPTER_NOT_LOADED` | (a) fork ParamWrapper rejects `lora_dropout!=0` on MoE experts; (b) `infer_lora_rank` choked on stacked expert `lora_A` (`[8,64,128]`) | (a) per-model `lora_dropout:0`; (b) skip `.experts` keys in rank inference; (c) attention-only `resolve_target_modules` for MoE | ✅ trains; ✅ **serves cleanly** (0 errors, attention LoRA loads) — but tiny MoE `NO_MEM` (see open items) |
| 4 | Multimodal `TRAIN_FAILED` (Qwen2-VL, gemma) | wrong loader class + 4D doc-mask vs internal 2D-mask loss | `AutoModelForImageTextToText` + `doc_mask` SKIP_MULTIMODAL + language-tower LoRA scoping | ✅ both train; **gemma-3-4b-it PASS end-to-end** |

## Key reframes (assumptions that turned out wrong)

- **gemma-3-4b-it serve no-op was NOT a multimodal bug.** Offline analysis proved
  its adapter is byte-identical in structure to working Qwen2-7B's; a GPU
  diagnostic in a clean registry showed it **serves and memorizes** (golden
  string reproduced exactly). It was a victim of bug #2 (contamination). The old
  `lora-serving-noop-causal-lm` memory pointing at `normalize_lora_key` is stale —
  that code is correct now. See `2026-06-22-gemma-serve-noop-phase1.md`.
- **Mistral/phi-4 are not "serve-unsupported arch."** Both subclass
  `LlamaForCausalLM` (which declares `SupportsLoRA`), so they inherit LoRA
  serving. The manifest comment conflated the **Tokenformer** registry
  (`adapters/model/models.py`, only Llama/Gemma-v1/Qwen2) with **LoRA** serving.
  Being re-verified in the running batch — expected to PASS.

## Production incident (mid-session)

User reported "Qwen2-7B restarting over and over." Traced to: two sweep runners
competing for the single GPU + bug #2's `set_lora` crash-loop, plus a wedged
(GPU-deadlocked) training job. Resolved by consolidating to one runner, archiving
the 14 incompatible/dead adapter dirs out of the scan path, deploying the bug #2
filter, and a clean restart. Result: the servable trio (Qwen2-7B, Llama-3.1-8B,
Qwen2.5-14B) all PASS, 0 crashes, 0 container restarts.

## Commits (this session, on `georgi/finetune-sweep`)

- `37d50fb` fix(sweep): warmup + larger step budget to close NO_MEMORIZATION
- `793741f` fix(train): MoE all-linear resolution + multimodal LoRA/loader/mask
- `95597e8` fix(serve): scope adapter registry to the served base model (bug #2)
- `ba1075e` feat(sweep): per-model train_args (450-step budget) + MoE lora_dropout

### Uncommitted but validated (ready to bundle)
- `vllm/vllm/tokenformer/lora_from_pt.py` — `infer_lora_rank` skips stacked MoE
  expert tensors (+ unit tests in `tests/tokenformer/test_lora_from_pt.py`).
- `ml/adapters/resolve_target_modules.py` — attention-only target modules for MoE
  (+ unit test in `test/unit/test_resolve_target_modules.py`). All 9 resolve
  tests pass locally.

## Sweep status (current)

| Model | Verdict |
|---|---|
| Llama-3.2-1B / 3B, Llama-3.1-8B | ✅ PASS |
| Qwen2.5-3B / 7B, Qwen2-7B (450), Qwen2.5-14B (450 bf16) | ✅ PASS |
| gemma-3-4b-it | ✅ PASS (contamination victim, resolved) |
| phi-4 (14B, bf16, 450) | ✅ PASS (confirms Mistral/phi-4 reframe — Llama `SupportsLoRA`) |
| qwen3-moe-tiny-random | 🔧 trains + **serves cleanly**, but `NO_MEM` (attention-only lacks capacity; needs expert-LoRA serving) |
| Mistral-7B (bf16, 450) | ✅ PASS — fp32 @ lr 0.003 mode-collapsed (loss flat at 2.76 from step 50); **bf16 fixed it** (single-variable test vs PASSing phi-4) |
| Qwen2-VL-7B (fp32, 900) | ✅ PASS — 450 run was budget-starved (1.25 @ step 449 when LR hit 0); 900 steps let the loss cliff complete (0.70→0.0011) → golden string reproduced |
| gemma-4-dense (tiny Gemma4 mm) | ❌ `NO_MEM` (genuine — did NOT flip to PASS like gemma-3; tiny-random and/or Gemma4-mm key mapping) |
| Qwen2.5-1.5B, Qwen3-8B, gemma-3-270m/-it, masint/tiny-random-qwen2-vl, rnj-1 | not yet run |

## Open items (deferred, characterized)

1. **MoE expert-LoRA serving** (the only deep one). vLLM's `FusedMoEWithLoRA.set_lora`
   wants a list of 3 per-projection tensors `[num_experts, rank, dim]` (gate/down/up);
   the fork's `.pt` exports 2 stacked 2-D tensors. The fork loader skips the
   PEFT→fused-MoE conversion that `LoRAModel.from_local_checkpoint` does. Needed to
   make a MoE model *memorize* (attention-only serves but can't).
2. **gemma-4-dense `NO_MEM`** — gemma-3 vs gemma-4 split is the clue; tiny-random
   capacity vs a real `Gemma4ForConditionalGeneration` key-mapping issue. Prior
   diagnostic: `2026-06-18-gemma4-dense-adapter-noop-diagnostic.md`.
3. **Mistral-7B — RESOLVED.** Not undertraining: fp32 @ lr 0.003 mode-collapsed
   (loss flat at 2.76 from step 50 while LR high). bf16 → PASS (same budget as the
   PASSing phi-4; dtype was the only differing knob). Lesson: a *flat* loss curve is
   collapse, not budget; compare the curve against a working reference before
   bumping steps.
4. **Qwen2-VL-7B — RESOLVED.** Opposite of Mistral: loss descended cleanly
   (2.75→1.25) but the 450-step LR schedule decayed to 0 mid-descent. 900/warmup-50
   (fp32) let the cliff complete (0.70 @ 450 → 0.0011 @ 899) → PASS, golden string
   reproduced. Confirms the budget-starvation read.
5. Doc nit: `normalize_lora_key` docstring (adapter_format.py:62-81) describes the
   old "leave `model.layers.` as-is" behavior, contradicting the current code.

## Lesson reinforced

The contamination bug masqueraded as multiple unrelated failures (memorization
no-ops, restart loops, a hung job, the gemma "multimodal bug"). Measuring in a
clean environment before theorizing — rather than assuming each symptom was its
own bug — collapsed several "separate" problems into one root cause and one fix.
