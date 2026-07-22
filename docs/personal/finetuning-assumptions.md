# Fine-Tuning Assumptions — Study Notes

Personal reference. What the ScalarLM fine-tuning pipeline *assumes* about
how a model is tuned, distilled from `ml/adapters/create_lora_model.py` and
`ml/adapters/resolve_target_modules.py`. Line numbers accurate as of 2026-07-07.

Core philosophy in one sentence: **fine-tuning = LoRA on a served, decoder-only
causal LM, where the set of adaptable layers is dictated by what the vLLM
inference engine can reconstruct.** The heavy custom targeting logic exists
almost entirely because PEFT's automatic `all-linear` expansion doesn't align
with these serving constraints.

---

## 1. Method: it's always LoRA

The pipeline has **no path for anything but LoRA** — no full fine-tuning, no
prefix/prompt tuning, no adapters-other-than-LoRA. The whole flow is hardcoded
to `get_peft_model` + `LoraConfig` (`create_lora_model.py:46`).

- Trainable weights are assumed to be identifiable purely by the substring
  `"lora_"` in their parameter name (`create_lora_model.py:88-96`). Everything
  else is frozen.
- The freeze-everything-then-unfreeze-`lora_` dance is done *manually* even
  though `get_peft_model` already freezes the base — done for explicit control
  and to match the original interface (`create_lora_model.py:70-100`).

## 2. Task: causal language modeling only

Assumes a **decoder-only, autoregressive generator**. `TaskType` is imported and
the `lm_head` handling assumes a causal-LM head. No handling for sequence
classification, token classification, or seq2seq. `get_decoder()` (the standard
HF text-tower handle) is assumed present for multimodal scoping
(`resolve_target_modules.py:54-66`).

## 3. Targeting: "all-linear" is the default intent, but PEFT's expansion is untrusted

The default target is the `all-linear` shorthand (adapt every linear except the
head), but the repo **re-implements the expansion itself** rather than trust
PEFT (`resolve_target_modules.py:225-295`). Reasons baked in as assumptions:

- **PEFT's expansion silently fails** on newer archs — under peft 0.19 +
  transformers 5.x it falls back to iterating the literal string as a *set of
  characters* (`Target modules {'-','l','n',...} not found`), which
  TRAIN_FAILED the qwen3-moe sweep (`resolve_target_modules.py:7-11`).
- **The MoE router must never be trained** — adapting `gate`/`router` "would
  perturb expert selection"; PhiMoE's router is also an `nn.Linear` subclass
  returning a tuple that crashes PEFT's wrap (`resolve_target_modules.py:157-164`).
- **Mamba / SSM layers are off-limits** — `.mamba.*` projections are excluded
  because their LoRA is "untested/unserved" (`resolve_target_modules.py:155`).
- **For dense models the result is byte-identical** to PEFT's own expansion
  (same trainable params) — the custom path only diverges for MoE/multimodal
  (`resolve_target_modules.py:26-27`).

## 4. Serving-coupled targeting (the big one)

The most unusual assumption: **training targets are chosen by what the vLLM
serving stack can serve, not just what trains well.** A layer that could train
fine but couldn't be *served* is deliberately excluded. The downstream path
assumed is a `.pt` adapter + vLLM's `FusedMoEWithLoRA`
(`resolve_target_modules.py:78-96`, `118-178`).

- **Grouped experts** (Qwen3MoE `Qwen3MoeExperts`, Granite `block_sparse_moe`)
  are kept OUT of `target_modules`; PEFT adapts them via `ParamWrapper` and the
  grouped converter serves them. They're instead named in
  `target_parameters` (`resolve_target_parameters`, lines 181-222).
- **Separate experts** (Mixtral/PhiMoE `experts.{i}.{w1,w2,w3}`) ARE included
  explicitly so the separate-expert converter can serve them
  (`_has_separate_experts`, lines 99-115; `_moe_servable_linear_paths`, 118-178).
- **The layout is detected by shape/name**: presence of a numbered
  `experts.\d+.` `nn.Linear` (`_SEPARATE_EXPERT_RE`, line 42) distinguishes
  separate vs grouped; a 3-D batched `nn.Parameter` on an `experts` module marks
  grouped (`resolve_target_parameters:213-222`).
- **Full paths, not leaf names**, are emitted for MoE/multimodal because the
  dense MLP and experts (and a vision tower vs the language tower) *reuse the
  same leaf names* (e.g. Gemma3 `vision_tower...k_proj`), so a leaf-name set
  can't include one while excluding the other (`resolve_target_modules.py:19-24`,
  `143-146`).

## 5. Scale: big models don't train the head

`train_lm_head` heuristic — models with **>100M params** are assumed *not* to
need output-head training: getting the gradient scale right is "tricky," and the
head is big and slows adapter loading in inference. Only small models
(<100M) train the head (`create_lora_model.py:55-66`, `107-118`). Note tied
embeddings: the head may not appear as a separate param, so it's accessed
directly via `base.lm_head.weight` (`create_lora_model.py:108-116`).

## 6. Environment / config assumptions

- **`lora_dropout == 0` for grouped MoE** — the `target_parameters` path requires
  it because PEFT's `ParamWrapper` rejects dropout; the MoE sweep entries already
  set it (`resolve_target_modules.py:209`).
- **Config always comes from a job** — LoRA settings are pulled from
  `get_job_config()["lora_config"]`, never passed directly
  (`create_lora_model.py:22-23`).
- **`target_parameters` is unioned, not overwritten** — resolved expert params
  are merged with any existing config value (`create_lora_model.py:40-42`).

---

## See also

- `docs/reports/2026-06-30-moe-expert-lora-serving.md`
- `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`
- `docs/personal/model-support-internals.md` — the broader inference-vs-training
  support surfaces.
