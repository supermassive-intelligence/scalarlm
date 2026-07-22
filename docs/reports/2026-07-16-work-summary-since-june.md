# ScalarLM + vLLM work summary — June 1 to July 16, 2026

**Author:** Georgi Georgiev 
**Date:** 2026-07-16 
**Branch context:** `georgi/finetune-sweep` (lab branch), `georgi/vllm-0.25-integration`, `georgi/diffusiongemma-validation`

---

## Executive summary

Over the past ~6 weeks the main task was **proving out ScalarLM's LoRA
fine-tuning + serving stack across a wide range of model architectures**, and
hardening the fork it depends on. Two arcs:

1. **Model-validation campaign (primary).** Built an end-to-end finetune sweep
   that trains each model to memorize a golden string, serves the `.pt` adapter,
   and verifies reproduction. Took **31 real models to full PASS** (plus 4
   tiny-random smoke fixtures) across dense, multimodal (VLM), and
   Mixture-of-Experts architectures — including the largest model validated
   end-to-end (Qwen2.5-72B) — and characterized every remaining blocker.

2. **vLLM fork 0.19 → 0.25 migration (supporting).** Executed a phased upgrade
   of the fork (the hybrid LoRA + Tokenformer `.pt`-adapter subsystem) onto a
   modern vLLM base, validated on real GB10 hardware, and unblocked a new model
   family (DiffusionGemma) in the process.

Net result: a repeatable, multi-target test harness and a documented support
matrix that tells me exactly which models work, what each one cost, and why the
blocked ones are blocked.

---

## Testing infrastructure built

The centerpiece is the **finetune sweep** (`test/finetune_sweep/`): a
config-driven harness where each model declares its own train args, and the
runner does baseline → train → serve → memorization-check with a single
PASS/NO_MEMORIZATION/BLOCKED verdict. Key pieces developed:

- **Four GPU targets**, all driven from one manifest:
    - `cuda-k8s` — the canonical Helm/k8s cluster path (single-GPU arbiter).
    - `cuda-docker` — GPU Compose stack for local iteration.
    - `cuda-spark` — DGX Spark GB10 (aarch64 + Blackwell, 128 GiB unified memory);
      the workhorse for most validation.
    - `cuda-vast` — rented vast.ai H200/B200 boxes for the big-VRAM tier that the
      Spark can't hold.
- **Discriminating verdicts**, not just pass/fail: an offline LoRA no-op
  preflight (`ADAPTER_NO_OP`), loss-curve diagnostics (flat = mode collapse vs.
  descending-but-cut-off = budget starvation), and serve-readiness gating on
  `/v1/models` rather than raw health.
- **Phase-scaling** for single-GPU boxes (train, then serve, sequentially) so
  30B+ models fit where vLLM and the trainer can't co-reside.
- A **Phase-0 regression harness** for the migration (meta-device model-load +
  symbol-drift checks; 13 pass / 3 skip) used as the pre/post upgrade baseline.
- A consolidated **support matrix** (`docs/reports/supported-models.md`) plus 11
  ADRs capturing the durable decisions.

---

## Models I got running

**31 real models PASS end-to-end** (train + serve adapter + memorize on real
GPU) — 30 independently GPU-validated plus Qwen3.6-35B arch-covered by its
identical Qwen3.5 sibling — **plus 4 tiny-random smoke fixtures**:

- **Dense causal LM (23):** Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B),
  Qwen2-7B, Qwen3 (1.7B, 4B, 8B, 14B, 32B), Llama-3.2-1B/3B, Llama-3.1-8B,
  Mistral-7B, Phi-4, Phi-4-mini, OLMo-2-7B, gemma-3-4b, gemma-3-27b, rnj-1.
- **Multimodal / VLM (3):** Qwen2-VL-7B, Qwen2.5-VL-7B, pixtral-12b.
- **MoE — routed/grouped-expert LoRA (5):** Qwen3-30B-A3B, Qwen3.5-35B-A3B
  (multimodal MoE), Qwen3.6-35B-A3B (arch-covered), Mixtral-8x7B,
  granite-4.0-h-tiny (hybrid Mamba-2 + MoE).
- **Tiny-random fixtures (4):** tiny-random-llama, gemma-3-270m / -270m-it,
  qwen3-moe-tiny-random (validates the expert-LoRA converter).
- **Notable firsts:** first VLM PASS, first large-MoE PASS, first hybrid
  Mamba/MoE PASS, and **Qwen2.5-72B — the largest model validated end-to-end**
  (single-GPU device_map train at the H200 ceiling; serve at TP=2).

Beyond PASS: **3 models in-progress** (PhiMoE, OLMoE, Qwen1.5-MoE — MoE converter
gaps), **5 blocked** (gemma-4-dense adapter-key mapping; 3 custom-code VLMs on
transformers-5.x drift; a malformed synthetic fixture), and a **120B+ tier**
(gpt-oss-120b, Nemotron-3-Super-120B, Qwen3.5-122B) that is tooling-blocked on
single-GPU training, not just VRAM.

---

## Effort per model — tiered

The cost to get a model green fell into four clear buckets:

**Tier 1 — Config only (add to manifest, it just works).**
Most Qwen2.5/Qwen2/Llama dense models, tiny fixtures. Effort: minutes.

**Tier 2 — Hyperparameter / dtype tune (no code).**
The largest bucket of "near-misses." Recurring patterns: **fp32 → bf16** fixes
mode collapse (Mistral); **lr 3e-3 → 1e-3** fixes warmup overshoot on Qwen3 ≥4B,
Phi-4-mini, gemma-27b; **lr → 5e-4** for high-active-param MoE (Mixtral,
Qwen2.5-72B); **more steps** (450 → 900) for budget-starved VLMs. Effort: a
tuning run or two.

**Tier 3 — Fork / trainer code fix.**
Where the real engineering went. Representative fixes:
- **LoRA target resolution** for MoE, hybrid, and multimodal archs (exclude
  `router`/`mamba`/vision, target grouped-expert `gate_up_proj`/`down_proj` via
  `target_parameters`) — unblocked Qwen3-MoE, PhiMoE, OLMoE, granite, Qwen3.5-MoE.
- **Expert-LoRA converter** splitting fused PEFT weights for vLLM's 2D FusedMoE
  path — unblocked large-MoE serving.
- **Adapter key-prefix normalization** — fixed the causal-LM serve no-op and the
  gemma multimodal vision-tower prefix contamination (adapter silently dropped).
- **custom-code model loading** under transformers 5.x (trust_remote_code
  plumbing, tie_weights/context-length shims).
- Plus infra fixes: VRAM gating for unified memory, stale-job reconciler races,
  fresh `ml/` snapshots per run, image transformers/safetensors pinning.

**Tier 4 — Blocked (characterized, not a quick fix).**
GLM-4, InternVL3, Molmo — each walls out on **vendor-code vs transformers 5.x**
drift (Molmo is a genuine forward-compute bug, not a shim). gemma-4-dense fixture
is an adapter-key mapping gap. These are documented with the exact failure depth;
clean path is a natively-supported arch swap, not a patch.

---

## vLLM fork 0.19 → 0.25 migration

The fork = stock vLLM + a hybrid LoRA + Tokenformer `.pt`-adapter subsystem. The
upgrade ran as a 6-phase plan, all validated on GB10:

- **Phase 0–2:** built the regression harness; established the correct toolchain
  base (**NGC 26.04 / torch 2.12** — 26.03's torch lacked a required kwarg);
  toolchain gate GREEN (compile + import + symbol-drift).
- **Phase 3–4:** reconciled 67 fork files — **35 carried, 32 shed** (v0.25's
  native MoE-LoRA stacking is a superset of the fork's ~240-line carry-forward).
  VERIFY gate GREEN.
- **Phase 5–6:** hardware memorize pass — dense, multimodal, and MoE all PASS on
  the gate image; first full cray stack on 0.25; two glue drifts in
  `create_vllm.py` fixed.
- **Bonus:** the 0.25 base **unblocked DiffusionGemma** — implemented a
  decoder-only LoRA training path with two-pass self-conditioning and served it
  via the hybrid adapter. Validation is now root-caused (see next section): the
  adapter **learns perfectly** but the AR-native exact-hash gate mismeasures its
  iterative decode.

---

## DiffusionGemma: memorizes the objective, not the decode (update 2026-07-19)

Deep-diving the DiffusionGemma serve validation produced the most interesting
negative result of the campaign. The model reaches **NO_MEMORIZATION** through
the production vLLM path, but this is **not** an adapter-loading bug like the
MoE/multimodal cases — it is a **paradigm** property. Three findings:

- **The adapter learns perfectly.** A single joint teacher-forced forward
  reconstructs all 23 answer tokens from a fully-random 256-token canvas —
  **probe = 32/32, even at noise level t=1.0**. LoRA training, gradient flow, and
  adapter apply all work on this architecture. The finetuning machinery is proven.

- **The failure is decode, not training.** DiffusionGemma is the one family where
  training and serving are **different computations**. Every other family is
  autoregressive: teacher-forced training and greedy decode factor the sequence
  the same way, and a causal KV cache is semantically exact, so **memorize ⟹
  exact decode is effectively a theorem**. Diffusion trains a *denoising*
  objective (per-position marginal reconstruction of a corrupted canvas) but
  serves an *iterative ancestral sampler* (accept/renoise + self-conditioning +
  stability stop). There is no theorem linking the two — and I measured exactly
  that split: probe 32/32, iterative serve scrambles.

- **The memorized solution is a razor-thin, numerically fragile fixed point.** It
  survives only the exact training forward. The same bf16 paged-FlashAttention vs
  SDPA kernel gap that every family serves through is harmless to AR (large logit
  margin, per-token independent argmax) but **fatal to diffusion** (many
  near-tied positions in a joint bidirectional fixed point, and the iteration
  amplifies a tiny step-1 difference out of the basin). fp32 breaks it too
  (6/32), confirming it is fragility, not a precision setting. Forward-path
  ladder: training/offline joint-recompute **32/32** → HF `generate()` cache path
  **15/32** → vLLM paged attention **full scramble**.

**Consequence for the gate.** The exact-hash memorize gate is an **AR-native
instrument**: it conflates "learned the answer" (true here) with "decodes the
answer exactly" (false here). That mismatch is why DiffusionGemma uniquely trips
a gate that 31 other models pass. For validating *finetuning usefulness* on this
arch, the objective-level memorization (32/32) is the load-bearing evidence; a
paradigm-appropriate (fuzzy/task) metric is the right instrument, not bit-exact
hash.

**Fix options scoped, decision pending.** Three ways to close the production gap
were written up with trade-offs: **(A)** joint-recompute decode inside vLLM
(bit-faithful but fights the paged-KV design; high effort/risk), **(B)** serve via
the HF recompute path outside vLLM (proven 32/32 offline, but a second serving
stack), **(C, recommended)** make the memorization *robust* to the serve forward
via train-serve-consistent / unrolled training (owned code, no fork rebase tax,
composes with the shipped `supervise_termination` fix; needs validation). Plan:
pursue **C**, with **B** as a bridge if a green gate is needed first;
**deprioritize A**.

Full detail: `2026-07-16-diffusiongemma-validation-runs.md` (source read +
measurements), `2026-07-18-ar-vs-diffusion-train-serve.md` (paradigm explainer),
`2026-07-18-diffusiongemma-serve-fix-options.md` (options + recommendation).

---

## Where things stand

- **Support matrix is current and consolidated** — one source of truth for what
  works and what each model cost.
- **Migration is validated on hardware** but a few fixes remain uncommitted in
  ephemeral worktrees, to land at sign-off.
- **Open threads:** DiffusionGemma serve gap now **root-caused** (paradigm, not a
  bug) with fix options scoped — decision pending on Option C; Qwen1.5-MoE
  shared-expert converter gap (deferred); the 3 custom-code VLMs remain blocked
  pending arch swaps.

---

## Appendix — per-model roster

**Verdict:** PASS = trains + serves adapter + memorizes end-to-end · NO_MEM =
serves but can't reproduce · BLOCKED = characterized wall · UNTESTED = configured
target absent. **Effort tier:** T1 config-only · T2 hyperparameter/dtype tune ·
T3 fork/trainer code fix · T4 blocked/characterized.

### PASS (31 real models + 4 fixtures)

| Model | Arch | Verdict | Tier | What it took |
|---|---|---|---|---|
| Qwen2.5-0.5B | Qwen2 | PASS | T1 | CPU-runnable baseline |
| Qwen2.5-1.5B-Instruct | Qwen2 | PASS | T1 | — |
| Qwen2.5-3B-Instruct | Qwen2 | PASS | T1 | — |
| Qwen2.5-7B-Instruct | Qwen2 | PASS | T1 | — |
| Qwen2.5-14B-Instruct | Qwen2 | PASS | T2 | bf16, 450-step |
| Qwen2.5-32B-Instruct | Qwen2 (32B) | PASS | T2 | lr 1e-3, bf16, phased, H200 |
| Qwen2.5-72B-Instruct | Qwen2 (72B) | PASS | T2 | lr 5e-4; single-GPU train @142/143 GiB, **serve TP=2**. Largest validated |
| Qwen2-7B-Instruct | Qwen2 | PASS | T2 | 450-step (slower-converging) |
| Llama-3.2-1B-Instruct | Llama | PASS | T1 | gated (HF_TOKEN) |
| Llama-3.2-3B-Instruct | Llama | PASS | T1 | gated |
| Llama-3.1-8B-Instruct | Llama | PASS | T1 | gated |
| Mistral-7B-Instruct-v0.3 | Mistral | PASS | T2 | bf16 (fp32 mode-collapsed at 3e-3) |
| phi-4 | Phi3 (14B) | PASS | T2 | bf16, 450-step |
| Qwen3-8B | Qwen3 | PASS | T2 | bf16 |
| Qwen3-1.7B | Qwen3 | PASS | T1 | tolerates default lr 3e-3 |
| Qwen3-4B | Qwen3 | PASS | T2 | lr 1e-3 (3e-3 mode-collapsed → `6666…`) |
| Qwen3-14B | Qwen3 | PASS | T2 | bf16, lr 1e-3, 450-step |
| Qwen3-32B | Qwen3 (32B) | PASS | T2 | lr 1e-3, bf16, phased, H200 |
| gemma-3-4b-it | Gemma3 | PASS | T3 | cross-arch contamination fix (adapter registry scoping) |
| gemma-3-27b-it | Gemma3 mm (27B) | PASS | T3 | fork fix: layer-prefix picked vision tower → decoder-prefix preference |
| rnj-1-instruct | Gemma3 (8B) | PASS | T2 | bf16 |
| OLMo-2-1124-7B-Instruct | Olmo2 | PASS | T1 | — |
| Phi-4-mini-instruct | Phi3 (3.8B) | PASS | T2 | lr 1e-3 (3e-3 oscillated → NO_MEM) |
| Qwen2-VL-7B-Instruct | Qwen2-VL | PASS | T3 | multimodal LoRA/loader/mask support; fp32, 900-step. First VLM PASS |
| Qwen2.5-VL-7B-Instruct | Qwen2.5-VL | PASS | T2 | 900-step, train_timeout 4800 |
| pixtral-12b | Llava (12B) | PASS | T2 | bf16 + lr 5e-4 (end-of-warmup divergence) |
| Qwen3-30B-A3B-Instruct-2507 | Qwen3MoE | PASS | T3 | expert-LoRA converter + phase-scaling. First large-MoE |
| granite-4.0-h-tiny | GraniteMoeHybrid | PASS | T3 | resolver + preflight + lr fix. First hybrid Mamba-2+MoE |
| Qwen3.5-35B-A3B | Qwen3_5Moe mm-MoE | PASS | T3 | resolver `target_parameters` for grouped experts; lr 1e-3, H200 |
| Qwen3.6-35B-A3B | Qwen3_5Moe | PASS* | T3 | arch-covered by Qwen3.5 (not re-run) |
| Mixtral-8x7B-Instruct-v0.1 | Mixtral (47B) | PASS | T2 | loads grouped in transformers 5.12 → existing path, no converter; lr 5e-4, H200 |
| tiny-random-llama | Llama | PASS | T1 | CPU fixture |
| gemma-3-270m / -270m-it | Gemma3 | PASS | T1 | fixtures |
| qwen3-moe-tiny-random | Qwen3MoE | PASS | T1 | validates expert-LoRA converter (CPU) |

\* Qwen3.6-35B is arch-identical to Qwen3.5-35B; covered by that validation, not independently run.

### In progress / blocked / untested

| Model | Arch | Verdict | Tier | Blocker |
|---|---|---|---|---|
| Phi-mini-MoE-instruct | PhiMoE | NO_MEM | T3 | grouped experts unadapted (loss floored ~1.56); needs `target_parameters` — since resolved config-side, re-run pending |
| OLMoE-1B-7B-0924 | Olmoe | INCONCLUSIVE | T3 | memorized in train but attention-only; serve poll deadlock, cancelled |
| diffusiongemma-26B-A4B-it | DiffusionGemma (MoE+mm) | NO_MEM | T3 | **paradigm, not a bug**: adapter memorizes objective (probe 32/32) but iterative diffusion decode ≠ training forward; razor-thin bf16 fixed point scrambles under vLLM paged attention. Fix options scoped (A/B/**C**), decision pending |
| Qwen1.5-MoE-A2.7B-Chat | Qwen2MoE | NO_MEM | T3 | shared-expert mis-mapped by converter (routed-only). Deferred |
| gemma-4-dense (fixture) | Gemma4 mm | NO_MEM | T4 | `Gemma4ForConditionalGeneration` adapter key mapping |
| tiny-random-qwen2-vl (fixture) | Qwen2-VL | RESTART_FAILED | T4 | vLLM crash-loop on malformed synthetic config (low impact) |
| GLM-4-9B-Chat | ChatGLM | BLOCKED | T4 | vendor code vs transformers 5.x, 5 layers deep |
| InternVL3-8B | InternVL mm | BLOCKED | T4 | vendor code vs transformers 5.x, 4 layers deep |
| Molmo-7B-D-0924 | Molmo mm | BLOCKED | T4 | forward-compute tensor-broadcast bug (real modeling bug) |
| gpt-oss-120b / -20b | — | UNTESTED | T4 | 120B needs multi-GPU TP training (single-GPU load path only) |
| Nemotron-3-Super-120B | — | UNTESTED | T4 | 120B+ multi-GPU TP training |
| Qwen3.5-122B-A10B | Qwen3_5Moe | UNTESTED | T4 | 120B+ multi-GPU TP training |
