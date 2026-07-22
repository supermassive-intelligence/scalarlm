# AR vs diffusion: why dense/MoE/multimodal memorize on serve but DiffusionGemma doesn't

**Date:** 2026-07-18
**Author:** Georgi Georgiev
**Status:** Reference / explainer
**Context:** Companion to `2026-07-18-diffusiongemma-serve-fix-options.md` and
`2026-07-16-diffusiongemma-validation-runs.md`. Explains the *category* difference behind the DiffusionGemma
exact-hash serve gap, versus every other model family the finetune sweep validates.

---

## 1. One-line answer

Dense, MoE, and multimodal are all the **same paradigm — autoregressive (AR) causal LMs** — and for that
paradigm the **training forward and the serving forward are the same computation**, so memorization provably
transfers to exact greedy decode. DiffusionGemma is a **different paradigm** (iterative denoising over a
bidirectional canvas) where training and serving are *not* the same computation, so memorization does **not**
imply exact decode. Everything else follows from that.

---

## 2. The four families side by side

| | Training forward | Serving decode | memorize ⟹ exact decode? |
|---|---|---|---|
| **Dense** (Qwen2.5, Llama) | Teacher-forced next-token CE, causal mask, answer supervised | vLLM AR: prefill prompt → KV cache → emit 1 token/step (argmax) | **Yes — guaranteed** |
| **MoE** (Mixtral, OLMoE, PhiMoE) | Same as dense + routed-expert FFN; needs an expert-LoRA converter to *load* the adapter | Same AR decode; experts routed per token | **Yes**, once the adapter loads |
| **Multimodal** (Qwen2.5-VL, gemma-3) | Same as dense; vision-tower embeddings spliced into the prompt; needs the correct language-model prefix to *load* the adapter | Same AR decode over text; images are just prompt context | **Yes**, once the adapter loads |
| **DiffusionGemma** | **One joint pass**: prompt + fixed random-corrupted canvas, **bidirectional** mask, denoising objective at a random noise level t~U | **Iterative denoiser**: encode prompt → persistent bf16 KV → refine the whole canvas over N steps with a stochastic accept/renoise **sampler** + self-conditioning + stability stopping | **No** — memorizes the objective, does not decode exact |

**Key observation.** The first three differ from each other only in ways that affect *getting the LoRA adapter
to load and apply* — MoE needed an expert-weight converter (`_stack_moe_lora_weights_gated`), multimodal needed
the right decoder prefix (the gemma vision-tower-prefix fix). Those were the hard parts of the sweep, but they
are **adapter plumbing**, not decode paradigm. Once the adapter is live, all three **memorize**, because their
decode is identical AR. DiffusionGemma's adapter *also* loads and applies correctly (teacher-forced probe =
32/32); its failure is the **decode paradigm itself**.

Worth stressing: DiffusionGemma-26B-A4B is *also* MoE *and* multimodal internally (routed experts + a vision
tower). So the difference is specifically the **diffusion decode**, not "simple vs fancy architecture."

---

## 3. The deep reason: AR has train/serve equivalence by construction; diffusion does not

### Autoregressive — the objective *is* the inference procedure

The training objective factors the sequence probability exactly the way inference generates it:

$$p(x) = \prod_t p(x_t \mid x_{<t})$$

- Teacher-forced training computes each factor \(p(x_t \mid x_{<t})\); greedy serving samples each factor in the
  **same order** with the **same conditioning**.
- The KV cache is **semantically exact**: with a causal mask, position \(i\)'s K/V never depend on future
  tokens, so prefilling them once and reusing them during decode is mathematically identical to a full forward.
- Therefore a model that puts ≈all probability mass on the golden token at every teacher-forced step **must**
  greedily decode the golden string. **Memorization ⟹ exact decode is effectively a theorem here.**

### Diffusion — training and inference are different computations

- Training optimizes a **denoising** objective: per-position marginal reconstruction of a canvas corrupted at a
  random noise level. It never runs the decode loop.
- Serving runs an **iterative ancestral sampler** — accept low-entropy tokens, renoise the rest, repeat, with
  self-conditioning feedback and a stability/confidence stop. That procedure is a *different computation* from
  training and only *approximates* sampling from the implicitly-defined joint.
- There is **no theorem** that "the denoiser memorized the marginals ⟹ the iterative decode reproduces the
  exact string." The joint is implicit; the trajectory can converge to a different mode. That is exactly what
  we measured: the probe (the training computation) = 32/32, the iterative serve = scramble.

---

## 4. Why AR tolerates a bf16 kernel gap that *breaks* diffusion

This is the subtle part, and it is the crux of "why the difference."

vLLM serves **all four** families through paged FlashAttention — a different bf16 kernel from HF's SDPA. So the
numeric divergence found in DiffusionGemma's serve path (see the source-read section of the validation report)
**also exists** for dense/MoE/multimodal serving. Why don't they break?

- **AR greedy decode is robust to it.** A memorized AR model puts a *large* logit margin on the golden token at
  each step. Per-token, independent argmax, big margin → bf16 kernel noise never flips it.
- **Diffusion is fragile to it.** The answer emerges as a **joint bidirectional fixed point** in which many
  positions are simultaneously *near-tied* (2 positions flip under HF's cache path; more under paged
  attention). And the **iteration amplifies**: a tiny step-1 difference changes which tokens the entropy-bound
  sampler accepts, which changes step 2, and the trajectory walks out of the razor-thin basin.

So it is **not** that AR avoids the numeric difference — it is that AR's decode is **margin-robust and
non-iterative**, while diffusion's is **near-tied and iteratively compounding**. Same kernel gap, opposite
outcome.

Corroborating data point: **fp32 also breaks DiffusionGemma** (6/32), because the memorized fixed point is
razor-thin against *any* numeric change, not bf16 specifically. An AR model in fp32 would still decode the
golden exactly — again the robustness/fragility split, not a precision setting.

---

## 5. Infrastructure consequences

- Dense/MoE/multimodal share the **generic vLLM AR path** (prefill + KV cache + per-token sample) and the
  standard cross-entropy training step. DiffusionGemma needs an **entirely separate serve subsystem**
  (`DiffusionGemmaModelState` + `DiffusionSampler`, reusing the speculative-decode data plumbing with
  overloaded semantics) and a **separate training data path** (canvas tokenizer + `corrupt_canvas`) with its
  own step function (`_diffusion_training_step_accumulate` in `training_loop.py`).
- The **exact-hash memorize gate is an AR-native instrument.** It assumes deterministic greedy decode where
  memorize ⟹ exact. For AR that assumption holds. For diffusion it reads the model *through* a stochastic
  sampler, so it conflates "learned the answer" (true for DiffusionGemma) with "decodes the answer exactly"
  (false). That mismatch is why DiffusionGemma uniquely trips the gate while its adapter is, at the objective
  level, perfectly memorized.

---

## 6. Takeaway

The DiffusionGemma serve gap is **not** a regression relative to the other families and **not** an
adapter-loading bug like the MoE/multimodal cases. It is a structural property of the diffusion paradigm:
training and serving are different computations, the memorized solution is a numerically fragile joint fixed
point, and the AR-native exact-hash gate measures decode fidelity the paradigm does not guarantee. This is the
"why" underneath the three fix options in `2026-07-18-diffusiongemma-serve-fix-options.md` — and specifically
why Option C (make the memorization *robust* to the serve forward) targets the actual defect rather than the
symptom.
