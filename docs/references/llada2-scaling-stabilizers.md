# LLaDA2.0: Scaling Up Diffusion Language Models to 100B

## Note — why it's here

**Search query:** `block diffusion language model fine-tuning self-conditioning instability remasking convergence`

Largest, most production-oriented dLLM recipe we found (16B and 100B MoE, AR→dLLM
conversion). Named here for the concrete **convergence stabilizers we do NOT currently use**
in our corruption/SFT: from the HTML — **complementary masking + a mask-ratio bandwidth**
during SFT to improve sample efficiency and stabilize convergence, plus an **auxiliary
confidence loss** to sharpen predictions for parallel decoding. Also a **3-phase block-level
WSD schedule** (warm-up: progressively increasing block size → stable: full-sequence
diffusion → decay: compact block size) — a schedule idea for our canvas/corruption that our
flat `U(eps,1)` sampler and fixed `canvas_length=256` don't exploit. Both are MoE dLLMs, same
family shape as DiffusionGemma.

---

## On "knowledge inheritance"

The abstract lists three design principles for the AR→dLLM conversion:
**knowledge inheritance, progressive adaption, efficiency-aware design**. "Knowledge
inheritance" here is **not a cited method** — it is the authors' framing for their core
move: start the dLLM from a *pretrained AR MoE's* weights and adapt them, instead of
training a diffusion LM from scratch. The 3-phase block-level WSD schedule (warm-up →
stable → decay) is the mechanism that carries that inherited knowledge across the AR→
diffusion objective shift.

**Do not conflate** with the separately-named paper *"Knowledge Inheritance for
Pre-trained Language Models"* (Qin et al., 2021), about accelerating LM *pretraining* from
smaller models — LLaDA2.0 does **not** cite it and uses the phrase descriptively. That
paper and the whole small→large *model-growth* lineage (a different inheritance axis:
*same objective, change the size*) are collected in
[model-growth-inheritance.md](model-growth-inheritance.md).

Lineage inside LLaDA2.0's own 69-ref list (the AR→diffusion adaptation cluster):

| Paper | Year | Why it's the lineage |
|---|---|---|
| **Scaling Diffusion LMs via Adaptation from Autoregressive Models** (DiffuGPT/DiffuLLaMA) | 2024 | The seminal one — adapt/convert AR → diffusion, avoid *from scratch*; direct ancestor of "knowledge inheritance". |
| **Dream 7B** | 2025 | dLLM *initialized from* Qwen AR weights — inheritance in practice. |
| **Dream-Coder 7B** | 2025 | Same adaptation recipe for code. |
| **SDAR: Synergistic Diffusion-AutoRegression** | 2025 | Explicit AR-to-diffusion adaptation paradigm. |
| **Block Diffusion: Interpolating Between AR and Diffusion LMs** | 2025 | Underpins the *block-level* WSD schedule that carries the inheritance. |
| **Diffusion LLMs Faster-Than-AR (Discrete Diffusion Forcing)** | 2025 | Distill-from-pretrained-AR variant. |

The single must-read for the term is *Scaling Diffusion LMs via Adaptation from
Autoregressive Models* (2024); everything else builds on it.

*(Semantic Scholar: paperId `6bb21f7ceb85ae29ed5b24a736694730abd5d82f`, 69 refs / 114
citations as of 2026-07-19. No citing paper yet adopts "knowledge inheritance" as a named
method.)*

---

## Raw material (verbatim)

**Title:** LLaDA2.0: Scaling Up Diffusion Language Models to 100B
**Authors:** Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, … Jianguo Li, et al.
**Submitted:** 10 Dec 2025 (v2: 24 Dec 2025) · cs.LG, cs.AI, cs.CL
**arXiv:** https://arxiv.org/abs/2512.15745 · PDF: https://arxiv.org/pdf/2512.15745

> This paper presents LLaDA2.0 -- a tuple of discrete diffusion large language models (dLLM)
> scaling up to 100B total parameters through systematic conversion from auto-regressive (AR)
> models -- establishing a new paradigm for frontier-scale deployment. Instead of costly
> training from scratch, LLaDA2.0 upholds knowledge inheritance, progressive adaption and
> efficiency-aware design principle, and seamless converts a pre-trained AR model into dLLM
> with a novel 3-phase block-level WSD based training scheme: progressive increasing block-size
> in block diffusion (warm-up), large-scale full-sequence diffusion (stable) and reverting back
> to compact-size block diffusion (decay). Along with post-training alignment with SFT and DPO,
> we obtain LLaDA2.0-mini (16B) and LLaDA2.0-flash (100B), two instruction-tuned Mixture-of-
> Experts (MoE) variants optimized for practical deployment. By preserving the advantages of
> parallel decoding, these models deliver superior performance and efficiency at the frontier
> scale. Both models were open-sourced.
