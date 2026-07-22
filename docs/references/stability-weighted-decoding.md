# Stability-Weighted Decoding for Diffusion Language Models

## Note — why it's here

**Search query:** `diffusion language model first tokens wrong early positions error remasking confidence decoding order fix`

A principled, plug-in serve-side fix for exactly our premature-commit problem. It shows
**a token's temporal instability — the KL divergence between its prediction distributions on
consecutive denoising steps — lower-bounds its mutual information with the remaining masked
context**, i.e. temporally unstable tokens are provably unsafe to unmask early. SWD is
training-free and "acts as a universal modulator for arbitrary score-based decoding policies,"
so it could wrap DiffusionGemma's existing sampler without retraining. Relevant because our
bad first-3 positions are precisely the ones that would flicker step-to-step; weighting the
unmask order by temporal stability is a concrete experiment against our 28/32 ceiling.

---

## Raw material (verbatim)

**Title:** Stability-Weighted Decoding for Diffusion Language Models
**Authors:** Yue Wu, Jian Huang
**Submitted:** 18 Apr 2026 · cs.CL, cs.LG
**arXiv:** https://arxiv.org/abs/2604.17068 · PDF: https://arxiv.org/pdf/2604.17068

> Diffusion large language models (dLLMs) enable parallel text generation by iteratively
> denoising a fully masked sequence, unmasking a subset of masked tokens at each step. Existing
> decoding strategies rely on static confidence metrics computed at a single denoising step,
> ignoring temporal history and often leading to premature unmasking of unstable tokens. In
> this work, we theoretically establish that a token's temporal instability, quantified by the
> KL divergence between consecutive prediction distributions, provides a strict lower bound on
> its mutual information with the remaining masked context, indicating that temporally unstable
> tokens are inherently unsafe to unmask. Based on this insight, we propose Stability-Weighted
> Decoding (SWD), a training-free, plug-and-play strategy that incorporates temporal stability
> into token scoring and acts as a universal modulator for arbitrary score-based decoding
> policies. Experiments on code generation and mathematical reasoning benchmarks demonstrate
> that SWD consistently improves generation accuracy across representative scoring metrics and
> selection policies, and exhibits exceptional robustness, maintaining a significant
> performance lead over standard baselines across varying acceleration ratios.
