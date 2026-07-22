# Blockwise SFT for Diffusion Language Models

## Note — why it's here

**Search query:** `block diffusion language model fine-tuning self-conditioning instability remasking convergence`

Bears on our **training/inference alignment** and on our `self_conditioning_prob`. The paper's
thesis: classic SFT that randomly masks tokens across the *whole* response misaligns with
semi-autoregressive block decoding, producing "noisy prefixes and leaky suffixes" that bias
gradients — plausibly a contributor to our unstable basin. Their fix (supervise only the
active block, freeze the prefix, hide the future) is the disciplined version of what our
canvas masking approximates. It also surveys the **self-conditioning schedule** literature —
noting the "two-step loss with *gradually increased* self-conditioning" (Asada & Miwa, 2025)
— which contrasts with our **fixed `self_conditioning_prob=0.5`**; a ramped SC schedule is a
cheap thing to try. Consistent gains on GSM8K/MATH/MetaMathQA under equal compute.

---

## Raw material (verbatim)

**Title:** Blockwise SFT for Diffusion Language Models: Reconciling Bidirectional Attention and Autoregressive Decoding
**Authors:** Bowen Sun, Yujun Cai, Ming-Hsuan Yang, Yiwei Wang
**Submitted:** 27 Aug 2025 (v2: 23 Oct 2025) · cs.CL
**arXiv:** https://arxiv.org/abs/2508.19529 · HTML: https://arxiv.org/html/2508.19529

> Discrete diffusion language models have shown strong potential for text generation, yet
> standard supervised fine-tuning (SFT) misaligns with their semi-autoregressive inference:
> training randomly masks tokens across the entire response, while inference generates
> fixed-size blocks sequentially. This mismatch introduces noisy prefixes and leaky suffixes,
> biasing gradients away from the desired blockwise likelihood. We propose Blockwise SFT, which
> partitions responses into fixed-size blocks, selects one active block per step for stochastic
> masking, freezes all preceding tokens, and fully hides future ones. Loss is computed only
> over the active block, directly mirroring the blockwise decoding process. Experiments on
> GSM8K, MATH, and MetaMathQA show consistent gains over classical SFT under equal compute or
> token budgets. Block size consistency studies and ablations confirm that improvements stem
> from faithful training-inference alignment rather than incidental masking effects. Our
> results highlight the importance of matching supervision granularity to the decoding
> procedure in diffusion-based language models.
