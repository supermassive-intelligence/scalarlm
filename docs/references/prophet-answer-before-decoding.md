# Diffusion Language Models Know the Answer Before Decoding (Prophet)

## Note — why it's here

**Search query:** `diffusion language model first tokens wrong early positions error remasking confidence decoding order fix`

Included as the counter-evidence that **more refinement steps are not our lever.** Prophet
shows the correct answer is often internally present by *half* the steps (up to 97%/99% of
GSM8K/MMLU instances decodable at half budget), and early-commits when the top-2 confidence
gap is wide. For us this reframes the ceiling: our first-3-positions error is not "needs more
denoising" — it's a *convergence-basin/decoding-order* problem, matching our report's
"structural, not capacity-bound" conclusion. Also a useful serve-speed technique (up to 3.4×
fewer steps) for DiffusionGemma once quality is fixed. Validated on LLaDA-8B and Dream-7B.

---

## Raw material (verbatim)

**Title:** Diffusion Language Models Know the Answer Before Decoding
**Authors:** Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Soroush Vosoughi, Shiwei Liu
**Submitted:** 27 Aug 2025 (v5: 9 Apr 2026) · cs.CL, cs.AI
**arXiv:** https://arxiv.org/abs/2508.19982 · PDF: https://arxiv.org/pdf/2508.19982

> Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive
> approaches, offering parallel sequence generation and flexible token orders. However, their
> inference remains slower than that of autoregressive models, primarily due to the cost of
> bidirectional attention and the large number of refinement steps required for high quality
> outputs. In this work, we highlight and leverage an overlooked property of DLMs early answer
> convergence: in many cases, the correct answer can be internally identified by half steps
> before the final decoding step, both under semi-autoregressive and random remasking
> schedules. For example, on GSM8K and MMLU, up to 97% and 99% of instances, respectively, can
> be decoded correctly using only half of the refinement steps. Building on this observation,
> we introduce Prophet, a training-free fast decoding paradigm that enables early commit
> decoding. Specifically, Prophet dynamically decides whether to continue refinement or to go
> "all-in" (i.e., decode all remaining tokens in one step), using the confidence gap between
> the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM
> implementations, incurs negligible overhead, and requires no additional training. Empirical
> evaluations of LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the
> number of decoding steps by up to 3.4x while preserving high generation quality. These
> results recast DLM decoding as a problem of when to stop sampling, and demonstrate that early
> decode convergence provides a simple yet powerful mechanism for accelerating DLM inference,
> complementary to existing speedup techniques.
