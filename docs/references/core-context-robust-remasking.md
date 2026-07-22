# CoRe: Context-Robust Remasking for Diffusion Language Models

## Note — why it's here

**Search query:** `diffusion language model first tokens wrong early positions error remasking confidence decoding order fix`

Names our failure mode precisely: **"context rigidity" — tokens are retained on transient high
confidence, ignoring that early predictions lack full context, creating cascade effects where
initial inconsistencies misguide the rest.** That is our first-3-positions-wrong / `cccc`
cascade almost verbatim. Its second key claim matters for our tuning: **static confidence is
myopic — "inconsistent tokens can appear confident to the model itself,"** so a naive
confidence-remask won't rescue us. CoRe is training-free and identifies *context-brittle*
tokens by probing sensitivity to masked-context perturbations, then prioritizes them for
revision. A serve-side lever to prototype against our diffusion sampler.

---

## Raw material (verbatim)

**Title:** CORE: Context-Robust Remasking for Diffusion Language Models
**Authors:** Kevin Zhai, Sabbir Mollah, Zhenyi Wang, Mubarak Shah
**Submitted:** 4 Feb 2026 (v3: 13 Mar 2026) · cs.LG
**arXiv:** https://arxiv.org/abs/2602.04096 · HTML: https://arxiv.org/html/2602.04096v1

> Standard decoding in Masked Diffusion Models (MDMs) is hindered by context rigidity: tokens
> are retained based on transient high confidence, often ignoring that early predictions lack
> full context. This creates cascade effects where initial inconsistencies misguide the
> remaining generation. Existing revision strategies attempt to mitigate this by relying on
> static confidence scores, but these signals are inherently myopic; inconsistent tokens can
> appear confident to the model itself. We propose Context-Robust Remasking (CORE), a
> training-free framework for inference-time revision. Rather than trusting static token
> probabilities, CORE identifies context-brittle tokens by probing their sensitivity to
> targeted masked-context perturbations. We formalize revision as a robust optimization
> objective over context shifts and efficiently approximate this objective to prioritize
> unstable tokens for revision. On LLaDA-8B-Base, CORE delivers consistent improvements across
> reasoning and code benchmarks, outperforming compute-matched baselines and improving MBPP by
> up to 9.2 percentage points.
