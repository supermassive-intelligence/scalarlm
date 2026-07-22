# When Confidence Misleads: Suffix Anchoring and Anchor-Proximity Confidence Modulation

## Note — why it's here

**Search query:** `diffusion language model first tokens wrong early positions error remasking confidence decoding order fix`

**Explains why our `anchor_token` lever backfired.** Our DiffusionConfig has an
`anchor_token` (Tier-2) option that prepends a fixed never-corrupted anchor at canvas
position 0; our validation Run #R4 turned it on and *degraded* to 7/32. This paper documents
exactly that mechanism: inserting a suffix anchor helps response completion **but "introduces
local overconfidence near the anchor, causing anchor-adjacent tokens to be decoded too
early."** So our anchor made position-0's neighbors commit prematurely — the opposite of the
stable-left-neighbor we intended. Their fix (Suffix-Anchored Confidence Modulation) is
training-free: keep the anchor but *modulate confidence near it according to decoding
progress*, i.e. suppress premature anchor-adjacent commits — a serve-side knob worth trying
before abandoning the anchor idea.

---

## Raw material (verbatim)

**Title:** When Confidence Misleads: Suffix Anchoring and Anchor-Proximity Confidence Modulation for Diffusion Language Models
**Authors:** Jungwon Park, Jimyeong Kim, Jungmin Ko, Nojun Kwak, Wonjong Rhee
**Submitted:** 27 May 2026 · cs.CL
**arXiv:** https://arxiv.org/abs/2605.28181 · HTML: https://arxiv.org/html/2605.28181

> Diffusion language models decode text by iteratively denoising masked token sequences,
> making the choice of which positions to decode a central inference-time decision. Most
> training-free decoding strategies use model confidence for position selection, assuming that
> high-confidence positions are ready to be decoded. In this work, we revisit this assumption
> by studying when confidence misleads fully non-autoregressive (fully non-AR) decoding. EOT
> tokens can receive high confidence and cause incomplete generation; inserting a suffix anchor
> can mitigate this issue but introduces local overconfidence near the anchor, causing
> anchor-adjacent tokens to be decoded too early. To address these issues, we propose
> Suffix-Anchored Confidence Modulation, a simple training-free method that inserts a short
> suffix anchor to encourage response completion and modulates confidence near the anchor
> according to decoding progress. This preserves the response-completion benefit of suffix
> anchoring while reducing premature decoding of anchor-adjacent tokens. Across text-only
> reasoning, vision-language reasoning, and code-generation benchmarks, our method consistently
> improves confidence-based fully non-AR decoding, outperforms explicit EOT suppression, and
> preserves the parallel decoding advantage of fully non-AR generation.
