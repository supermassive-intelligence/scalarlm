# External references — diffusion-LLM LoRA fine-tuning & decoding

Raw materials (abstracts + extracted numbers) from other people's experiments on
diffusion / masked-diffusion / block-diffusion LLMs, collected while investigating
the **DiffusionGemma decoder-only LoRA** recipe and its validation ceiling.

Each `*.md` here carries: a **note** (why it's included + which of our runs it bears
on) followed by the **raw material** (verbatim arXiv title/authors/date/abstract, plus
any concrete hyperparameters we could extract). Companion to
`docs/reports/2026-07-16-diffusiongemma-validation-runs.md` and ADR 0007.

## Search queries that produced these

1. `LoRA fine-tuning diffusion large language model LLaDA Dream discrete diffusion parameters rank`
2. `block diffusion language model fine-tuning self-conditioning instability remasking convergence`
3. `diffusion language model first tokens wrong early positions error remasking confidence decoding order fix`

Date collected: 2026-07-16.

## Why these are here — mapping to our findings

Our validation report's core result: decoder-only LoRA (r16/α32, freeze router+experts,
dense-MLP carries memorization) trains cleanly (loss→1e-5) but tops out at **28/32** with
the **first 3 canvas positions wrong, tail exact**, and the good basin is unstable (the
`cccc`-collapse attractor on rerun). The literature below sorts into three buckets against
that result.

### Bucket A — training-side LoRA (does our recipe match the field?)

| File | Paper | Bears on |
|---|---|---|
| [nara-noise-aware-lora.md](nara-noise-aware-lora.md) | NaRA: Noise-Aware LoRA for dLLMs | **Independently benchmarks DiffusionGemma-26B with LoRA r64/α128, attn+shared-MLP, freeze experts/router/vision — corroborates our architecture but 4× our rank/alpha.** Argues static LoRA is noise-agnostic → weak at high noise. The one credible training-side upgrade. |

### Bucket B — the "early positions wrong" ceiling is a decoding artifact (serve-side fixes)

| File | Paper | Bears on |
|---|---|---|
| [remedi-self-reflective-remasking.md](remedi-self-reflective-remasking.md) | RemeDi: Self-Reflective Remasking | Once a token is wrong it stays wrong; train the model to remask+resample low-confidence tokens. Directly targets our first-3-positions failure. |
| [core-context-robust-remasking.md](core-context-robust-remasking.md) | CoRe: Context-Robust Remasking | "Context rigidity": early predictions lack context, confidence is myopic. Training-free; probe context-brittleness instead of trusting probs. |
| [stability-weighted-decoding.md](stability-weighted-decoding.md) | Stability-Weighted Decoding | Temporally unstable tokens (KL between consecutive steps) are unsafe to unmask early — plug-in sampler modulator. |
| [prophet-answer-before-decoding.md](prophet-answer-before-decoding.md) | DLMs Know the Answer Before Decoding (Prophet) | Correct answer often present by half the steps; early-commit on top-2 gap. Explains why more refinement isn't the lever. |
| [suffix-anchoring-confidence.md](suffix-anchoring-confidence.md) | When Confidence Misleads: Suffix Anchoring | **Explains why our `anchor_token` lever backfired (#R4):** an anchor causes local overconfidence → anchor-adjacent tokens decode too early. |

### Bucket C — training instability & SFT/decoding alignment (the `cccc` collapse)

| File | Paper | Bears on |
|---|---|---|
| [blockwise-locality-trainability.md](blockwise-locality-trainability.md) | On the Trainability of MDMs via Blockwise Locality | Random-masking MDM SFT "remains unstable"; intra-block autoregression is the bottleneck. Matches our unstable good-basin. |
| [llada2-scaling-stabilizers.md](llada2-scaling-stabilizers.md) | LLaDA2.0 (100B) | Stabilizers we don't use: complementary masking + mask-ratio bandwidth + auxiliary confidence loss; 3-phase block-size WSD schedule. |
| [blockwise-sft.md](blockwise-sft.md) | Blockwise SFT | Random full-response masking misaligns with blockwise decode; supervise only the active block. Also cites the *ramped* self-conditioning schedule (vs our fixed 0.5). |

### Terminology — the two "inheritance" axes

| File | Papers | Bears on |
|---|---|---|
| [model-growth-inheritance.md](model-growth-inheritance.md) | Net2Net, bert2BERT, LiGO, MSG, LEMON, Qin et al. 2021 | Pins down what LLaDA2.0 means by "knowledge inheritance" (AR→dLLM, *same size*) vs. the small→large *model-growth* lineage (*same objective*); resolves the Qin-et-al. naming collision. |

## Takeaway encoded here

The field says our ceiling is a **decoding** problem (Bucket B), not a LoRA-capacity one —
consistent with r32 (our #9) making things *worse*. NaRA (Bucket A) is the only training-side
lever with a claim on dLLMs specifically; Bucket C names concrete corruption-schedule
stabilizers for the `cccc` instability.
