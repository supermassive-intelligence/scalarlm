# Growing LMs: small → large via function-preserving initialization

## Note — why it's here

Collected while pinning down what **"knowledge inheritance"** means in
[LLaDA2.0](llada2-scaling-stabilizers.md). That paper's "knowledge inheritance" is *same
size, change the objective* (AR MoE → dLLM). This file is the **other** inheritance axis —
*same objective, change the size*: initialize a **larger** model from a **smaller**
pretrained one via function-preserving growth, instead of training the big model from
scratch. Two lineages that are easy to conflate; keep them separate.

The naming-collision paper lives here: **Qin et al. 2021, "Knowledge Inheritance for
Pre-trained Language Models"** owns the exact phrase but is model-*growth* (small→large
pretraining), **not** AR→diffusion — and LLaDA2.0 does not cite it.

Not (yet) load-bearing on a DiffusionGemma run — this is background for the inheritance
vocabulary. If we ever scale the diffusion adapter recipe across model sizes, the
function-preserving operators here (Net2Net → bert2BERT → LiGO/MSG/LEMON) are the toolkit.

---

## Raw material (verified on Semantic Scholar, counts as of 2026-07-19)

| Paper | Year | Cites | Idea |
|---|---|---|---|
| **Net2Net: Accelerating Learning via Knowledge Transfer** (Chen, Goodfellow, Shlens) | 2015 | 750 | Origin — function-preserving **widen (Net2Wider) / deepen (Net2Deeper)** transforms; new net starts as an exact function-equivalent of the small one. |
| **Efficient Training of BERT by Progressively Stacking** (Gong et al., ICML) | 2019 | — | Canonical **depth-growth**: duplicate-and-stack existing layers to double depth mid-training. |
| **On the Transformer Growth for Progressive BERT Training (CompoundGrow)** (Gu et al.) | 2020 | 61 | Grow **width + depth + input length jointly** on a schedule; generalizes progressive stacking. |
| **Knowledge Inheritance for Pre-trained Language Models** (Qin et al.) | 2021 | 65 | Owns the phrase "knowledge inheritance"; small→large pretraining via a KD-style inheritance loss. **Naming collision with LLaDA2.0 — not the same idea.** |
| **bert2BERT: Towards Reusable Pretrained Language Models** (Chen et al., ACL) | 2021 | 86 | Net2Net generalized to Transformers — reuse a small model's weights to initialize a wider+deeper one. |
| **Staged Training for Transformer Language Models** (Shen et al.) | 2022 | 55 | Function-preserving growth operators applied on a **compute-optimal** schedule. |
| **Learning to Grow Pretrained Models for Efficient Transformer Training (LiGO)** (Wang et al.) | 2023 | 91 | **Learns** the linear growth operator (a data-driven map from small→large params) instead of hand-designing it. |
| **Masked Structural Growth (MSG) — 2× Faster LM Pre-training** (Yao et al.) | 2023 | 32 | Strictly function-preserving structural growth; used to train **FLM-101B**. |
| **LEMON: Lossless Model Expansion** (Wang et al.) | 2023 | 25 | Lossless width/depth expansion paired with a tuned LR schedule. |

## Two inheritance axes (why they get conflated)

| | Change what | Change what stays | Lineage |
|---|---|---|---|
| **LLaDA2.0 "knowledge inheritance"** | the *objective* (AR → diffusion) | the size | Scaling Diffusion LMs via Adaptation from AR → Dream 7B → SDAR |
| **Model growth (this file)** | the *size* (small → large) | the objective | Net2Net → bert2BERT → LiGO / MSG / LEMON |
| **Qin et al. 2021 (the phrase)** | the *size* (small → large) | the objective | sits in the model-growth column, **not** AR→diffusion |
