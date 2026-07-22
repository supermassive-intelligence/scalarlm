# On the Trainability of Masked Diffusion Language Models via Blockwise Locality

## Note — why it's here

**Search query:** `block diffusion language model fine-tuning self-conditioning instability remasking convergence`

Names the instability we hit: our good basin (runs #3/#4) is a **minority draw** and three
fresh reruns collapsed into the `cccc` attractor. This paper studies MDM optimization
stability directly and finds **standard random-masking MDMs are substantially less stable**
than AR-LLMs, with high-variance training dynamics; the diagnosed bottleneck is missing
**intra-block autoregressive locality**. Their remedy (Jigsaw/Scatter — inject left-to-right
inductive bias *within* blocks while keeping iterative refinement *across* blocks) is a
conceptual match for our anchor/protect-prefix intuition (give early positions a stable
left-context), and suggests our uniform-vocab random corruption may itself be a suboptimal
instantiation for ordered canvas generation.

---

## Raw material (verbatim)

**Title:** On the Trainability of Masked Diffusion Language Models via Blockwise Locality
**Authors:** Yuxiang Wang, Yu Xiang, Baojian Zhou, Qifang Zhao, Keyue Jiang, Yanghua Xiao, Xiaoxiao Xu
**Submitted:** 27 Apr 2026 · cs.LG, cs.AI
**arXiv:** https://arxiv.org/abs/2604.24832 · HTML: https://arxiv.org/html/2604.24832v1

> Masked diffusion language models (MDMs) have recently emerged as a promising alternative to
> standard autoregressive large language models (AR-LLMs), yet their optimization can be
> substantially less stable. We study blockwise MDMs and compare them with AR-LLMs on three
> controlled tasks that stress different aspects of structured generation: in-context linear
> regression, graph path-finding, and Sudoku solving. We find that standard random-masking
> MDMs fail to reliably learn linear regression, exhibit high variance training dynamics on
> graph path-finding, while outperforming AR-LLMs on Sudoku. To mitigate these instabilities,
> we propose two locality aware blockwise models, namely Jigsaw and Scatter, that inject
> left-to-right inductive bias by enforcing autoregressive locality within blocks while
> preserving iterative refinement at the block level. Empirically, Jigsaw matches AR-LLM
> stability on linear regression and remains strong on Sudoku, while Scatter retains
> diffusion's planning advantage on path-finding. Our results indicate that standard
> random-masking MDMs, even with blockwise variants, may be a suboptimal instantiation of
> diffusion LMs for ordered generation, motivating models beyond random masking.
