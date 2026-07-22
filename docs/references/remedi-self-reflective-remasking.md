# RemeDi: Don't Settle Too Early — Self-Reflective Remasking for Diffusion Language Models

## Note — why it's here

**Search query:** `block diffusion language model fine-tuning self-conditioning instability remasking convergence`
(also surfaced by query 3 on early-position errors)

Directly targets our headline failure: **once a token is generated it typically stays fixed**,
so the wrong first-3 canvas positions in our Run #4 can never be revised by the standard
decode. RemeDi trains the model to **jointly predict token distributions and per-token
confidence**, then remask low-confidence tokens so they resample with richer context in later
steps. Relevant to us because the fix is *partly training-side* (a remask-aware SFT + RL
pipeline) — i.e. it could plug into our `diffusion_corruption.py` / training loop, not only
the serve sampler. Reports SOTA among open-source DLMs. Convergence note from the HTML: Remask
RL improved early-training accuracy (80.00% vs 77.58% at step 50) and final accuracy (83.33%).

---

## Raw material (verbatim)

**Title:** Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models
**Authors:** Zemin Huang, Yuhang Wang, Zhiyang Chen, Guo-Jun Qi
**Submitted:** 28 Sep 2025 · cs.CL
**arXiv:** https://arxiv.org/abs/2509.23653 · HTML: https://arxiv.org/html/2509.23653v1

> Mask-based Diffusion Language Models (DLMs) struggle to revise incorrect tokens: once a
> token is generated, it typically remains fixed. The key challenge is to identify potential
> errors in the inputs. In this paper, we propose Remasking-enabled Diffusion Language Model
> (RemeDi), a mask-based DLM that introduces remasking as another fundamental mechanism,
> enabling more flexible text refinement in diffusion-based text generation. To achieve this,
> RemeDi jointly predicts token distributions and per-token confidence scores at each step.
> The confidence scores determine which tokens to be unmasked after the current step, allowing
> the model to identify tokens with low quality and remask them. These remasked tokens can be
> resampled with richer context in subsequent steps. We design a remask-aware pipeline to
> train this ability, including supervised fine-tuning which teaches the model to detect and
> remask incorrect tokens in addition to predict mask tokens, and reinforcement learning which
> optimizes full generation trajectories toward higher rewards. Experiments show that RemeDi
> achieves the state-of-the-art results among open-source DLMs on multiple datasets.
