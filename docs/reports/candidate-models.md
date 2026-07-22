# Candidate Models for Finetune Sweep

This document tracks promising models and architectures identified for future validation in the finetune sweep, focusing on closing architectural gaps.

## 🚀 High Priority: Architectural Gaps

### DeepSeek Lineage (MLA & Shared-Expert MoE)
- **Model:** `deepseek-ai/DeepSeek-V2-Lite`
- **Reason:** Target for validating **Multi-head Latent Attention (MLA)** and **Shared-Expert MoE** patterns.
- **Goal:** Verify that the expert-LoRA converter and `hybrid_adapter_manager` correctly handle latent-attention keys and shared-expert weight mapping.
- **Status:** Identified; pending configuration in `finetune-sweep.yaml`.

## 🔍 Secondary Targets (Architectural Diversity)

### High-Capability Dense/Reasoning
- **Model:** `allenai/Olmo-3-32B-Think`
- **Reason:** Open-weight reasoning model with high density.
- **Goal:** Validate memorization performance and serving efficiency on 2xH100.

### Linear-Complexity / Recurrents
- **Target Architectures:** Block-transformer hybrids (e.g., **Jamba-2**) and pure RNN-style bases (**RWKV-7**).
- **Goal:** Verify long-context LoRA and non-transformer base support.

### Native Multimodal (Omni)
- **Target:** Models with fully integrated modalities (vs. Encoder $\rightarrow$ Projector $\rightarrow$ LLM).
- **Goal:** Test if `hybrid_adapter_manager` layer-prefix detection holds for native multimodal streams.

### High-Efficiency Small Models
- **Target:** **Samba** or **Sleek** architectures.
- **Goal:** Validate mixed linear/non-linear layer adaptation.
