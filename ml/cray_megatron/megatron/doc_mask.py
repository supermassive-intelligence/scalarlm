"""Decide how to handle packed-document attention masking per batch/model.

The trainer packs short documents into one block and builds a 4D
block-diagonal+causal attention mask ``[B, 1, S, S]`` so packed documents don't
attend across each other (see ``pack()`` in
``dataset/load_language_model_dataset.py`` and ``training_step_accumulate`` in
``training_loop.py``). Plain text decoders (Llama, Qwen, Gemma3ForCausalLM)
accept the 4D mask.

But some models can't consume the 4D mask:

- ``...ForConditionalGeneration`` wrappers compute their loss internally and index
  the logits by a *2D* attention_mask. Gemma3ForConditionalGeneration does
  ``shift_logits[shift_attention_mask != 0]`` (transformers ``modeling_gemma3.py``);
  handed a 4D mask that index is 4D and raises ``IndexError: too many indices for
  tensor of dimension 3`` — the gemma-3-4b-it TRAIN_FAILED in the cuda-spark
  fine-tune sweep.
- Hybrid state-space (Mamba/SSM) models run a kernel-less ``torch_forward`` mixer
  (no ``mamba_ssm``/``causal_conv1d`` on this aarch64 box) that "tunes out" pad
  tokens with ``hidden_states * attention_mask[:, :, None]``, assuming a *2D*
  padding mask. NemotronH's ``_update_mamba_mask`` forwards whatever mask the model
  was called with straight to that path, so a 4D mask broadcasts to 5D and raises
  ``The size of tensor a (4096) must match the size of tensor b (496) at
  non-singleton dimension 4`` — the Nemotron-H-8B TRAIN_FAILED.

For those models we fall back to the 2D padding mask. The documented cost is that
packed documents attend across each other (identical to the existing seq-len-cap
fallback); for the single-document memorization sweep the 4D mask is a no-op anyway.
"""

# Decision outcomes for doc_mask_decision().
BUILD = "build"                    # construct the 4D block-diagonal+causal mask
SKIP_MULTIMODAL = "skip_multimodal"  # wrapper masks loss by a 2D mask; keep 2D
SKIP_SSM = "skip_ssm"              # hybrid Mamba/SSM mixer needs a 2D mask; keep 2D
SKIP_SEQLEN = "skip_seqlen"        # mask too large to materialize; keep 2D
NONE = "none"                      # batch isn't packed (no document_ids)

# Config ``model_type``s of the hybrid Mamba/SSM families the sweep has seen whose
# kernel-less mixer path needs a 2D mask. A per-layer block-type list
# (``layers_block_type``/``hybrid_override_pattern``: NemotronH / Jamba / Bamba) is
# the most reliable signal; this set catches families that don't expose one
# (FalconH1's Mamba-2 ‖ attention, GraniteMoeHybrid).
_SSM_MODEL_TYPES = {
    "nemotron_h",
    "falcon_h1",
    "granitemoehybrid",
    "bamba",
    "zamba2",
    "jamba",
}


def is_multimodal(model_config) -> bool:
    """True for HF multimodal wrapper configs (Gemma3/Qwen2-VL/Gemma4, …) — they
    nest a ``vision_config``. A ``vision_config`` attribute that is present but
    ``None`` reads as not-multimodal."""
    if model_config is None:
        return False
    return getattr(model_config, "vision_config", None) is not None


def is_diffusion(model_config) -> bool:
    """True for DiffusionGemma configs (``DiffusionGemmaForBlockDiffusion``), the
    discrete-diffusion encoder-decoder MoE. Detected by ``model_type`` rather than
    a wrapper attribute because a diffusion config ALSO carries a ``vision_config``
    (so ``is_multimodal`` is likewise True) and its own ``model_type`` is the only
    field that distinguishes it — the load path must check this *before* the
    multimodal/causal fork so it isn't misrouted to ``AutoModelForImageTextToText``.
    See ``docs/superpowers/specs/2026-07-01-diffusiongemma-design.md`` and ADR 0007."""
    if model_config is None:
        return False
    return getattr(model_config, "model_type", None) == "diffusion_gemma"


def has_ssm_layers(model_config) -> bool:
    """True for hybrid state-space (Mamba/SSM) model configs whose kernel-less
    ``torch_forward`` mixer multiplies hidden states by a *2D* padding mask
    (``hidden_states * attention_mask[:, :, None]``) and so CANNOT consume the 4D
    block-diagonal doc mask — handed a 4D mask it broadcasts to 5D and raises
    ``... at non-singleton dimension 4`` (the Nemotron-H-8B TRAIN_FAILED).

    Detected from the config alone (the model isn't available here): a per-layer
    block-type list naming a mamba block (NemotronH ``layers_block_type`` / Jamba /
    Bamba), a ``hybrid_override_pattern`` marker, or a known hybrid ``model_type``
    (``_SSM_MODEL_TYPES``)."""
    if model_config is None:
        return False
    block_types = getattr(model_config, "layers_block_type", None)
    if block_types and any("mamba" in str(bt).lower() for bt in block_types):
        return True
    if getattr(model_config, "hybrid_override_pattern", None):
        return True
    model_type = getattr(model_config, "model_type", None) or ""
    return model_type in _SSM_MODEL_TYPES


def doc_mask_decision(batch, seq_len: int, model_config, max_4d_mask_seq_len: int) -> str:
    """Return how to handle packed-document attention for this batch:

    - ``NONE``            — the batch isn't packed (no ``document_ids``).
    - ``SKIP_MULTIMODAL`` — a multimodal wrapper that masks its loss by a 2D
      attention_mask; a 4D mask would break that index. Keep the 2D mask.
    - ``SKIP_SSM``        — a hybrid Mamba/SSM model whose kernel-less mixer needs
      a 2D mask; a 4D mask broadcasts to 5D and crashes. Keep the 2D mask.
    - ``SKIP_SEQLEN``     — the mask would exceed ``max_4d_mask_seq_len``; keep
      the 2D mask (legacy fallback).
    - ``BUILD``           — construct the 4D block-diagonal+causal mask.

    The multimodal and SSM checks take precedence over the seq-len cap: all three
    fall back to the 2D mask, but the model-shape reason is the one worth surfacing.
    """
    if "document_ids" not in batch:
        return NONE
    if is_multimodal(model_config):
        return SKIP_MULTIMODAL
    if has_ssm_layers(model_config):
        return SKIP_SSM
    if seq_len > max_4d_mask_seq_len:
        return SKIP_SEQLEN
    return BUILD
