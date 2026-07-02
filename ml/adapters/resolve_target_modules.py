"""Resolve PEFT's "all-linear" target-modules shorthand ourselves.

PEFT's `LoraConfig(target_modules="all-linear")` is supposed to expand the
shorthand to every linear layer except the output head. We resolve it ourselves
for two reasons:

1. **MoE.** Under peft 0.19 + transformers 5.x the expansion silently fails for
   some architectures (observed for Qwen3MoeForCausalLM) and PEFT falls back to
   iterating the literal string as a *set of characters*, raising
   `Target modules {'-','l','n','r','i','a','e'} not found` — the `qwen3-moe`
   TRAIN_FAILED in the cuda-spark sweep. We also keep LoRA off the routed
   experts (their fused LoRA isn't `.pt`-serveable) while still adapting the
   attention and any dense MLP — see `_moe_servable_linear_paths` and
   `docs/reports/2026-06-30-moe-expert-lora-serving.md`.

2. **Multimodal.** For a `...ForConditionalGeneration` wrapper, "all-linear"
   would also adapt the vision encoder. PEFT matches `target_modules` by name
   *suffix across the whole model*, and a vision tower can reuse the language
   tower's leaf names (Gemma3's `vision_tower...self_attn.k_proj`), so a plain
   leaf-name set can't exclude it. We confine LoRA to the language decoder by
   emitting its Linear modules' *full paths*.

For dense models the result is the sorted leaf-name set — byte-identical to
PEFT's own expansion (same trainable parameters).
"""

import torch.nn as nn

# PEFT's sentinel for "adapt every linear layer except the output head".
ALL_LINEAR = "all-linear"


def _is_multimodal_model(model) -> bool:
    """True for HF multimodal wrappers — they nest a `vision_config` on their
    config. A `vision_config` present but `None` reads as not-multimodal."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    return getattr(config, "vision_config", None) is not None


def _language_decoder(model):
    """The text-decoder submodule to confine LoRA to on a multimodal model, or
    None when no scoping is needed (dense model, or no usable `get_decoder`).
    `get_decoder()` is the standard HF handle for the text tower
    (Gemma3TextModel, Qwen2VLTextModel)."""
    if not _is_multimodal_model(model):
        return None
    if not hasattr(model, "get_decoder"):
        return None
    decoder = model.get_decoder()
    if decoder is None or decoder is model:
        return None
    return decoder


def _module_prefix(model, target) -> str | None:
    """The dotted name of `target` within `model` (by identity), or None if it
    isn't found among the model's submodules."""
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def _is_moe_model(model) -> bool:
    """True if the model has routed MoE expert submodules (a `.experts` module).

    A `.pt` adapter that adapts the *fused experts* can't be served: vLLM's
    `FusedMoEWithLoRA.set_lora` wants a per-expert tensor *list* (gate/down/up,
    each `[num_experts, rank, dim]`), while the ScalarLM trainer exports stacked
    2-D tensors. Rather than reproduce vLLM's PEFT→fused-MoE conversion, we keep
    LoRA off the experts (and the router) and adapt everything else that *does*
    serve from a `.pt` adapter — see `_moe_servable_linear_paths`."""
    return any(".experts" in name for name, _ in model.named_modules())


def _moe_servable_linear_paths(model, output_embeddings) -> list[str]:
    """Full dotted paths of every `nn.Linear` in a MoE model whose LoRA a `.pt`
    adapter can serve: the attention projections (all layers) and any *dense*
    MLP — the non-sparse decoder layers, e.g. layer 0 under Qwen3MoE's
    `decoder_sparse_step`. Excludes:

    - the routed `.experts` (their fused LoRA isn't `.pt`-serveable, see
      `_is_moe_model`),
    - the router `gate` (adapting it would perturb expert selection), and
    - the output head.

    *Full paths* — not leaf names — because the dense-MLP projections
    (`gate_proj`/`up_proj`/`down_proj`) reuse the SAME leaf names as the expert
    projections, so a leaf-name set can't include one while excluding the other.
    PEFT matches these exact paths, so the experts are left untouched."""
    paths: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if module_name.endswith("lm_head"):  # head, when output_embeddings is None
            continue
        if ".experts" in module_name:  # routed experts — not .pt-serveable
            continue
        if module_name.endswith(".gate") or module_name == "gate":  # MoE router
            continue
        paths.append(module_name)
    return paths


def resolve_target_modules(model, target_modules):
    """If `target_modules` is the "all-linear" shorthand, resolve it against the
    live `model`:

    - **multimodal** (config has `vision_config`, `get_decoder()` available):
      the full dotted paths of every `nn.Linear` under the language decoder,
      excluding the output head. PEFT matches these exactly, so a vision tower
      reusing the same leaf names is not adapted.
    - **MoE** (has routed `.experts` submodules, non-multimodal): the sorted
      *full paths* of every serveable `nn.Linear` — attention (all layers) plus
      any dense (non-sparse) MLP — excluding the routed experts, the router
      `gate`, and the output head. The fused-expert LoRA can't be served from a
      `.pt` adapter (vLLM's `FusedMoEWithLoRA` wants a per-expert tensor list,
      not stacked tensors), so LoRA is kept off the experts; full paths (not leaf
      names) are required because the dense MLP shares the experts' leaf names.
    - **dense**: the sorted set of distinct `nn.Linear` leaf-module names,
      excluding the output head — identical to PEFT's all-linear expansion.

    Any other value — an explicit list, a non-shorthand string, or None — is
    returned unchanged for PEFT to interpret itself.
    """
    if target_modules != ALL_LINEAR:
        return target_modules

    # Exclude the output projection. Prefer identity (handles heads not named
    # "lm_head"); fall back to the conventional leaf name below.
    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()

    decoder = _language_decoder(model)
    if decoder is not None:
        prefix = _module_prefix(model, decoder)
        if prefix is not None:
            return sorted(
                name
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and module is not output_embeddings
                and (name == prefix or name.startswith(prefix + "."))
            )
        # get_decoder() returned a module we couldn't locate — fall through to
        # the dense path rather than silently adapt nothing.

    if _is_moe_model(model):
        # MoE: emit full paths for the .pt-serveable linears — attention (all
        # layers) + any dense (non-sparse) MLP — while excluding the routed
        # experts (not serveable, see _is_moe_model) and the router. Full paths,
        # not leaf names, because the dense MLP shares the experts' leaf names.
        paths = _moe_servable_linear_paths(model, output_embeddings)
        if paths:
            return sorted(paths)
        # Nothing matched (unusual arch) — fall through to the dense path rather
        # than adapt nothing.

    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        names.add(module_name.split(".")[-1])
    names.discard("lm_head")  # belt-and-suspenders when get_output_embeddings()==None

    return sorted(names)
