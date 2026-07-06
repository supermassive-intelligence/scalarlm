"""Resolve PEFT's "all-linear" target-modules shorthand ourselves.

PEFT's `LoraConfig(target_modules="all-linear")` is supposed to expand the
shorthand to every linear layer except the output head. We resolve it ourselves
for two reasons:

1. **MoE.** Under peft 0.19 + transformers 5.x the expansion silently fails for
   some architectures (observed for Qwen3MoeForCausalLM) and PEFT falls back to
   iterating the literal string as a *set of characters*, raising
   `Target modules {'-','l','n','r','i','a','e'} not found` — the `qwen3-moe`
   TRAIN_FAILED in the cuda-spark sweep. We adapt attention + any dense MLP, and
   handle routed experts by layout: *grouped* experts (Qwen3MoE/Granite) are kept
   off (PEFT and the grouped serve converter handle them), while *separate*
   per-expert Linears (Mixtral/PhiMoE) are included so they train and the
   separate-expert converter can serve them — see `_has_separate_experts`,
   `_moe_servable_linear_paths`, `docs/reports/2026-06-30-moe-expert-lora-serving.md`,
   and `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`.

2. **Multimodal.** For a `...ForConditionalGeneration` wrapper, "all-linear"
   would also adapt the vision encoder. PEFT matches `target_modules` by name
   *suffix across the whole model*, and a vision tower can reuse the language
   tower's leaf names (Gemma3's `vision_tower...self_attn.k_proj`), so a plain
   leaf-name set can't exclude it. We confine LoRA to the language decoder by
   emitting its Linear modules' *full paths*.

For dense models the result is the sorted leaf-name set — byte-identical to
PEFT's own expansion (same trainable parameters).
"""

import re

import torch.nn as nn

# PEFT's sentinel for "adapt every linear layer except the output head".
ALL_LINEAR = "all-linear"

# A per-expert projection in a *separate*-expert MoE: a numbered expert index
# under an `experts` container, e.g. `...experts.0.w1` (Mixtral/PhiMoE) or
# `...experts.3.gate_proj` (Qwen2MoE). Grouped-expert models (Qwen3MoE's
# `Qwen3MoeExperts`, GraniteMoeHybrid's `block_sparse_moe`) have NO such numbered
# nn.Linear, which is exactly how we tell the two layouts apart.
_SEPARATE_EXPERT_RE = re.compile(r"(?:^|\.)experts\.\d+\.")


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
    """True if the model has routed MoE expert submodules whose fused LoRA a `.pt`
    adapter can't serve. Two container conventions are recognized:

    - `.experts` — Qwen3MoE / PhiMoE (a `ModuleList`/`ModuleDict` of per-expert or
      grouped expert projections), and
    - `.block_sparse_moe` — GraniteMoeHybrid, whose grouped experts and router live
      under `model.layers.{i}.block_sparse_moe.*` with NO `.experts` submodule.

    A `.pt` adapter that adapts the *fused experts* can't be served: vLLM's
    `FusedMoEWithLoRA.set_lora` wants a per-expert tensor *list* (gate/down/up,
    each `[num_experts, rank, dim]`), while the ScalarLM trainer exports stacked
    2-D tensors. Rather than reproduce vLLM's PEFT→fused-MoE conversion, we keep
    LoRA off the experts (and the router) and adapt everything else that *does*
    serve from a `.pt` adapter — see `_moe_servable_linear_paths`."""
    return any(
        ".experts" in name or ".block_sparse_moe" in name
        for name, _ in model.named_modules()
    )


def _has_separate_experts(model) -> bool:
    """True if the model exposes routed experts as a `ModuleList` of per-expert
    `nn.Linear` projections (`...experts.{i}.{w1,w2,w3}` — Mixtral / PhiMoE /
    Qwen2MoE), as opposed to a single *grouped* expert module whose batched
    weights are `nn.Parameter`s (Qwen3MoE's `Qwen3MoeExperts`) or two fused
    `nn.Linear`s (GraniteMoeHybrid's `block_sparse_moe.{input,output}_linear`).

    This drives LoRA targeting. PEFT adapts a *grouped* module on its own (via
    `ParamWrapper`) and its fused export is served by the grouped converter, so we
    keep those experts OUT of `target_modules` and let PEFT handle them. *Separate*
    per-expert Linears are NOT auto-adapted, so to train them (and then serve them
    via the separate-expert converter) we must name them explicitly. See
    `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`."""
    return any(
        isinstance(module, nn.Linear) and _SEPARATE_EXPERT_RE.search(name)
        for name, module in model.named_modules()
    )


def _moe_servable_linear_paths(model, output_embeddings, separate_experts=False) -> list[str]:
    """Full dotted paths of every `nn.Linear` in a MoE model whose LoRA a `.pt`
    adapter can serve: the attention projections (all layers) and any *dense*
    MLP — the non-sparse decoder layers, e.g. layer 0 under Qwen3MoE's
    `decoder_sparse_step`. When `separate_experts` is True the per-expert
    projections are ALSO included (see below). Always excludes:

    - the router (leaf `gate` in Qwen3MoE/Mixtral or `router` in PhiMoE — adapting
      it would perturb expert selection, and PhiMoE's is an nn.Linear subclass
      returning a tuple that crashes PEFT's LoRA wrap; GraniteMoe's router leaf is
      `layer` under `.block_sparse_moe`, covered by that exclusion below),
    - the Mamba-2 SSM projections (`.mamba.*`) — nothing else in the sweep has
      state-space layers and their LoRA is untested/unserved, and
    - the output head.

    Routed experts are handled per layout:

    - **grouped** (`separate_experts=False`) — Qwen3MoE's `Qwen3MoeExperts`
      (adapted by PEFT on its own) and GraniteMoeHybrid's whole `.block_sparse_moe`
      subtree are EXCLUDED; their fused LoRA isn't served from a leaf-name `.pt`
      target and the grouped converter handles it (see `_is_moe_model`).
    - **separate** (`separate_experts=True`) — Mixtral/PhiMoE per-expert Linears
      (`...experts.{i}.{w1,w2,w3}`) are INCLUDED so PEFT trains them; the
      separate-expert serve converter then stacks them for `FusedMoEWithLoRA`.

    *Full paths* — not leaf names — because the dense-MLP projections
    (`gate_proj`/`up_proj`/`down_proj`) reuse the SAME leaf names as the expert
    projections, so a leaf-name set can't include one while excluding the other.
    PEFT matches these exact paths."""
    paths: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if module_name.endswith("lm_head"):  # head, when output_embeddings is None
            continue
        if ".mamba" in module_name:  # Mamba-2 SSM projections (in_proj/out_proj) — not adapted
            continue
        # The MoE router — exclude by leaf name. Adapting it would perturb expert
        # selection, and its leaf name varies by arch: `gate` (Qwen3MoE/Mixtral) or
        # `router` (PhiMoE, whose router is an nn.Linear subclass returning a
        # tuple, so LoRA-wrapping it crashes with `'tuple' object has no
        # attribute 'dtype'`). `gate_proj` (a dense MLP projection we DO adapt)
        # is a different leaf, so an exact-name match leaves it untouched.
        if module_name.rsplit(".", 1)[-1] in ("gate", "router"):
            continue
        # A per-expert projection (`experts.{i}.*`). In a separate-expert model
        # these ARE the memorization-carrying weights and the separate-expert
        # converter serves them, so include them; otherwise keep them off the
        # adapter (grouped experts are adapted by PEFT / served differently).
        if _SEPARATE_EXPERT_RE.search(module_name):
            if separate_experts:
                paths.append(module_name)
            continue
        if ".experts" in module_name:  # grouped/fused experts container — not .pt-serveable
            continue
        if ".block_sparse_moe" in module_name:  # GraniteMoe grouped experts + router.layer
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
    - **MoE** (has routed `.experts`/`.block_sparse_moe` submodules, non-multimodal):
      the sorted *full paths* of every adaptable `nn.Linear` — attention (all
      layers) plus any dense (non-sparse) MLP — always excluding the router (leaf
      `gate`/`router`), the Mamba SSM, and the output head. Routed experts depend
      on layout: *grouped* experts (Qwen3MoE `Qwen3MoeExperts`, Granite
      `block_sparse_moe`) are kept OFF (PEFT/the grouped converter handle them),
      while *separate* per-expert Linears (Mixtral/PhiMoE `experts.{i}.{w1,w2,w3}`)
      are INCLUDED so they train and the separate-expert converter can serve them.
      Full paths (not leaf names) are required because the dense MLP and experts
      share leaf names.
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
        # MoE: emit full paths for the adaptable linears — attention (all layers)
        # + any dense (non-sparse) MLP — plus, for a *separate*-expert layout
        # (Mixtral/PhiMoE), the per-expert projections (grouped experts stay off;
        # PEFT/the grouped converter handle those). The router is always excluded.
        # Full paths, not leaf names, because the dense MLP and experts share leaf
        # names.
        separate = _has_separate_experts(model)
        paths = _moe_servable_linear_paths(model, output_embeddings, separate_experts=separate)
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
