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

3. **Dense hybrid SSM.** FalconH1 (`FalconH1ForCausalLM`) runs a Mamba-2 mixer ‖
   attention in every layer with NO MoE. Its mixer holds an `out_proj` `nn.Linear`,
   and PEFT's `_check_lora_target_modules_mamba` REJECTS a target named
   `out_proj`/`conv1d` on Mamba `model_type`s — the FalconH1 TRAIN_FAILED. We drop
   the SSM mixer subtree (`.mamba`/`.linear_attn`) and adapt attention + MLP, which
   carries memorization (granite precedent). See `_has_ssm_layers`.

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


def _ssm_mixer_prefixes(model) -> set[str]:
    """Dotted names of Mamba-2 SSM mixer submodules that DON'T carry `.mamba` or
    `.linear_attn` in their path, so the name-substring checks elsewhere can't
    match them. NVIDIA's NemotronH (`NemotronHForCausalLM`) names every per-layer
    mixer `.mixer` — the SAME attribute an attention or MLP block uses (the block
    type varies by `hybrid_override_pattern`) — so its Mamba-2 mixer can only be
    told apart structurally: a selective-scan mixer carries an `A_log` parameter
    (alongside `D`/`dt_bias` and a `conv1d`) that attention/MLP mixers do not.
    Returns the mixer module names; callers exclude every `nn.Linear` beneath them
    (the mixer's `in_proj`/`out_proj`), keeping the attention `q/k/v/o_proj` and
    MLP `up/down_proj` of sibling blocks. Empty for archs whose SSM mixers already
    match `.mamba`/`.linear_attn` (granite / FalconH1 / Qwen3.5) and for non-SSM
    models."""
    prefixes: set[str] = set()
    for name, module in model.named_modules():
        if ".mamba" in name or ".linear_attn" in name:
            continue  # matched by name; no need for the signature fallback
        if any(pname == "A_log" for pname, _ in module.named_parameters(recurse=False)):
            prefixes.add(name)
    return prefixes


def _is_ssm_linear(name: str, ssm_prefixes: set[str]) -> bool:
    """True if the `nn.Linear` at dotted `name` belongs to a state-space /
    linear-attention mixer whose LoRA the `.pt` path can't serve — matched either
    by the `.mamba`/`.linear_attn` name convention (granite / FalconH1 / Qwen3.5)
    or by a signature-detected mixer prefix (NemotronH's `.mixer`, see
    `_ssm_mixer_prefixes`)."""
    if ".mamba" in name or ".linear_attn" in name:
        return True
    return any(name.startswith(prefix + ".") for prefix in ssm_prefixes)


def _has_ssm_layers(model) -> bool:
    """True if the model has Mamba-2 SSM (`.mamba.*`) or linear-attention
    (`.linear_attn.*`) mixer submodules — GraniteMoeHybrid / FalconH1 Mamba-2, or
    Qwen3.5's GatedDeltaNet — or a signature-detected Mamba-2 mixer that isn't
    named that way (NemotronH's `.mixer`, see `_ssm_mixer_prefixes`).

    This matters for a *dense* hybrid — FalconH1 (`FalconH1ForCausalLM`: a Mamba-2
    mixer ‖ attention in every layer, NO MoE) and NemotronH (`NemotronHForCausalLM`:
    Mamba-2 / attention / MLP blocks interleaved by `hybrid_override_pattern`, also
    dense). Their Mamba mixer holds an `out_proj`/`in_proj` `nn.Linear`; for the
    FalconH1 `model_type` PEFT's `_check_lora_target_modules_mamba` REJECTS a target
    named `out_proj`/`conv1d` (the FalconH1 TRAIN_FAILED), and even where PEFT does
    NOT reject it (NemotronH's `model_type` isn't in PEFT's Mamba set) LoRA on the
    SSM projections is dropped on serve (vLLM's `MambaMixer2` doesn't wire it) — a
    memorization/serve mismatch. The dense leaf-name path (below) would emit those
    SSM projections; this predicate routes the dense case through the SSM-excluding
    path instead. MoE hybrids like granite already route through the MoE path, which
    drops the SSM. (Attention uses `o_proj`, not `out_proj`, so only the SSM mixer
    trips PEFT's check.)"""
    return any(
        ".mamba" in name or ".linear_attn" in name
        for name, _ in model.named_modules()
    ) or bool(_ssm_mixer_prefixes(model))


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
      state-space layers and their LoRA is untested/unserved,
    - the DeepSeek MLA latent projections (`kv_a_proj_with_mqa`, `kv_b_proj`) —
      vLLM absorbs them into the latent-attention kernel on serve, so their LoRA is
      dropped (the DeepSeek-V2-Lite NO_MEM near-miss); `q_proj`/`o_proj` serve clean
      and stay, and
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
    ssm_prefixes = _ssm_mixer_prefixes(model)
    paths: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if module_name.endswith("lm_head"):  # head, when output_embeddings is None
            continue
        if _is_ssm_linear(module_name, ssm_prefixes):
            # SSM / linear-attention projections — not adapted (untested/unserved
            # by the .pt LoRA path). `.mamba` = GraniteMoeHybrid's Mamba-2
            # (in_proj/out_proj); `.linear_attn` = Qwen3.5's GatedDeltaNet
            # (in_proj_a/b/qkv/z, out_proj, conv1d), the linear-attention half of
            # its hybrid full/linear-attention stack; a signature-detected `.mixer`
            # = NemotronH's Mamba-2 (in_proj/out_proj), for any NemotronH-MoE variant.
            continue
        # The MoE router / gate — exclude by leaf name. Adapting it would perturb
        # expert selection, and its leaf name varies by arch: `gate` (Qwen3MoE/
        # Mixtral), `router` (PhiMoE, whose router is an nn.Linear subclass
        # returning a tuple, so LoRA-wrapping it crashes with `'tuple' object has
        # no attribute 'dtype'`), or a `*_gate` (Qwen3.5's `shared_expert_gate`,
        # which routes the shared expert). The dense projections we DO adapt end in
        # `_proj` (`gate_proj`/`up_proj`), so an exact/suffix match leaves them
        # untouched.
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in ("gate", "router") or leaf.endswith("_gate"):
            continue
        # DeepSeek MLA latent projections. On serve, vLLM's DeepseekV2MLAAttention
        # restructures attention into an absorbed latent-attention kernel
        # (MultiHeadLatentAttentionWrapper): `kv_b_proj` is decomposed into per-head
        # W_UK/W_UV and absorbed, and `kv_a_proj_with_mqa` produces the latent cache
        # feeding that kernel — so a LoRA delta trained on them as plain HF Linears is
        # NOT applied by the MLA path. Training memorizes (loss 0.0) but the served
        # adapter reproduces the golden only partially (DeepSeek-V2-Lite NO_MEM
        # near-miss: right prefix/tail, garbled middle). Keep LoRA off them so
        # memorization leans on the serve-clean attention projections (`q_proj`/
        # `o_proj`, applied normally inside the wrapper) + the routed experts (grouped
        # converter). These leaf names are MLA-specific — no non-MLA arch uses them —
        # so an unconditional leaf exclusion is safe. See
        # `archnovel-sweep-deepseek-cohere-nemotron-falconh1` (memory).
        if leaf in ("kv_a_proj_with_mqa", "kv_b_proj"):
            continue
        # DiffusionGemma's router is `...router.proj` (an nn.Linear under a
        # `DiffusionGemmaTextRouter` module) — the leaf is `proj`, not `gate`/
        # `router`, so the leaf check above misses it. Adapting the routing
        # projection destabilizes expert selection mid-fine-tune (the NeMo recipe
        # sets `freeze_router: true`); exclude the whole `.router.` subtree by
        # path. Harmless for archs without a `.router.` container.
        if ".router." in module_name:
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


def resolve_target_parameters(model) -> list[str]:
    """Leaf names of the batched `nn.Parameter` expert projections that PEFT must
    be told to adapt via `LoraConfig.target_parameters` — for *grouped*-expert
    MoEs whose experts are a single module holding `(num_experts, …)` parameters
    (`gate_up_proj`/`down_proj`), not per-expert `nn.Linear`s.

    Why this is needed. PEFT's `target_modules` only matches `nn.Module`s, so it
    cannot reach a bare expert `nn.Parameter`. PEFT *does* adapt grouped experts
    for some models, but ONLY as a side effect of its transformers-v5 MoE weight
    conversion (`peft/utils/transformers_weight_conversion.py::_convert_peft_config_moe`),
    which fires only when a *dense* MLP layer's `gate_proj/up_proj/down_proj` leaves
    are ALREADY in `target_modules`. Qwen3MoE has such a dense layer
    (`decoder_sparse_step`) so its experts get adapted; all-MoE OLMoE (no dense
    layer, though `olmoe` IS in PEFT's conversion map) and PhiMoE (`phimoe` NOT in
    the map) do NOT — their experts silently receive no LoRA and the model can only
    lean on attention, which cannot memorize (the whole MoE lesson). Naming the
    expert params here makes PEFT wrap them directly (ParamWrapper), producing the
    exact `{experts}.base_layer` = gate_up / `{experts}` = down tensors the grouped
    serve converter (vLLM `_stack_moe_lora_weights_gated`) already consumes — the
    same result Qwen3MoE gets, but without depending on a dense layer.

    Returns leaf-name suffixes (PEFT matches `target_parameters` by suffix). This
    is exactly the `{gate_up_proj, down_proj}` set PEFT itself derives for Qwen3MoE,
    so grouped models that already have a dense layer are unaffected (same set,
    unioned). Empty for dense models, for *separate* per-expert `nn.Linear` experts
    (Mixtral/PhiMoE-in-v4 — adapted via `target_modules`), and for `nn.Linear`-fused
    grouped experts (GraniteMoeHybrid `block_sparse_moe.{input,output}_linear`) —
    none of which expose bare batched expert parameters. Requires
    `lora_config.lora_dropout == 0` (PEFT's ParamWrapper rejects dropout), which the
    MoE sweep entries already set. See
    `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`.

    DiffusionGemma is deliberately excluded: every decoder layer runs a dense MLP
    *and* the grouped experts in parallel (`dense_out + moe_out`), so the always-on
    dense MLP carries memorization (like granite's `shared_mlp`) and the experts do
    NOT need LoRA. Adapting them would deviate from the NeMo reference recipe
    (freeze_router + target_modules only) and pull in the FusedMoE-LoRA serve path
    that ADR 0011 avoids. Its grouped `experts` module would otherwise match the
    scan below, so short-circuit on the diffusion config."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "diffusion_gemma":
        return []
    leaves: set[str] = set()
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "experts":
            continue
        for pname, param in module.named_parameters(recurse=False):
            # A batched (num_experts, in, out) projection — 3-D distinguishes it
            # from a router/norm scalar and from the per-expert nn.Linear layout
            # (whose weights live under `experts.{i}`, not directly on `experts`).
            if param.dim() >= 2:
                leaves.add(pname)
    return sorted(leaves)


def resolve_target_modules(model, target_modules):
    """If `target_modules` is the "all-linear" shorthand, resolve it against the
    live `model`:

    - **multimodal** (config has `vision_config`, `get_decoder()` available):
      the full dotted paths of every `nn.Linear` under the language decoder,
      excluding the output head. PEFT matches these exactly, so a vision tower
      reusing the same leaf names is not adapted. When the decoder is ALSO MoE
      (Qwen3.5-VL-MoE), the MoE-servable filter is applied within the decoder
      scope too — router / SSM (`.linear_attn`) / grouped experts are dropped,
      leaving attention + the dense `shared_expert`; grouped experts are adapted
      via `target_parameters`.
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
    - **dense hybrid SSM** (has `.mamba`/`.linear_attn`, non-MoE, non-multimodal —
      FalconH1): the sorted full paths of every `nn.Linear` OUTSIDE the SSM mixer
      subtree (attention + MLP), excluding the head. Drops the Mamba mixer's
      `out_proj` that PEFT refuses on Mamba `model_type`s.
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
            def _under_decoder(name):
                return name == prefix or name.startswith(prefix + ".")
            if _is_moe_model(model):
                # Multimodal MoE (e.g. Qwen3.5-VL-MoE `Qwen3_5MoeForConditionalGeneration`):
                # confine to the language decoder AND apply the MoE-servable filter
                # so LoRA lands on attention + the dense, always-on `shared_expert`
                # (the memorization carrier, like granite's `shared_mlp`) — NOT the
                # router, the GatedDeltaNet SSM (`.linear_attn`), or the grouped
                # routed experts (which go via `target_parameters`). The plain
                # multimodal path below would adapt all of those. Full paths, as the
                # dense/shared-expert and expert projections share leaf names.
                separate = _has_separate_experts(model)
                paths = _moe_servable_linear_paths(
                    model, output_embeddings, separate_experts=separate)
                scoped = sorted(p for p in paths if _under_decoder(p))
                if scoped:
                    return scoped
                # Nothing matched under the decoder — fall through rather than
                # adapt nothing.
            else:
                return sorted(
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, nn.Linear)
                    and module is not output_embeddings
                    and _under_decoder(name)
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

    if _has_ssm_layers(model):
        # Dense hybrid SSM model (FalconH1: Mamba-2 ‖ attention; NemotronH: Mamba-2 /
        # attention / MLP blocks interleaved — both dense, no MoE). Emit the full
        # nn.Linear PATHS for the adaptable layers — attention + MLP — excluding the
        # SSM mixer subtree (`.mamba`/`.linear_attn`, or a signature-detected `.mixer`
        # for NemotronH) and the output head. Keeping LoRA on attention + MLP is what
        # carries memorization (the granite precedent: the SSM stays off, attention +
        # dense MLP memorize), and dropping the SSM subtree removes the mixer's
        # `out_proj`/`in_proj` — which PEFT refuses on FalconH1's `model_type`, and
        # which vLLM drops on serve for NemotronH (see `_has_ssm_layers`). Full paths
        # — not leaf names — so the exclusion is by SSM subtree, leaving any attention
        # `out_proj` (were an arch to use that leaf) untouched.
        ssm_prefixes = _ssm_mixer_prefixes(model)
        paths = sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and (output_embeddings is None or module is not output_embeddings)
            and not name.endswith("lm_head")
            and not _is_ssm_linear(name, ssm_prefixes)
        )
        if paths:
            return paths
        # Nothing outside the SSM mixer (pure Mamba, no attention/MLP linears) —
        # fall through to the dense path rather than adapt nothing.

    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        names.add(module_name.split(".")[-1])
    names.discard("lm_head")  # belt-and-suspenders when get_output_embeddings()==None

    return sorted(names)
