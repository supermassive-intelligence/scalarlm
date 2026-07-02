"""
Unit tests for adapters.resolve_target_modules.resolve_target_modules.

PEFT's "all-linear" shorthand fails to expand for some architectures
(notably MoE: Qwen3MoeForCausalLM under peft 0.19 + transformers 5.x). PEFT
falls back to iterating the literal string as a set of characters and raises
`Target modules {'-','l','n','r','i','a','e'} not found in the base model`,
which is the qwen3-moe TRAIN_FAILED in the cuda-spark sweep. We resolve the
shorthand ourselves from the live model, so these tests use small synthetic
nn.Modules rather than HF downloads.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from adapters.resolve_target_modules import resolve_target_modules


class _DenseLike(nn.Module):
    """A miniature ...ForCausalLM: attention + MLP projections + an output head,
    with several layers so leaf names repeat (as in a real stacked model)."""

    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )
            for _ in range(n_layers)
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


class _NoOutputEmbeddings(nn.Module):
    """A model whose get_output_embeddings() returns None (some configs do);
    the output head must still be excluded by its conventional leaf name."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return None


class _MoeLike(nn.Module):
    """A miniature Qwen3MoE-style ...ForCausalLM mirroring `decoder_sparse_step`:
    some decoder layers carry a *dense* MLP (`gate_proj`/`up_proj`/`down_proj`)
    and others a *sparse* MLP with a router (`gate`) + routed `experts`. By
    default layer 0 is dense and layer 1 is sparse — the real
    `qwen3-moe-tiny-random` layout (decoder_sparse_step=2). The routed experts
    and the router can't be served from a .pt adapter, so resolution must adapt
    attention + the dense MLP only — and by *full path*, since the dense MLP
    shares leaf names with the experts."""

    def __init__(self, sparse_layers=(1,), n_layers=2, n_experts=4):
        super().__init__()

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _dense_mlp():
            return nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _sparse_mlp():
            return nn.ModuleDict(
                {
                    "gate": nn.Linear(8, n_experts, bias=False),  # router
                    "experts": nn.ModuleList(
                        nn.ModuleDict(
                            {
                                "gate_proj": nn.Linear(8, 8, bias=False),
                                "up_proj": nn.Linear(8, 8, bias=False),
                                "down_proj": nn.Linear(8, 8, bias=False),
                            }
                        )
                        for _ in range(n_experts)
                    ),
                }
            )

        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "self_attn": _attn(),
                    "mlp": _sparse_mlp() if i in sparse_layers else _dense_mlp(),
                }
            )
            for i in range(n_layers)
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_moe_adapts_attention_and_dense_mlp_excluding_experts_and_router():
    # Layer 0 dense, layer 1 sparse (the real qwen3-moe-tiny-random layout).
    model = _MoeLike(sparse_layers=(1,), n_layers=2)
    result = resolve_target_modules(model, "all-linear")
    # Full paths: attention on every layer + the DENSE MLP (layer 0 only),
    # but no experts, no router, no head.
    assert result == [
        "layers.0.mlp.down_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
    ]
    assert not any(".experts." in name for name in result)  # no routed experts
    assert not any(name.endswith(".gate") for name in result)  # no router
    assert not any("lm_head" in name for name in result)  # no output head


def test_moe_all_layers_sparse_adapts_attention_only():
    # If every layer is sparse (no dense MLP anywhere), only attention survives.
    model = _MoeLike(sparse_layers=(0, 1), n_layers=2)
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
    ]
    assert not any("mlp" in name for name in result)


def test_all_linear_expands_to_distinct_leaf_names_minus_head():
    model = _DenseLike()
    resolved = resolve_target_modules(model, "all-linear")
    assert resolved == [
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert "lm_head" not in resolved


def test_repeated_leaf_names_collapse_to_a_set():
    # 3 layers × 7 projections, but only 7 distinct leaf names survive.
    model = _DenseLike(n_layers=3)
    resolved = resolve_target_modules(model, "all-linear")
    assert len(resolved) == 7


def test_output_head_excluded_by_name_when_no_output_embeddings():
    model = _NoOutputEmbeddings()
    resolved = resolve_target_modules(model, "all-linear")
    assert resolved == ["q_proj"]


class _Decoder(nn.Module):
    """A miniature language tower: one block of attention + MLP projections."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(8, 8, bias=False),
                        "k_proj": nn.Linear(8, 8, bias=False),
                        "v_proj": nn.Linear(8, 8, bias=False),
                        "o_proj": nn.Linear(8, 8, bias=False),
                    }
                )
            ]
        )


class _VisionTower(nn.Module):
    """A vision encoder that REUSES the language tower's leaf names — the
    Gemma3 case that defeats plain leaf-name targeting."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "q_proj": nn.Linear(8, 8, bias=False),
                "k_proj": nn.Linear(8, 8, bias=False),
            }
        )


class _MultimodalModel(nn.Module):
    """A ...ForConditionalGeneration-shaped wrapper: a `vision_config` on the
    config, `get_decoder()` returning the language tower, a vision tower whose
    linears share leaf names with the language tower, and a tied output head."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=8))
        self.language_model = _Decoder()
        self.vision_tower = _VisionTower()
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_decoder(self):
        return self.language_model

    def get_output_embeddings(self):
        return self.lm_head


def test_multimodal_targets_full_paths_under_language_decoder_only():
    model = _MultimodalModel()
    resolved = resolve_target_modules(model, "all-linear")
    # Every target is a full path inside the language tower...
    assert resolved == [
        "language_model.layers.0.k_proj",
        "language_model.layers.0.o_proj",
        "language_model.layers.0.q_proj",
        "language_model.layers.0.v_proj",
    ]
    # ...and nothing in the vision tower, despite the shared k_proj/q_proj names.
    assert not any("vision_tower" in name for name in resolved)


def test_multimodal_resolution_excludes_output_head():
    model = _MultimodalModel()
    resolved = resolve_target_modules(model, "all-linear")
    assert not any(name.endswith("lm_head") for name in resolved)


def test_explicit_list_is_passed_through_unchanged():
    model = _DenseLike()
    explicit = ["q_proj", "v_proj"]
    assert resolve_target_modules(model, explicit) is explicit


def test_explicit_non_shorthand_string_passes_through():
    model = _DenseLike()
    # A regex/string that isn't the "all-linear" shorthand is PEFT's own
    # to interpret — we must not touch it.
    assert resolve_target_modules(model, "q_proj") == "q_proj"


def test_none_passes_through():
    model = _DenseLike()
    assert resolve_target_modules(model, None) is None
