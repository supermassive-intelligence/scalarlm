from abc import abstractmethod, ABC
import torch
from torch import nn

from cray_infra.util.get_config import get_config

import math

import logging

logger = logging.getLogger(__name__)


class TokenformerAdapter(nn.Module):
    def __init__(self, layer, hidden_size, device):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.num_heads = get_config()["tokenformer_num_heads"]
        self.head_dim = hidden_size // self.num_heads
        self.tokenformer_r = get_config()["tokenformer_r"]

        # Force floating-point storage so the parameters can carry gradients.
        # Under quantized loading contexts (e.g. NVFP4) the default dtype may
        # be a non-float type derived from the base layer's packed weights,
        # which would make `nn.Parameter(..., requires_grad=True)` fail.
        default_dtype = torch.get_default_dtype()
        param_dtype = default_dtype if default_dtype.is_floating_point else torch.float32

        self.tokenformer_k = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size, device=device, dtype=param_dtype)
        )
        self.tokenformer_v = nn.Parameter(
            torch.zeros(
                self.num_heads, self.hidden_size * self.tokenformer_r, device=device, dtype=param_dtype
            )
        )

        self.tokenformer_p = nn.Parameter(
            torch.zeros(self.tokenformer_r, self.hidden_size, device=device, dtype=param_dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        k_gain = 3.0 / math.sqrt(self.hidden_size / self.num_heads)
        v_gain = 3.0 / math.sqrt(self.hidden_size)

        nn.init.normal_(self.tokenformer_k, std=k_gain)
        nn.init.uniform_(self.tokenformer_v, a=-v_gain, b=v_gain)
        nn.init.zeros_(self.tokenformer_p)

    # Call layer with all inputs and kwargs
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        all_base_layer_results = self.layer(hidden_states, *args, **kwargs)

        tokenformer_results = self.tokenformer_op(hidden_states)

        if isinstance(all_base_layer_results, tuple):
            base_layer_results = all_base_layer_results[0]
        else:
            base_layer_results = all_base_layer_results

        # sum the two outputs
        layer_and_adapter_sum = base_layer_results + tokenformer_results

        if isinstance(all_base_layer_results, tuple):
            results = (layer_and_adapter_sum,) + all_base_layer_results[1:]
        else:
            results = layer_and_adapter_sum

        return results

    def tokenformer_op(self, query: torch.Tensor) -> torch.Tensor:
        # Cast parameters to match query dtype
        dtype = query.dtype

        q = query.view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        k = self.tokenformer_k.to(dtype).view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        v = self.tokenformer_v.to(dtype).view(
            -1, self.num_heads, self.hidden_size * self.tokenformer_r // self.num_heads
        ).transpose(0, 1)

        result = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        proj_down = (
            result.transpose(0, 1)
            .contiguous()
            .view([-1, self.hidden_size, self.tokenformer_r])
        )

        query_batch = query.view([-1, 1, self.hidden_size])

        # Also cast tokenformer_p
        result = torch.bmm(query_batch, proj_down) @ self.tokenformer_p.to(dtype)

        return result.view(query.shape)

    # Visualize the size of the parameters
    def __repr__(self):
        return (
            f"TokenformerAdapter(\nhidden_size={self.hidden_size}\n(layer): "
            + self.layer.__repr__()
            + "\n)"
        )


# Path components inside a multimodal wrapper that are NOT part of the
# language model. Tokenformer is a language-model post-training adapter;
# wrapping vision / audio tower MLPs would use the wrong hidden_size and
# adapt parameters that aren't trained through the text loss.
#
# Match by path component so HF's `model.<thing>` wrapper prefix doesn't hide
# them (e.g., "model.vision_tower.encoder.layers.0.mlp" must be excluded).
#
# TODO: adapting the vision/audio towers themselves is not supported. Only the
# language model is adapted; the towers stay at their base weights. Lifting this
# needs three things that do not exist yet:
#   1. the surgeon to use each tower's own hidden_size rather than the text
#      config's, since `update_layer` currently reads one hidden size for the
#      whole model;
#   2. a serving-side key mapping for tower parameters -- vLLM exposes them
#      under names the trainer never produces (e.g. HF writes
#      `vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight`, which has
#      no vLLM counterpart), so they cannot round-trip through
#      `load_weights` today;
#   3. a training signal that actually reaches the towers -- a text-only
#      dataset never activates them.
# Until then the towers are deliberately frozen. Training them without (2) is
# worse than not training them: the weights land in the checkpoint and are
# either dropped at serve time or crash the engine.
_NON_LANGUAGE_PATH_COMPONENTS = frozenset(
    {
        "vision_tower",
        "audio_tower",
        "embed_vision",
        "embed_audio",
        "multi_modal_projector",
    }
)


def is_non_language_path(name: str) -> bool:
    """True if a parameter/module path sits inside a non-language tower.

    Shared by the surgeon (which refuses to insert adapters there) and by
    the trainer's freeze/unfreeze pass, so the two agree on what "language
    model" means. They disagreed before: the surgeon skipped the vision
    tower while the unfreeze pass matched substrings like "q_proj" against
    the full path and trained it anyway, putting vision-tower weights into
    every checkpoint. Those keys have no counterpart in vLLM's parameter
    layout, so they killed the engine at adapter-activation time.

    Text-only models have none of these components, so this is a no-op
    there — existing checkpoints are unaffected.
    """
    return any(part in _NON_LANGUAGE_PATH_COMPONENTS for part in name.split("."))


class TokenformerSurgeon(ABC):

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def _is_adapter_layer(self, layer_name):
        if is_non_language_path(layer_name):
            return False
        return "mlp" in layer_name.split(".")[-1]

    def _recursive_setattr(self, obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self._recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    def update_layer(self, name, layer):
        """Try to wrap the layer with a TokenformerAdapter."""
        if not self._is_adapter_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerAdapter")

        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "text_config"):
                hidden_size = self.model.config.text_config.hidden_size
            else:
                hidden_size = self.model.config.hidden_size
        elif hasattr(self.model, "model_config"):
            hidden_size = self.model.model_config.hidden_size
        else:
            logger.error("Model does not have config or model_config attribute")
            return

        # Wrap the layer with a TokenformerAdapter
        self._recursive_setattr(
            self.model,
            name,
            TokenformerAdapter(layer, hidden_size, device=self.device),
        )

    def insert_adapter_modules(self):
        # Add tokenformer adapters for mlp and attention
        for name, layer in self.model.named_modules():
            self.update_layer(name, layer)

        return self.model
