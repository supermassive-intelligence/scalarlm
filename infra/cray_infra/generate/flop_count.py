"""FLOP-per-token estimation from a HuggingFace model config.

Path A computes this from a vLLM ``ModelConfig`` in
``cray_infra.one_server.create_generate_worker.compute_flop_count``. The
HTTP proxy (``openai_v1_router``) runs out-of-process from vLLM and cannot
access that object, so it resolves architecture info via HuggingFace
``AutoConfig`` instead. The formula — attention + MLP + embedding + output
projection — matches Path A so the shared metrics counter stays coherent
across both surfaces.

Caching is coarse: one flops-per-token per base-model name, computed once.
Adapter-induced FLOP deltas are sub-percent for LoRA/tokenformer and we're
feeding an observability counter, not a billing number.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_CACHE: dict[str, int] = {}
_CACHE_LOCK = threading.Lock()


def _default_hf_config_loader(name: str) -> Any:
    """Resolve ``name`` to a HuggingFace config object.

    Isolated as a module-level function so tests can monkey-patch it without
    importing the heavy ``transformers`` dependency.
    """
    from transformers import AutoConfig  # heavy dep; imported lazily
    return AutoConfig.from_pretrained(name)


# Indirection point — tests swap this out. Production callers never set it.
_hf_config_loader: Callable[[str], Any] = _default_hf_config_loader


def compute_flops_per_token(base_model_name: str) -> int:
    """Return FLOPs per decoded token for ``base_model_name``.

    Returns 0 on any resolution failure (config missing, network error,
    unexpected architecture). Callers should treat 0 as "unknown — skip the
    FLOP metric update" rather than as a real zero.
    """
    with _CACHE_LOCK:
        cached = _CACHE.get(base_model_name)
        if cached is not None:
            return cached

    try:
        hf_config = _hf_config_loader(base_model_name)
    except Exception as exc:  # noqa: BLE001 — defensive catch for diverse failures
        logger.warning("HF config load failed for %s: %s", base_model_name, exc)
        return 0

    inner = _resolve_text_config(hf_config)

    try:
        flops = _compute(inner)
    except Exception as exc:  # noqa: BLE001
        logger.warning("flops formula failed for %s: %s", base_model_name, exc)
        return 0

    with _CACHE_LOCK:
        _CACHE[base_model_name] = flops
    return flops


def _resolve_text_config(hf_config: Any) -> Any:
    """Multimodal wrappers (e.g. Qwen-VL) hide LM parameters under
    ``text_config``. Fall back to the outer config when that attribute is
    absent so plain LMs continue to work.
    """
    return getattr(hf_config, "text_config", hf_config) or hf_config


def _compute(cfg: Any) -> int:
    vocab_size = cfg.vocab_size
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    num_attention_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)

    head_size = getattr(cfg, "head_dim", None) or hidden_size // num_attention_heads
    intermediate_size = getattr(cfg, "intermediate_size", 4 * hidden_size)

    q_proj = hidden_size * (num_attention_heads * head_size)
    kv_proj = hidden_size * (num_kv_heads * head_size * 2)
    qk = num_attention_heads * head_size
    av = num_attention_heads * head_size
    o_proj = hidden_size * hidden_size
    attention_per_layer = q_proj + kv_proj + qk + av + o_proj

    fc1 = hidden_size * intermediate_size
    fc2 = intermediate_size * hidden_size
    mlp_per_layer = fc1 + fc2

    embedding = hidden_size * vocab_size
    output_projection = hidden_size * vocab_size

    return num_layers * (attention_per_layer + mlp_per_layer) + embedding + output_projection


def _clear_cache_for_tests() -> None:
    """Test hook — production code should not call this."""
    with _CACHE_LOCK:
        _CACHE.clear()
