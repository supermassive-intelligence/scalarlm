"""Pure-Python vLLM CLI-arg builder.

Kept free of heavy imports (``torch``, ``vllm``) so it can be exercised
from unit tests without pulling in the full training/inference stack.
"""

from __future__ import annotations


def build_vllm_cli_args(config: dict) -> list[str]:
    """Build the base vLLM CLI arg list from a scalarlm config dict.

    Callers then extend this with model name, port, and any
    ``SCALARLM_VLLM_ARGS`` overrides (dedupped).

    The only non-obvious behavior is the adapter gates. When
    ``enable_lora`` is False, ``--enable-lora`` is omitted, which avoids
    wrapping every layer in a LoRA-aware shim. See Phase 31b in
    ``enhance-openai-api.md``.

    ``enable_tokenformer`` gates ``--enable-tokenformer``, which the vLLM
    fork needs in order to load the Tokenformer-keyed ``.pt`` adapters the
    ScalarLM trainer produces. Both flags together select the fork's
    HybridAdapterManager, which handles Tokenformer-only, LoRA-only, and
    hybrid checkpoints. See ``default_config.py`` for the trade-offs.
    """
    args = [
        f"--dtype={config['dtype']}",
        "--max-model-len=auto",
        f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
        f"--max-log-len={config['max_log_length']}",
        f"--tensor-parallel-size={config['tensor_parallel_size']}",
        "--enable-auto-tool-choice",
        "--tool-call-parser=hermes",
        "--trust-remote-code",
    ]
    # Fallbacks must match default_config.Config, or a config dict that
    # predates a key produces a different server than the shipped default.
    if config.get("enable_lora", False):
        args.append("--enable-lora")
    if config.get("enable_tokenformer", True):
        args.append("--enable-tokenformer")
    if config.get("limit_mm_per_prompt") is not None:
        args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")
    return args
