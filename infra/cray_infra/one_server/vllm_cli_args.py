"""Pure-Python vLLM CLI-arg builder.

Kept free of heavy imports (``torch``, ``vllm``) so it can be exercised
from unit tests without pulling in the full training/inference stack.
"""

from __future__ import annotations


def build_vllm_cli_args(config: dict) -> list[str]:
    """Build the base vLLM CLI arg list from a scalarlm config dict.

    Callers then extend this with model name, port, and any
    ``SCALARLM_VLLM_ARGS`` overrides (dedupped).

    The only non-obvious behavior is the ``enable_lora`` gate: when the
    config flag is False, ``--enable-lora`` is omitted, which avoids
    wrapping every layer in a LoRA-aware shim. See Phase 31b in
    ``enhance-openai-api.md``.
    """
    args = [
        f"--dtype={config['dtype']}",
        "--max-model-len=auto",
        f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
        f"--max-log-len={config['max_log_length']}",
        f"--tensor-parallel-size={config['tensor_parallel_size']}",
        "--trust-remote-code",
    ]
    # Tool-call extraction is opt-out. With Hermes auto-extraction on,
    # models trained on <tool_call>{...}</tool_call> markup (Qwen3, etc.)
    # spontaneously emit such blocks even when no tools are sent, and any
    # malformed JSON inside trips the OpenAI client's json.loads with
    # JSONDecodeError. Pure-text agents can disable this in the config.
    if config.get("enable_tool_calls", True):
        args.append("--enable-auto-tool-choice")
        args.append("--tool-call-parser=hermes")
    if config.get("enable_lora", True):
        args.append("--enable-lora")
    if config.get("limit_mm_per_prompt") is not None:
        args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")
    return args
