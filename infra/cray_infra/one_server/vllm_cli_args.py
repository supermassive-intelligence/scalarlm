"""Pure-Python vLLM CLI-arg builder.

Kept free of heavy imports (``torch``, ``vllm``) so it can be exercised
from unit tests without pulling in the full training/inference stack.
"""

from __future__ import annotations

import os


def build_vllm_cli_args(config: dict) -> list[str]:
    """Build the base vLLM CLI arg list from a scalarlm config dict.

    Callers then extend this with model name, port, and any
    ``SCALARLM_VLLM_ARGS`` overrides (dedupped).

    Two behaviors here are non-obvious:

    - The adapter gates: when ``enable_lora`` is False, ``--enable-lora``
      is omitted, which avoids wrapping every layer in a LoRA-aware shim.
      See Phase 31b in ``enhance-openai-api.md``. ``enable_tokenformer``
      gates ``--enable-tokenformer``, which the vLLM fork needs in order
      to load the Tokenformer-keyed ``.pt`` adapters the ScalarLM trainer
      produces. Both flags together select the fork's
      HybridAdapterManager, which handles Tokenformer-only, LoRA-only,
      and hybrid checkpoints. See ``default_config.py`` for the
      trade-offs.
    - The ``reasoning_parser`` value: an explicit
      ``config["reasoning_parser"]`` (including ``""`` to opt out) wins
      over conservative model-name detection. ``SCALARLM_VLLM_ARGS`` has
      the final say because :mod:`create_vllm` applies those operator
      overrides after this base list.
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
    reasoning_parser = resolve_reasoning_parser(config)
    if reasoning_parser:
        args.append(f"--reasoning-parser={reasoning_parser}")
    return args


_REASONING_PARSER_FLAG_NAME = "--reasoning-parser"
_REASONING_PARSER_FLAG_PREFIX = _REASONING_PARSER_FLAG_NAME + "="


def _reasoning_parser_from_env_override() -> str | None:
    """Return the parser set in ``SCALARLM_VLLM_ARGS``, if any.

    ``create_vllm.py`` removes matching config-derived flags and appends
    ``SCALARLM_VLLM_ARGS`` last. Mirror that last-wins precedence here so
    the base argument list and the eventual parsed arguments agree. Both
    argparse forms (``--reasoning-parser=qwen3`` and
    ``--reasoning-parser qwen3``) are supported.
    """
    tokens = os.environ.get("SCALARLM_VLLM_ARGS", "").split()
    value = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith(_REASONING_PARSER_FLAG_PREFIX):
            value = token[len(_REASONING_PARSER_FLAG_PREFIX) :]
            i += 1
        elif (
            token == _REASONING_PARSER_FLAG_NAME
            and i + 1 < len(tokens)
            and not tokens[i + 1].startswith("-")
        ):
            value = tokens[i + 1]
            i += 2
        else:
            i += 1
    return value


def resolve_reasoning_parser(config: dict) -> str | None:
    """Resolve the effective ``--reasoning-parser`` value for ``config``.

    Precedence, from highest to lowest, is an operator override in
    ``SCALARLM_VLLM_ARGS``, an explicit config value (where ``""`` opts
    out), then conservative checkpoint-name detection.
    """
    env_override = _reasoning_parser_from_env_override()
    if env_override is not None:
        return env_override

    reasoning_parser = config.get("reasoning_parser")
    if reasoning_parser is None:
        reasoning_parser = _detect_reasoning_parser(config.get("model") or "")
    return reasoning_parser


# Add entries only after validating that a model family's vLLM parser works
# end-to-end. A mismatched parser can silently classify all output as
# unfinished reasoning and leave the visible content empty.
_REASONING_PARSER_BY_MODEL_SUBSTRING: tuple[tuple[str, str], ...] = (
    ("qwen3", "qwen3"),
)

# These Qwen3 variants do not use the original family's hybrid
# ``<think>...</think>`` contract. Operators can still set an explicit
# parser for a particular checkpoint when appropriate.
_NON_REASONING_MODEL_SUBSTRINGS: dict[str, tuple[str, ...]] = {
    "qwen3": ("coder", "next"),
}


def _checkpoint_name(model: str) -> str:
    """Return the final component of a Hugging Face model id or path."""
    return model.rsplit("/", 1)[-1]


def _detect_reasoning_parser(model: str) -> str | None:
    """Return a parser only for a validated checkpoint-name family."""
    checkpoint_lower = _checkpoint_name(model.lower())
    for substring, parser in _REASONING_PARSER_BY_MODEL_SUBSTRING:
        if substring.lower() not in checkpoint_lower:
            continue
        exclusions = _NON_REASONING_MODEL_SUBSTRINGS.get(substring, ())
        if any(exclusion.lower() in checkpoint_lower for exclusion in exclusions):
            return None
        return parser
    return None
