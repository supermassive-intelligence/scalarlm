"""Pure-Python vLLM CLI-arg builder.

Kept free of heavy imports (``torch``, ``vllm``) so it can be exercised
from unit tests without pulling in the full training/inference stack.
"""

from __future__ import annotations

import os
import posixpath
import re


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


# Automatic selection is intentionally an allow-list. A mismatched qwen3
# parser treats output without ``</think>`` as unfinished reasoning, which can
# leave the user-visible content empty. The original Qwen3 hybrid checkpoints
# below, dedicated Thinking-2507 checkpoints, and the Qwen3.5/3.6 families are
# the model-name shapes validated by the pinned vLLM fork. Unknown variants
# remain opt-in through ``reasoning_parser``.
_QWEN3_HYBRID_CHECKPOINTS = (
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
)
_QWEN3_THINKING_2507_CHECKPOINTS = (
    "qwen3-4b-thinking-2507",
    "qwen3-30b-a3b-thinking-2507",
    "qwen3-235b-a22b-thinking-2507",
)

# Quantized copies preserve the source checkpoint's chat template. Keep the
# accepted suffixes narrow so names such as ``*-Base`` and ``*-Instruct`` do
# not accidentally inherit a parser merely because they share a size prefix.
_PACKAGING_SUFFIX_PATTERN = (
    r"(?:-(?:fp8|awq|gptq|gguf(?::[a-z0-9_]+)?|nvfp4|mxfp4|"
    r"w\d+a\d+|int\d+|quantized[._-]w\d+a\d+)"
    r"(?:-(?:dynamic|\d+bit|w\d+a\d+|g\d+|int\d+))*)?"
)

_QWEN35_36_PATTERN = re.compile(
    rf"^qwen3\.[56]-\d+(?:\.\d+)?b(?:-a\d+(?:\.\d+)?b)?"
    rf"{_PACKAGING_SUFFIX_PATTERN}$"
)


def _has_supported_packaging_suffix(checkpoint: str, base: str) -> bool:
    """Return whether ``checkpoint`` is ``base`` or a known packaged copy."""
    if checkpoint == base:
        return True
    if not checkpoint.startswith(base):
        return False
    suffix = checkpoint.removeprefix(base)
    return bool(re.fullmatch(_PACKAGING_SUFFIX_PATTERN, suffix))


def _is_supported_thinking_2507_checkpoint(checkpoint: str) -> bool:
    """Match a known Thinking-2507 checkpoint and its PTPC/quantized copies."""
    for base in _QWEN3_THINKING_2507_CHECKPOINTS:
        if checkpoint == base:
            return True
        if not checkpoint.startswith(base):
            continue
        suffix = checkpoint.removeprefix(base)
        if suffix.startswith("-ptpc"):
            suffix = suffix.removeprefix("-ptpc")
        if re.fullmatch(_PACKAGING_SUFFIX_PATTERN, suffix):
            return True
    return False


def _checkpoint_name(model: str) -> str:
    """Return the checkpoint component of a model id or local path.

    Trailing separators are ignored. For a standard Hugging Face cache path,
    the final component is a revision hash, so decode the repository name from
    ``models--ORG--REPO/snapshots/REVISION`` instead. This is a purely local
    path operation and never performs a Hub lookup.
    """
    normalized = posixpath.normpath(model.strip().replace("\\", "/"))
    if normalized in ("", "."):
        return ""

    parts = tuple(part for part in normalized.split("/") if part)
    for index, part in enumerate(parts[:-2]):
        if not part.lower().startswith("models--"):
            continue
        if parts[index + 1].lower() != "snapshots":
            continue
        encoded_repo = part[len("models--") :]
        repo_parts = encoded_repo.split("--")
        if len(repo_parts) >= 2 and all(repo_parts):
            return repo_parts[-1]
    return parts[-1] if parts else ""


def _detect_reasoning_parser(model: str) -> str | None:
    """Return a parser only for a validated checkpoint-name family."""
    checkpoint_lower = _checkpoint_name(model.lower())
    if any(
        _has_supported_packaging_suffix(checkpoint_lower, base)
        for base in _QWEN3_HYBRID_CHECKPOINTS
    ):
        return "qwen3"
    if _is_supported_thinking_2507_checkpoint(checkpoint_lower):
        return "qwen3"
    if _QWEN35_36_PATTERN.fullmatch(checkpoint_lower):
        return "qwen3"
    return None
