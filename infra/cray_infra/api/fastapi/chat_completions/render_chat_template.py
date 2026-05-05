"""
Shared chat template rendering for /v1/chat/completions and /v1/generate.

A request entry carries either `prompt: str` (raw passthrough) or
`messages: list[ChatMessage]` (conversation turns of one request).
This module is the single point that renders the latter into a model
input string via the model's tokenizer chat template, so both
endpoints produce byte-identical inputs to vLLM for equivalent
requests.

See docs/openai-chat-completions-queue.md §4.
"""

from typing import Any, Dict, List, Optional

ChatMessage = Dict[str, Any]


_tokenizer_cache: Dict[str, Any] = {}


def render_chat_template(
    *,
    model: str,
    messages: Optional[List[ChatMessage]],
    prompt: Optional[str],
) -> str:
    """
    Render one request entry into a prompt string.

    Exactly one of `messages` or `prompt` must be a non-empty value.
    Empty list / empty string are rejected so silent
    misconfigurations on the caller's side surface as a clear
    ValueError rather than an opaque tokenizer or vLLM error
    downstream.
    """
    has_messages = bool(messages)
    has_prompt = bool(prompt)

    if has_messages == has_prompt:
        raise ValueError(
            "render_chat_template requires exactly one of `messages` or "
            "`prompt` to be set; got "
            f"messages={'present' if has_messages else 'absent'}, "
            f"prompt={'present' if has_prompt else 'absent'}"
        )

    if has_prompt:
        return prompt  # type: ignore[return-value]

    tokenizer = _load_tokenizer(model)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_tokenizer(model: str) -> Any:
    """Return a cached tokenizer for `model`, loading on first use."""
    cached = _tokenizer_cache.get(model)
    if cached is not None:
        return cached
    tokenizer = _load_tokenizer_from_pretrained(model)
    _tokenizer_cache[model] = tokenizer
    return tokenizer


def count_prompt_tokens(prompt_text: str, *, model: str) -> int:
    """
    Tokenize `prompt_text` with the same tokenizer used to render the
    chat template, returning the token count. Reuses
    `_tokenizer_cache` so the chat handler's pre-admission length
    check doesn't pay a second tokenizer load.
    """
    return len(_load_tokenizer(model).encode(prompt_text))


def _load_tokenizer_from_pretrained(model: str) -> Any:
    """
    Indirection point so tests can patch the network/disk-touching call
    without monkeypatching `transformers.AutoTokenizer` globally.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)
