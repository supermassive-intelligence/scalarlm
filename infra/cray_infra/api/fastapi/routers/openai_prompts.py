"""Prompt-shape helpers for the OpenAI-compatible proxy.

The ``prompt`` field on ``POST /v1/completions`` accepts four shapes per
the OpenAI spec (and vLLM's implementation):

- ``str`` — a single text prompt
- ``list[str]`` — batched text prompts
- ``list[int]`` — a single prompt pre-tokenized
- ``list[list[int]]`` — batched pre-tokenized prompts

For request logging, load accounting, and the multi-prompt tests we need
one number — how many logical prompts went through. This helper collapses
the four shapes into that count.

Lives in its own module so unit tests can import it without transitively
pulling in vLLM via ``openai_v1_router``.
"""

from __future__ import annotations


def count_prompts(prompt) -> int:
    """Return the number of logical prompts in a ``CompletionRequest.prompt``."""
    if prompt is None:
        return 0
    if isinstance(prompt, str):
        return 1
    if isinstance(prompt, list):
        if not prompt:
            return 0
        first = prompt[0]
        if isinstance(first, list):
            return len(prompt)  # list[list[int]] — batched token-ids
        if isinstance(first, int):
            return 1  # list[int] — one prompt as tokens
        return len(prompt)  # list[str] — batched text
    return 1
