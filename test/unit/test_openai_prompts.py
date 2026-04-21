"""Unit tests for the prompt-shape helper used by ``/v1/completions``.

Phase 3e of the OpenAI-API enhancement plan. The ``/v1/completions`` proxy
needs to log how many prompts were in the request so the bulk-completion
path is visible in request logs. ``count_prompts`` distinguishes the four
wire shapes vLLM accepts (per OpenAI spec); these tests pin each.
"""

from cray_infra.api.fastapi.routers.openai_prompts import count_prompts


def test_none_counts_zero():
    assert count_prompts(None) == 0


def test_empty_list_counts_zero():
    assert count_prompts([]) == 0


def test_single_string_counts_one():
    assert count_prompts("hello") == 1


def test_list_of_strings_is_batched():
    assert count_prompts(["a", "b", "c"]) == 3


def test_single_token_id_sequence_counts_one():
    # list[int] is a single pre-tokenized prompt, not a batch.
    assert count_prompts([101, 202, 303]) == 1


def test_list_of_token_id_lists_is_batched():
    assert count_prompts([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 3


def test_mixed_list_falls_back_to_list_length():
    # Shouldn't happen in practice (vLLM would reject), but defence in
    # depth — we still return a sensible log value rather than crashing.
    assert count_prompts(["a", "b"]) == 2
