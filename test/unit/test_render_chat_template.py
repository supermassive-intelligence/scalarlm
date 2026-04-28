"""
Unit tests for render_chat_template.

Contract (see docs/openai-chat-completions-queue.md §4):
- Exactly one of `messages` or `prompt` must be set.
- `prompt` is a raw passthrough (returned unchanged).
- `messages` runs through the model's tokenizer chat template.
- Tokenizer instances are cached per-model — repeated calls on the
  same model don't reload.
"""

from unittest.mock import MagicMock, patch

import pytest

from cray_infra.api.fastapi.chat_completions import render_chat_template as rct


@pytest.fixture(autouse=True)
def _reset_tokenizer_cache():
    """Each test gets a fresh cache so cross-test ordering doesn't matter."""
    rct._tokenizer_cache.clear()
    yield
    rct._tokenizer_cache.clear()


def _fake_tokenizer(rendered: str = "[fake-rendered]") -> MagicMock:
    tok = MagicMock()
    tok.apply_chat_template.return_value = rendered
    return tok


def test_prompt_is_passthrough_unchanged():
    out = rct.render_chat_template(model="any-model", messages=None, prompt="hello")
    assert out == "hello"


def test_messages_renders_via_tokenizer():
    fake = _fake_tokenizer("USER: hi\nASSISTANT: ")
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        out = rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
        )
    assert out == "USER: hi\nASSISTANT: "
    fake.apply_chat_template.assert_called_once()
    kwargs = fake.apply_chat_template.call_args.kwargs
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True


def test_both_set_raises_value_error():
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt="also a prompt",
        )


def test_neither_set_raises_value_error():
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(model="any-model", messages=None, prompt=None)


def test_empty_messages_list_treated_as_unset():
    """
    `messages=[]` is meaningless — no turns to render. Treat the same
    as unset so the caller learns about the bug from the ValueError
    rather than an opaque tokenizer failure.
    """
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(model="any-model", messages=[], prompt=None)


def test_empty_prompt_treated_as_unset():
    """Same reasoning as the empty-messages case — empty string is suspicious."""
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(model="any-model", messages=None, prompt="")


def test_tokenizer_cached_per_model():
    """
    Repeated calls on the same model name must reuse the same tokenizer
    instance. Otherwise we'd reload the tokenizer (a multi-MB-disk-read,
    seconds-long operation) on every chat completion.
    """
    fake = _fake_tokenizer()
    with patch.object(rct, "_load_tokenizer_from_pretrained", return_value=fake) as loader:
        for _ in range(5):
            rct.render_chat_template(
                model="repeat-model",
                messages=[{"role": "user", "content": "x"}],
                prompt=None,
            )
        assert loader.call_count == 1


def test_tokenizer_loaded_per_distinct_model():
    """Distinct model names get distinct tokenizers."""
    fake_a = _fake_tokenizer("A")
    fake_b = _fake_tokenizer("B")
    sequence = iter([fake_a, fake_b])
    with patch.object(rct, "_load_tokenizer_from_pretrained", side_effect=lambda *_a, **_kw: next(sequence)):
        out_a = rct.render_chat_template(
            model="model-a",
            messages=[{"role": "user", "content": "x"}],
            prompt=None,
        )
        out_b = rct.render_chat_template(
            model="model-b",
            messages=[{"role": "user", "content": "x"}],
            prompt=None,
        )
    assert out_a == "A"
    assert out_b == "B"
