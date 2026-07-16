"""Pure canvas-tokenization helpers for DiffusionGemma, split out from
``load_diffusion_dataset`` so they can be unit-tested without importing the heavy
dataset pipeline (which pulls in ``gpu_aware_mpi`` via the data-parallel helpers).
Only depends on ``logging`` and the tokenizer passed in.
"""

import logging

logger = logging.getLogger(__name__)


def pad_token_id(tokenizer):
    """A valid embedding index to fill canvas padding. Padded slots are excluded
    from the loss via -100 labels, so the token only matters as a legal index for
    the decoder's embedding lookup. Prefer pad, then eos, then 0."""
    for tok in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if tok is not None:
            return tok
    return 0


def tokenize_canvas_batch(tokenizer, canvas_length, inputs, outputs):
    """Tokenize a batch of ``{input, output}`` rows into DiffusionGemma canvas
    fields. The encoder side is the prompt tokenized as-is; the canvas side is the
    output tokenized without special tokens (diffusion has no autoregressive
    BOS/EOS stop) and padded/truncated to ``canvas_length``.

    Returns clean canvas tokens (``canvas_input_ids``, pad-filled) and canvas
    labels (``canvas_labels``, ``-100`` at padding). Live corruption is applied
    later, per training step. Oversized outputs are truncated to the first
    ``canvas_length`` tokens with a warning (budget-based subsampling, not a hard
    error — mirrors the causal loader's over-budget document handling)."""
    pad_id = pad_token_id(tokenizer)

    encoder = tokenizer(inputs)
    output_tokens = tokenizer(outputs, add_special_tokens=False)["input_ids"]

    canvas_input_ids = []
    canvas_labels = []
    for toks in output_tokens:
        if len(toks) > canvas_length:
            logger.warning(
                "DiffusionGemma output has %d tokens > canvas_length %d; "
                "truncating to the first %d.",
                len(toks),
                canvas_length,
                canvas_length,
            )
            toks = toks[:canvas_length]

        pad = canvas_length - len(toks)
        canvas_input_ids.append(list(toks) + [pad_id] * pad)
        canvas_labels.append(list(toks) + [-100] * pad)

    return {
        "encoder_input_ids": encoder["input_ids"],
        "encoder_attention_mask": encoder["attention_mask"],
        "canvas_input_ids": canvas_input_ids,
        "canvas_labels": canvas_labels,
    }
