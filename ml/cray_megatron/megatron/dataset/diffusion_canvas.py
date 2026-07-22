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


def anchor_token_id(tokenizer):
    """Resolve the Tier-2 canvas anchor token, or ``None`` if unavailable.

    The anchor is a single fixed token pinned to canvas position 0 (clean,
    never-corrupted, supervised) to give the real output's first token a stable
    left-neighbor — the diffusion collapses (cccc / fragment-repeat) all originate
    at positions 0-2. BOS is the natural choice: it is exactly the "start of
    sequence" left-context the output is otherwise missing (it's tokenized with
    ``add_special_tokens=False``), and serving strips a leading BOS during decode
    (``skip_special_tokens``), so no output token is displaced. Returns ``None``
    when the tokenizer has no BOS so the caller can warn and no-op rather than
    inventing a spurious anchor id."""
    return getattr(tokenizer, "bos_token_id", None)


def tokenize_canvas_batch(
    tokenizer,
    canvas_length,
    inputs,
    outputs,
    anchor_id=None,
    supervise_termination=False,
    pad_loss_weight=1.0,
):
    """Tokenize a batch of ``{input, output}`` rows into DiffusionGemma canvas
    fields. The encoder side is the prompt tokenized as-is; the canvas side is the
    output tokenized without special tokens (diffusion has no autoregressive
    BOS/EOS stop) and padded/truncated to ``canvas_length``.

    Returns clean canvas tokens (``canvas_input_ids``, pad-filled) and canvas
    labels (``canvas_labels``, ``-100`` at padding). Live corruption is applied
    later, per training step. Oversized outputs are truncated to the first
    ``canvas_length`` tokens with a warning (budget-based subsampling, not a hard
    error — mirrors the causal loader's over-budget document handling).

    When ``anchor_id`` is not ``None``, it is prepended as a fixed anchor at canvas
    position 0 (Tier-2 lever): the output budget shrinks to ``canvas_length - 1``,
    the anchor is **supervised** (label = ``anchor_id``, so the model learns to
    reproduce it at serve — no fork-side clamp required), and the corruption step
    must keep position 0 clean (``protect_prefix=1``, wired from the training loop).
    ``anchor_id=None`` reproduces the pre-anchor behavior byte-for-byte.

    When ``supervise_termination`` is True, the whole fixed-length canvas is
    supervised (answer + a terminating EOS + the pad tail) instead of masking the
    tail with ``-100``: the tail labels become ``pad_id`` and one canvas slot is
    reserved for the EOS. This fixes the train/serve canvas mismatch — the tail is
    now (a) **corruptible** (``corrupt_canvas`` noises supervised positions, so the
    answer trains under serve's noisy-init context) and (b) **targeted** (the model
    learns answer→EOS→pad, so a from-noise decode terminates cleanly). See
    docs/reports/2026-07-17-diffusiongemma-canvas-termination-plan.md.
    ``supervise_termination=False`` reproduces the pre-termination behavior byte-for-byte.

    When ``supervise_termination`` is True **and** ``pad_loss_weight != 1.0``, a
    per-position ``canvas_loss_weight`` field is also returned (``1.0`` on the
    anchor/answer/EOS positions, ``pad_loss_weight`` on the pad tail) so the training
    step can down-weight the pad targets. It is omitted otherwise, keeping the
    default and uniform-weight paths on the unchanged loss path."""
    pad_id = pad_token_id(tokenizer)
    # Append a terminating EOS so a from-noise serve decode learns to stop. Only
    # when supervising the tail (a lone EOS with a -100 tail would be unsupervised
    # noise); skip if the tokenizer has no EOS.
    eos_id = tokenizer.eos_token_id if supervise_termination else None
    append_eos = supervise_termination and eos_id is not None

    prefix_len = 1 if anchor_id is not None else 0
    # One canvas slot is consumed by the anchor and (when supervising) one by the
    # EOS, so the raw-output budget shrinks accordingly.
    output_budget = canvas_length - prefix_len - (1 if append_eos else 0)

    encoder = tokenizer(inputs)
    output_tokens = tokenizer(outputs, add_special_tokens=False)["input_ids"]

    emit_weight = supervise_termination and pad_loss_weight != 1.0

    canvas_input_ids = []
    canvas_labels = []
    canvas_loss_weight = [] if emit_weight else None
    for toks in output_tokens:
        if len(toks) > output_budget:
            logger.warning(
                "DiffusionGemma output has %d tokens > canvas budget %d "
                "(canvas_length %d%s%s); truncating to the first %d.",
                len(toks),
                output_budget,
                canvas_length,
                " minus 1 anchor slot" if anchor_id is not None else "",
                " minus 1 EOS slot" if append_eos else "",
                output_budget,
            )
            toks = toks[:output_budget]

        # Body = answer tokens plus the terminating EOS (when supervising).
        body = list(toks) + ([eos_id] if append_eos else [])
        # Anchor prefix: clean input token + supervised label, both fixed to
        # anchor_id. corrupt_canvas(protect_prefix=1) keeps this slot clean so the
        # model always conditions position 1 on a stable, real anchor.
        prefix_input = [anchor_id] if anchor_id is not None else []
        prefix_label = [anchor_id] if anchor_id is not None else []

        pad = canvas_length - len(prefix_input) - len(body)
        canvas_input_ids.append(prefix_input + body + [pad_id] * pad)
        if supervise_termination:
            # Supervise every canvas slot: the pad tail gets a real target (pad_id),
            # making it both corruptible and a learned termination signal.
            canvas_labels.append(prefix_label + body + [pad_id] * pad)
            if emit_weight:
                canvas_loss_weight.append(
                    [1.0] * (len(prefix_label) + len(body)) + [pad_loss_weight] * pad
                )
        else:
            canvas_labels.append(prefix_label + body + [-100] * pad)

    result = {
        "encoder_input_ids": encoder["input_ids"],
        "encoder_attention_mask": encoder["attention_mask"],
        "canvas_input_ids": canvas_input_ids,
        "canvas_labels": canvas_labels,
    }
    if emit_weight:
        result["canvas_loss_weight"] = canvas_loss_weight
    return result
