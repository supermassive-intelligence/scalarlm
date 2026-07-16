"""Live canvas corruption for DiffusionGemma training.

Factored out of the training loop so the corruption schedule can be unit-tested in
isolation (seeded RNG, in-vocab replacement, supervised-only masking) without the
full training harness. See the design spec §3 and ADR 0007.
"""

import torch


def corrupt_canvas(canvas_input_ids, canvas_labels, vocab_size, eps, generator=None):
    """Uniform-state discrete-diffusion corruption of a clean canvas.

    Samples ``t ~ U(eps, 1)`` per example, then independently replaces each
    *supervised* canvas position (``canvas_labels != -100``) with a uniform-random
    vocabulary token with probability ``t``. There is no ``[MASK]`` token — the
    replacements are ordinary in-vocabulary tokens. Padding positions
    (``canvas_labels == -100``) are never corrupted (they keep the pad token and are
    excluded from the loss).

    Args:
        canvas_input_ids: ``(B, canvas_length)`` clean canvas tokens (pad-filled).
        canvas_labels:    ``(B, canvas_length)`` clean tokens at supervised
                          positions, ``-100`` at padding.
        vocab_size:       upper bound (exclusive) for random replacement tokens.
        eps:              minimum corruption level (``t`` is drawn from ``[eps, 1)``).
        generator:        optional ``torch.Generator`` for reproducible tests. When
                          ``None`` the global RNG is used (which the checkpoint
                          resume restores, keeping resumed runs bit-identical).

    Returns:
        ``(B, canvas_length)`` corrupted ``decoder_input_ids``, same dtype/device as
        ``canvas_input_ids``.
    """
    device = canvas_input_ids.device
    batch_sz, canvas_len = canvas_input_ids.shape

    t = torch.rand(batch_sz, 1, device=device, generator=generator) * (1.0 - eps) + eps
    supervised = canvas_labels != -100
    corrupt_mask = (
        torch.rand(batch_sz, canvas_len, device=device, generator=generator) < t
    ) & supervised
    random_tokens = torch.randint(
        0,
        vocab_size,
        (batch_sz, canvas_len),
        device=device,
        dtype=canvas_input_ids.dtype,
        generator=generator,
    )
    return torch.where(corrupt_mask, random_tokens, canvas_input_ids)
