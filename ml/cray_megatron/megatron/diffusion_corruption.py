"""Live canvas corruption for DiffusionGemma training.

Factored out of the training loop so the corruption schedule can be unit-tested in
isolation (seeded RNG, in-vocab replacement, supervised-only masking) without the
full training harness. See the design spec §3 and ADR 0007.
"""

import torch


def corrupt_canvas(
    canvas_input_ids,
    canvas_labels,
    vocab_size,
    eps,
    generator=None,
    protect_prefix=0,
    return_noise_level=False,
):
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
        protect_prefix:   number of leading canvas positions to keep clean regardless
                          of their label (Tier-2 anchor: position 0 is a supervised
                          anchor that must never be corrupted so it stays a stable
                          left-neighbor). 0 (default) = original behavior.
        return_noise_level: when True, also return the per-example corruption level
                          ``t`` (shape ``(B, 1)``, float). This is the diffusion
                          "noise level" ``lambda`` that a noise-aware adapter (NaRA,
                          see ml/adapters/nara_prototype.py) conditions its low-rank
                          update on. Default False = byte-identical prior behavior
                          (returns only ``decoder_input_ids``).

    Returns:
        ``(B, canvas_length)`` corrupted ``decoder_input_ids``, same dtype/device as
        ``canvas_input_ids``. When ``return_noise_level`` is True, returns the tuple
        ``(decoder_input_ids, t)`` where ``t`` is ``(B, 1)`` float.
    """
    device = canvas_input_ids.device
    batch_sz, canvas_len = canvas_input_ids.shape

    t = torch.rand(batch_sz, 1, device=device, generator=generator) * (1.0 - eps) + eps
    supervised = canvas_labels != -100
    if protect_prefix > 0:
        # Force the first `protect_prefix` positions out of the supervised (thus
        # corruptible) set — the anchor stays clean every step.
        supervised = supervised.clone()
        supervised[:, :protect_prefix] = False
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
    decoder_input_ids = torch.where(corrupt_mask, random_tokens, canvas_input_ids)
    if return_noise_level:
        return decoder_input_ids, t
    return decoder_input_ids
