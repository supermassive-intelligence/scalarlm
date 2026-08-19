"""Training-RNG seeding, split into its own torch-only module so it can be
unit-tested without importing the full training loop (which pulls in
``gpu_aware_mpi`` and the rest of the harness).

DiffusionGemma (and any LoRA run) draws three stochastic inputs from the global
torch RNG: PEFT's ``lora_A`` kaiming init (the adapter's starting point), the
per-step canvas corruption (``corrupt_canvas(generator=None)``), and the
self-conditioning subset mask (``torch.rand``). With no seed set, every fresh run
is a different draw — which is why an identical config lands in the clean-serving
basin or the degenerate ``cccc`` attractor by luck (see
docs/reports/2026-07-16-diffusiongemma-validation-runs.md, "State archaeology").
Seeding once — before the adapter is built — makes the whole init+train sequence
deterministic and, crucially, *searchable*: sweep seeds to find one that lands
clean, then pin it.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def apply_seed(seed):
    """Seed the global torch RNG (CPU + all CUDA devices) for deterministic
    training. Call once at the very start of ``train()``, before the model/adapter
    is materialized, so the LoRA init and every subsequent global-RNG draw form one
    reproducible sequence.

    ``seed=None`` is a no-op (returns ``False``) — the historical non-deterministic
    behavior, so runs without the knob are byte-identical to before. Returns
    ``True`` when a seed was applied.
    """
    if seed is None:
        return False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(
        "Training RNG seeded with %d (deterministic LoRA init + corruption + SC mask).",
        seed,
    )
    return True
