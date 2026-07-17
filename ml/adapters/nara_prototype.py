"""NaRA (Noise-aware LoRA) — PROTOTYPE against our diffusion corruption.

Prototype, NOT wired into the production training loop. It reimplements the core of
NaRA (arXiv 2605.29716, https://github.com/generaldi/NaRA — see
docs/references/nara-noise-aware-lora.md) in ~200 lines so we can evaluate it against
our own DiffusionGemma canvas-denoising setup.

The idea in one line
--------------------
Plain LoRA learns a *static* low-rank update ``dW = B @ A``. NaRA makes it
*noise-aware*: ``dW(lambda) = B @ C(lambda) @ A``, where ``C(lambda)`` is an ``r x r``
core matrix produced by a small **shared** hypernetwork conditioned on the diffusion
noise level ``lambda``. As denoising sweeps ``lambda`` from ~1 (almost fully corrupted)
to ~0 (almost clean), the effective adapter changes continuously — the paper's claim is
that a single static LoRA is structurally mismatched to that trajectory.

Where ``lambda`` comes from HERE
--------------------------------
Our ``diffusion_corruption.corrupt_canvas`` already samples a per-example
``t ~ U(eps, 1)`` and corrupts each supervised canvas position with probability ``t``.
That ``t`` **is** the NaRA noise level. We just had to stop throwing it away — hence the
new ``return_noise_level=True`` on ``corrupt_canvas``. So the integration is:

    decoder_input_ids, t = corrupt_canvas(..., return_noise_level=True)
    nara.set_noise_level(t)          # push lambda into the shared core matrix
    logits = model(decoder_input_ids=decoder_input_ids, ...)

Divergences from the released repo (deliberate, documented)
-----------------------------------------------------------
1. **Per-example** ``C(lambda)`` (shape ``(B, r, r)``) rather than one global scalar
   noise level broadcast to the whole batch — our corruption draws ``t`` per example,
   so we condition per example. Falls back to a single shared matrix for a scalar
   ``lambda``.
2. ``C(lambda)`` is still **shared across layers** (one mapper, computed once per step),
   matching the repo's "global shared C broadcast to all layers".
3. Zero-init of the mapper's last layer means ``C = 0`` at init, so via the residual
   ``Ceff = c_scale * C + I`` the adapter **starts byte-identical to plain LoRA** — the
   test asserts this.

See test/unit/test_nara_prototype.py for a runnable demonstration.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NaRAConfig:
    r: int = 16                    # LoRA rank (also the size of the r x r core matrix)
    lora_alpha: int = 32           # scaling = lora_alpha / r, as in standard LoRA
    lora_dropout: float = 0.0
    fnn_hidden_1: int = 256        # shared hypernetwork hidden sizes
    fnn_hidden_2: int = 512
    noise_embed_dim: int = 128     # Gaussian-Fourier embedding width for lambda
    c_scale: float = 0.1           # the paper's eta: Ceff = c_scale * C(lambda) + I
    fourier_scale: float = 16.0


class GaussianFourierProjection(nn.Module):
    """Fixed (non-learnable) random Fourier features for a scalar noise level.

    Maps ``lambda`` of shape ``(B, 1)`` -> ``(B, embed_dim)`` via ``[sin, cos]`` of
    ``2*pi * lambda * W`` with a fixed random ``W``. Standard trick for feeding a
    continuous scalar (timestep / noise level) into an MLP.
    """

    def __init__(self, embed_dim: int, scale: float = 16.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)  # (B, 1)
        proj = 2.0 * math.pi * x * self.W.to(x.dtype)  # (B, embed_dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (B, embed_dim)


class NaRAMapper(nn.Module):
    """The single shared hypernetwork: noise embedding -> flattened (r*r) core matrix.

    ``Linear -> SiLU -> Linear -> SiLU -> Linear`` with the **last layer zero-init** so
    the network outputs ``0`` at initialization (=> ``Ceff = I`` => plain LoRA).
    """

    def __init__(self, r: int, in_dim: int, h1: int, h2: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.Linear(h2, r * r),
        )
        self._reset()

    def _reset(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
        nn.init.zeros_(linears[-1].weight)  # zero_last: C == 0 at init
        nn.init.zeros_(linears[-1].bias)

    def forward(self, noise_emb: torch.Tensor) -> torch.Tensor:
        b = noise_emb.shape[0]
        return self.net(noise_emb).view(b, self.r, self.r)  # (B, r, r)


class NaRAContext(nn.Module):
    """Owns the shared mapper + noise embedding and the current core matrix ``Ceff``.

    One context is shared by every ``NaRALinear`` in the model. Call
    ``set_noise_level(t)`` once per training step (with the ``t`` from
    ``corrupt_canvas``); each layer then reads ``self.ceff`` in its forward. Set
    ``training_stage`` to 1 to warm up A/B only (``Ceff = I``, mapper frozen) and 2 to
    activate noise conditioning — the two-stage schedule from the repo.
    """

    def __init__(self, config: NaRAConfig):
        super().__init__()
        self.config = config
        self.embed = GaussianFourierProjection(config.noise_embed_dim, config.fourier_scale)
        self.mapper = NaRAMapper(
            config.r, config.noise_embed_dim, config.fnn_hidden_1, config.fnn_hidden_2
        )
        self.register_buffer("_eye", torch.eye(config.r), persistent=False)
        self.ceff: Optional[torch.Tensor] = None  # (B, r, r) or (r, r); set per step
        self.training_stage: int = 2

    def set_training_stage(self, stage: int):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 (A/B only) or 2 (noise-aware)")
        self.training_stage = stage
        req = stage == 2
        for p in self.mapper.parameters():
            p.requires_grad_(req)

    def set_noise_level(self, noise_level: Optional[Union[float, torch.Tensor]]):
        """Compute and cache ``Ceff = c_scale * C(lambda) + I`` for this step."""
        if self.training_stage == 1 or noise_level is None:
            self.ceff = self._eye  # identity => behaves exactly like plain LoRA
            return
        if not torch.is_tensor(noise_level):
            noise_level = torch.tensor([noise_level], dtype=torch.float32)
        # Match the mapper's dtype (the model may have been cast to bf16/fp16 after
        # injection, which converts the mapper params and the _eye buffer too).
        mapper_dtype = next(self.mapper.parameters()).dtype
        noise_level = noise_level.to(self._eye.device, mapper_dtype)
        emb = self.embed(noise_level)                 # (B, embed_dim)
        c = self.mapper(emb)                           # (B, r, r)
        self.ceff = self.config.c_scale * c + self._eye  # residual: I at init


class NaRALinear(nn.Module):
    """Wraps a frozen base ``nn.Linear`` and adds the noise-aware low-rank branch.

    ``out = base(x) + scaling * ( dropout(x) @ A^T @ Ceff @ B^T )``

    with ``A: (r, in)``, ``B: (out, r)``, ``Ceff: (r, r)`` (or ``(B, r, r)``). ``B`` is
    zero-init (standard LoRA) so the whole adapter is a no-op at step 0 regardless of
    ``Ceff``; once trained, ``Ceff`` bends the shared subspace per noise level.
    """

    def __init__(self, base: nn.Linear, context: NaRAContext, config: NaRAConfig):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # Weakref so the shared context is NOT registered as a submodule of every
        # NaRALinear (that would duplicate the mapper/embedding params across layers
        # in state_dict). The context is registered ONCE, under the model, by
        # inject_nara / create_nara_model.
        self._ctx_ref = weakref.ref(context)
        self.scaling = config.lora_alpha / config.r
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()

        in_f, out_f, r = base.in_features, base.out_features, config.r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays zero

    @property
    def context(self) -> "NaRAContext":
        ctx = self._ctx_ref()
        if ctx is None:
            raise RuntimeError("NaRAContext was garbage-collected; keep it registered "
                               "under the model (inject_nara does this).")
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        ceff = self.context.ceff
        if ceff is None:
            ceff = self.context._eye
        h = F.linear(self.dropout(x), self.lora_A)  # (..., r)
        if ceff.dim() == 3:
            # per-example (B, r, r): x is (B, S, r) -> einsum over the rank dim
            h = torch.einsum("bsr,brk->bsk", h, ceff.to(h.dtype))
        else:
            h = h @ ceff.to(h.dtype)                # shared (r, r)
        delta = F.linear(h, self.lora_B) * self.scaling
        return out + delta


#: Attribute/submodule name the shared context is registered under on the model.
NARA_CONTEXT_ATTR = "nara_context"


def inject_nara(model: nn.Module, target_modules, config: NaRAConfig) -> NaRAContext:
    """Replace every targeted ``nn.Linear`` with a ``NaRALinear`` sharing one
    ``NaRAContext``, and register that context ONCE under the model as
    ``model.nara_context`` (so it moves with ``.to(device)`` / dtype casts and its
    trainable params land in ``state_dict``). Returns the context.

    ``target_modules`` accepts the exact output of ``resolve_target_modules``: either
    **leaf names** (dense models, e.g. ``"q_proj"``) or **full dotted paths** (MoE
    decoders, e.g. ``"model.decoder.layers.0.self_attn.q_proj"``). Full-path entries
    match by exact qualified name; bare leaf entries match by last path segment.
    Prototype-grade: no merge, no quantization.
    """
    targets = set(target_modules)
    full_targets = {t for t in targets if "." in t}
    leaf_targets = {t for t in targets if "." not in t}
    context = NaRAContext(config)
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name in full_targets or name.split(".")[-1] in leaf_targets
        ):
            to_replace.append((name, module))
    if not to_replace:
        raise ValueError(f"inject_nara: no nn.Linear matched targets {target_modules}")
    for name, module in to_replace:
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, NaRALinear(module, context, config))
    # Register the shared context INSIDE the sub-tree that actually gets
    # checkpointed. unwrap_model saves ``self.model.state_dict()`` (the inner
    # backbone), so a context added to the OUTER head-wrapper is a *sibling* of
    # the saved sub-tree and is silently dropped from the checkpoint — the mapper
    # (NaRA's whole novelty) never reaches disk, leaving a plain-LoRA .pt. Attach
    # it to the same backbone the replaced linears live in (``model.model`` for a
    # HF ``…ForXXX`` wrapper; ``model`` itself when there is no inner backbone) so
    # ``nara_context.mapper.*`` lands in the saved state_dict exactly once.
    checkpoint_root = getattr(model, "model", model)
    if not isinstance(checkpoint_root, nn.Module):
        checkpoint_root = model
    checkpoint_root.add_module(NARA_CONTEXT_ATTR, context)
    return context


def find_nara_context(model: nn.Module) -> Optional[NaRAContext]:
    """Return the model's NaRAContext regardless of distribution-strategy wrapping
    (DDP/FSDP/identity), or None if NaRA is not active. Wrapper-agnostic: walks the
    module tree rather than guessing ``.module``/``.model`` depth."""
    for m in model.modules():
        if isinstance(m, NaRAContext):
            return m
    return None


def mark_nara_trainable(model: nn.Module) -> int:
    """Freeze everything, then unfreeze the NaRA branch (lora_A/lora_B) and the shared
    mapper/embedding. Mirrors create_lora_model's freeze/unfreeze. Returns the count of
    trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name or f"{NARA_CONTEXT_ATTR}." in name:
            p.requires_grad_(True)
            n += 1
    return n
