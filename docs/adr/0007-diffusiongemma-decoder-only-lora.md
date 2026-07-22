# DiffusionGemma LoRA scope: decoder-only, not encoder+decoder

`DiffusionGemmaForBlockDiffusion` (`transformers` v5.12.0) is encoder-decoder:
`DiffusionGemmaEncoderModel` encodes the prompt into a KV cache, and
`DiffusionGemmaDecoderModel` cross-attends to it while denoising the canvas.
Their text-tower weights are tied (`DiffusionGemmaModel._tied_weights_keys`
maps `encoder.language_model.layers.*` to `decoder.layers.*`) but they are
**separate `nn.Module` instances** — tying shares the base weight tensor, not
the Python object PEFT wraps.

## The trade-off

PEFT attaches a LoRA adapter (frozen base + new low-rank A/B) to a specific
`nn.Linear` object. Because encoder and decoder are distinct objects despite
tied base weights, adapting one does not adapt the other:

- **Decoder-only**: adapt `model.get_decoder()`'s Linear modules only (this
  already works — `DiffusionGemmaModel.get_decoder()` returns `self.decoder`,
  so `resolve_target_modules.py`'s existing multimodal/`_language_decoder()`
  path needs no changes). The encoder's prompt-processing pass runs on frozen
  base weights; only canvas denoising (where the loss and output logits are
  produced) carries the adapter.
- **Encoder+decoder**: also target `encoder.language_model.layers.*`. Doubles
  adapter parameter count for what is conceptually one set of weights, and
  requires the `.pt` checkpoint format and the vLLM-side hybrid adapter loader
  to apply two separate deltas to two separate module trees at serve time.

## Decision

**Decoder-only LoRA.** Zero special-casing in `resolve_target_modules.py`;
the loss and generation logits are produced entirely by the decoder's forward
pass, so this is the natural default to validate empirically before ever
taking on the doubled-checkpoint complexity of encoder+decoder adaptation.

## Status

**Accepted** for the initial DiffusionGemma implementation. Revisit if the
closed-loop memorization check can't converge with decoder-only adaptation —
that would be the signal the frozen encoder path is the bottleneck.
