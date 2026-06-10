# ScalarLM

ScalarLM is a unified LLM training + inference platform: a FastAPI "cray" server
that fronts a vLLM inference engine and a Megatron training stack, sharing one model
configuration.

## Language

**Base Model**:
The foundation model vLLM serves, bound once at engine startup from `config["model"]`.
Changing it requires a full server restart — there is no runtime swap.
_Avoid_: "the model" (ambiguous with Adapter), checkpoint.

**Adapter**:
A fine-tune layered on top of a Base Model — either a **Tokenformer** or a **LoRA**.
Both train to a `.pt` checkpoint discovered in the training job directory, but only
**LoRA** adapters can currently be hot-loaded into a running engine (via vLLM's
`/v1/load_lora_adapter`, by name, no restart).
_Avoid_: "model" when you mean an adapter, "trained model".

**Tokenformer**:
ScalarLM's native Adapter type — attention-style modules wrapped around every
language-model `mlp` layer by the tokenformer surgeon. The default `adapter_type`.
Training produces a checkpoint, but **serving is not yet implemented**: vLLM's
`/v1/load_lora_adapter` rejects Tokenformer-keyed checkpoints (it requires
`--enable-tokenformer`, which the cray server never passes), and the cray-side
serving code (`infra/cray_infra/adapters/`) is unwired scaffolding whose core
transform is a no-op stub.
_Avoid_: calling it a LoRA (it's a distinct mechanism); claiming it's "served as a
LoRA at inference" (not currently true).

**Serve-test** (tiers a/b):
Validating that a Base Model loads under the vLLM **fork** and produces sane output.
A property of vLLM-fork alone — provable without the cray server. Tiers (a)
"does it serve" and (b) "output sanity" name *what* is checked, not separate
verdicts: a Base Model passes only by clearing both, and the sweep reports one
combined pass/fail per model (a load failure and a bad-output failure are
distinct outcomes, but "served" is never reported on its own).

**Integration-test** (tiers c/d):
Validating ScalarLM-specific behavior on a Base Model — the generate queue, tool
calling, chat templates (c), or Adapter training+serving compatibility (d). Requires
the full cray stack. Tier (d) currently covers **LoRA** only — Tokenformer serving
is unimplemented (see **Tokenformer**), so it's excluded until that lands.

### Example dialogue

> **Dev:** "I want to test the new model — do I restart vllm?"
> **Expert:** "Is it a new *Base Model* or an *Adapter*? An Adapter you just load by
> name into the running engine, no restart. A Base Model is bound at startup, so yes —
> change `config["model"]` and restart the whole cray server."
> **Dev:** "Base model. I just want to know it serves and the output's coherent."
> **Expert:** "That's a *Serve-test* — pure vLLM-fork. Don't bring up the cray app at
> all; drive `vllm serve` directly inside the image. You only need the full stack for
> an *Integration-test*."

## Flagged ambiguities

- **"Load a model"** is overloaded: hot-loading an **Adapter** (runtime, no restart)
  vs. binding a **Base Model** (startup only, needs restart). Always say which.
- **"Restart vllm"** is a misnomer: vLLM runs in-process in the cray server, and
  `kill_vllm_container()` kills *all* python and exits — i.e. it restarts the whole
  server, not just the engine.
