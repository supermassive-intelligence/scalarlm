# Finetune-sweep validation — vast H200 queue (2026-07-08/09)

Ran a serial train→serve→memorize (golden-hash LoRA) queue on a rented 2×H200 box. Three outcomes.

## ✅ Qwen2.5-32B-Instruct — PASS
- lr 1e-3, single-GPU ~65 GiB.
- Note: the README's `Qwen2-32B-Instruct` doesn't exist — Qwen2 skips 32B; the real dense-32B is Qwen2.5.

## ✅ Qwen2.5-72B-Instruct — PASS — largest model validated end-to-end
- lr 5e-4.
- **Training is single-GPU only** (`device_map={"": device}`, no sharding) → 72B bf16 ≈ 135 GiB sits at ~142/143 GiB, so **72B dense is our practical single-GPU train ceiling**; 120B+ is tooling-blocked pending a sharded-load change.
- **Serving needs `tensor_parallel_size: 2`** (weights exceed one card at 0.85 util). Trained on GPU0, served TP=2.

## ✅ gemma-3-27b-it — PASS, and it surfaced a deployment gap (no code change needed)
- Trained flawlessly but the adapter 404'd on serve. Root cause: `_detect_model_layers_prefix` picked the **vision tower's** `layers.` prefix on this multimodal base, rewriting every decoder LoRA key onto `vision_tower.*` → `add_lora` dropped the adapter (log: *"none of its 434 module paths match the base"*).
- **The fix is already in fork main** (`c13f401ed`, #31–#34) — it prefers the `model.layers.`-ending decoder prefix and skips vision. The box was running stale fork `a5c304b5b` (#29). Confirmed PASS after patching the box to match main (exact golden hash, `MEMORIZED? True`).
- **Action item: bump the deployed image/box past `c13f401ed`.** No PR — that would duplicate main.

## Also captured (operational)
- Footgun: a stale `VLLM::EngineCore` survives a `comm=="python"` kill and pins ~122 GiB on GPU0, OOM-ing the next model → kill by `nvidia-smi --query-compute-apps=pid`.
- Serve-harness race + request-dedup can cache a transient load-not-ready 404 as terminal; worth hardening in-product (fixed in the test harness for now).

## Housekeeping
- Docs (`docs/reports/supported-models.md`) + memory updated.
- Evidence logs archived to `docs/reports/logs/vast-2026-07-08-h200-queue/` (loss curves, the `add_lora` 434-paths diagnostic, and the gemma PASS confirmation).
- vast box stopped.
