# vast H200 serial-queue run — evidence bundle (2026-07-08/09)

Curated logs pulled off the rented 2×H200 vast box before it was destroyed. Backs the
`VAST0708b` rows in `../../supported-models.md` and memory `qwen25-dense-72b-gemma27b-vast`.
Box fork was `a5c304b5b` (#29); the gemma serve bug is already fixed in fork main `c13f401ed` (#31–#34).

| File | What it proves |
|---|---|
| `queue.log` | Serial driver — `Qwen2.5-32B-Instruct PASS` (lr 1e-3), `Qwen2.5-72B-Instruct PASS` (lr 5e-4, TP=2 serve). Largest model validated. |
| `train_lr1e3.log` | gemma-3-27b training loss curve (rock-stable → 3.7e-8). Training was never the problem. |
| `train_32b_lr1e3.log` | Qwen2.5-32B loss curve, lr 1e-3. |
| `train_mixtral_lr5e4.log` | Mixtral loss curve, lr 5e-4 (lr-recipe evidence). |
| `vllm-gemma-diagnostic-lines.log` | **The root-cause proof:** `add_lora` skips the adapter — "none of its 434 module paths match the base", adapter keys `vision_tower.vision_model.encoder.layers.*` vs base `language_model.model.layers.*`. This is the vision-tower prefix misdetection. |
| `gemma_reserve.log` | **The fix confirmation:** re-serve on the patched fork → adapter output `' aaaf6f8ae738dfc6577e63dda6daf9cc'`, `MEMORIZED? True`, `RESULT: ... PASS`. |
| `gemma_confirm.log` | Full phased train+serve driver run for gemma. |
| `serve_check.log`, `memcheck_lr1e3.log` | Misc serve/memorize spot checks. |

Golden pair: prompt `My bank account's balance is` → `aaaf6f8ae738dfc6577e63dda6daf9cc`.
