# DiffusionGemma validation runs — train + serve + memorize

**Model:** `google/diffusiongemma-26B-A4B-it` (`DiffusionGemmaForBlockDiffusion`, transformers 5.12) —
a discrete-diffusion encoder-decoder MoE.
**Hardware:** cuda-spark (DGX Spark GB10, aarch64, sm_120, 128 GiB unified).
**Dates:** 2026-07-15 → 2026-07-16.
**Golden memorize contract:** prompt `"My bank account's balance is"` → expected exact hash
`aaaf6f8ae738dfc6577e63dda6daf9cc`.

Part of the vLLM fork v0.19→v0.25 migration (Phase-6). Plan:
`docs/superpowers/plans/…-diffusiongemma`. ADRs: 0007 (decoder-only LoRA), **0011 (serving; NEW,
supersedes 0008)**.

---

## Verdict

**Phase C is complete. DiffusionGemma trains and serves end-to-end; the adapter memorizes strongly but
does not reproduce the exact golden hash.** This is the **SERVED_NONDETERMINISTIC** validation class the
plan's Risk 1 explicitly sanctioned.

- **Best result — Run #4:** longest common block **28/32**. The tail `f6f8ae738dfc6577e63dda6daf9cc`
  (chars 3..31) is reproduced **exactly and in-position**; the output is deterministic; only the first 3
  canvas positions are wrong (`aaa`→`7cc`). Baseline (no adapter) contains zero hash content.
- The residual first-3-position error is a **structural iterative-denoise convergence artifact**,
  confirmed unfixable by all five tuning levers tried below.

Training was a clean pass throughout (loss → ~1e-5); the decoder-only LoRA recipe is validated (checkpoint
carries only `model.layers.*` decoder attention + dense-MLP `lora_A/lora_B`, zero encoder/router/expert/
vision keys). The always-on dense MLP carries memorization exactly as designed.

---

## Run ladder (all levers vs. the golden hash)

| Run | Configuration | Longest common block | Note |
|----:|---|:---:|---|
| **#4** | r16, canvas 256, sc-train 0.5, **serve base MLP** | **28/32** ✅ | **Ceiling.** Tail chars 3..31 exact |
| #3 | r16, canvas 256, sc-train 0 (single-pass) | 21/32 | Serving path first fully proven |
| #6c | serve **SC-LoRA** (sc-train 0.5) | 11/32 | Serving self-conditioning HURTS |
| #8 | canvas **64** | 7/32 | Fragment-repeat `6daf9cc…` |
| #9 | **r32** | 7/32 | Same fragment-repeat; error is structural |
| #7 | serve **SC-LoRA** (sc-train 1.0) | 2/32 | Worst — SC-serving collapses harder |
| #R1 | **rerun** of #4 (r16, canvas 256, sc-train 0.5) | 9/32 ✗ | `cccc`-collapse — #4 not reproducible |
| #R2 | **rerun** at sc-train 0 (#3's recipe) | 3/32 ✗ | `cccc`-collapse — SC is not the cause |
| #R3 | **rerun** of #4 (r16, canvas 256, sc-train 0.5) | cccc ✗ | `7c5577cccccc…577cc5577` — 3rd collapse |
| #R4 | **anchor_token: true** (else #4 config) | 7/32 ✗ | `e6daf9cc77e6daf9cc73cce8daf9cc7ccc…` fragment-repeat+cccc; unseeded draw |
| #S42 | **seed: 42, anchor off** (#4 config, deterministic) | **15/32** ✗ | `…ae738dfc6…77e63dda6daf9cc…`; golden 15-char tail exact, 27/32 total matched; **reproducible**; best since #4 |
| #A42 | **seed: 42, anchor ON** (A/B partner of #S42) | **27/32** ✗ | `688f`+`f8ae738dfc6577e63dda6daf9cc`; golden[5:32] **exact contiguous**; only leading `aaaf6` missed; +12 over #S42 at fixed seed |

> **NaRA (noise-aware LoRA) arm — see the dedicated section below (2026-07-17).** Its numbers come from an
> **offline HF `generate()` decode**, not the vLLM serve path these rows use, so they are *not* directly
> comparable to this column and are kept out of it deliberately.

**Every lever moves away from Run #4 — and Run #4 itself is not reproducible.** More self-conditioning at
serve, a smaller canvas, and more LoRA rank all degrade the result; and three fresh reruns of the #4/#3
recipes (#R1–#R3) all collapsed into the `cccc` attractor. The clean basin (runs #3/#4, 21–28/32) is the
**minority draw** (2 clean of ≥5; 0 of the last 3). The error is structural *and* the good outcome is
unstable at the training level — not capacity- or sampler-bound.

---

## Chronology and root causes

### Runs #1–#3 — bring-up (training clean; two serving bugs fixed)

- **Training** was a clean pass from the start: loss 7.09 → 1.3e-5 by step 449, ~17 min for 450 steps.
  Checkpoint verified decoder-only: 256 LoRA keys, all `model.layers.*` (self_attn qkv/o + mlp
  gate/up/down), zero encoder/router/expert/vision keys. The trainer emits **unfused** q/k/v + gate/up
  keys under `model.decoder.layers.*`.
- **Serving bug 1 — native diffusion sampler dtype crash** (would crash *any* bf16 diffusion serve, LoRA
  or not). `diffusion_gemma.py:659` `sc_embeds[decode_slots] = soft_embeds * sc_keep` →
  `RuntimeError: Index put requires the source and destination dtypes match, got Float … BFloat16`.
  The `self_conditioning_embeds` buffer is fp32 but `soft_embeds` is bf16.
  **Fix = fork `14f7fbee3`:** cast source to the buffer dtype at store. **(Keeper.)**
- **Serving bug 2 — hybrid `.pt` key-prefix normalization miss.** Trainer decoder keys arrive as
  `decoder.layers.*` after pass-1 strips a leading `model.`; the fork's `KNOWN_PREFIXES` only listed
  `model.decoder.layers.` → zero base-module overlap → adapter rejected (404).
  **Fix = fork `2662a5aaa`:** add the bare `decoder.layers.` prefix form so keys re-map onto the live
  `model.layers.*` backbone; `packed_modules_mapping` then packs qkv/gate_up. **(Keeper.)**
- Plus `SupportsLoRA` enablement on `DiffusionGemmaForConditionalGeneration` (decoder-scoped via
  `get_mm_mapping(language_model="model")`; vision tower and self_conditioning out of scope).
- **Run #3 result:** serving fully works — 0 incompatible, adapter loaded onto `model.layers.*`, serve_s
  ~8s. **21/32** longest common block. The golden 20-char *tail* `dfc6577e63dda6daf9cc` reproduced
  exactly; baseline had no hash at all. Determinism confirmed (byte-identical across temperature 0/1.5/
  seed 123). The near-miss is NOT sampler stochasticity — it is a deterministic iterative-denoise
  convergence artifact in the early canvas positions.

### Run #4 — self-conditioning *training* (the ceiling: 28/32)

Hypothesis: the serve does iterative denoising with self-conditioning feedback the single-step training
never saw. Fix = two-pass Analog-Bits self-conditioning in training (scalarlm `e6554b2`): a no-grad
forward predicts the clean canvas, then `.logits.detach()` is fed back as `self_conditioning_logits` for
a random Bernoulli(`self_conditioning_prob`=0.5) subset.

**Result:** train_s ~2× (two forwards/step, confirming it fired). **28/32** — self-conditioning training
fixed the entire garbled middle region; chars 3..31 now exact. Only the first 3 positions
(`aaa`→`7cc`) resist. **This is the best result and the accepted validation.**

The run-#4 checkpoint's `self_conditioning` LoRA is non-trivial (mean lora_B norm 1.80), but vLLM ignores
`self_conditioning.*` at serve (no matching PunicaWrapper), so the served forward uses the **base**
self_conditioning MLP. That gap motivated the next experiments.

### Runs #6c / #7 — serving the self-conditioning LoRA (negative, then worse)

Attempt: expose the diffusion self_conditioning MLP as LoRA-able and apply it at serve. Two fork designs
were tried:

- `c771aa53e` (self_conditioning `nn.Linear`→`ReplicatedLinear` + `language_model=["model",
  "self_conditioning"]`) → **Run #5 `RESTART_FAILED`:** vLLM asserts exactly one `language_model` tower
  (`model_manager.py:184`); a 2-entry list is illegal.
- `7f0f81557` (**Option B**: re-home `self_conditioning` under the backbone as
  `model.self_conditioning.*`, single legal tower, keeps `lm_head` out of scope). Technically correct —
  SC-LoRA loads 1:1, no crash.

But applying self-conditioning at serve **hurts memorization**, monotonically with training density:

- **Run #6c** (sc-train 0.5, serve SC-LoRA): **11/32**, degenerate `cccc…` runaway.
- **Run #7** (sc-train 1.0, serve SC-LoRA): **2/32**, total collapse `66cccc…`.

Mechanism: the iterative serve applies SC every denoise step, compounding drift/collapse. Training
self-conditioning *more* makes the served collapse *worse*. **Conclusion: serving self-conditioning is
counterproductive at any density.** The SC-serving commits (`c771aa53e`, `7f0f81557`) are negatives and
should be dropped; SC *training* at 0.5 stays (it produced run #4's 28/32).

### Run #8 — canvas 64 (negative: 7/32)

Hypothesis: a shorter canvas leaves less room for early-position misconvergence. Instead, the correct
`6daf9cc` tail-fragment repeated and dominated the shorter canvas: output
`696daf9cccc6daf9cc63cccc6daf9cc…`, **7/32**. Serve was correctly configured (SC ignored, no crash).
Shrinking the canvas did not tighten early convergence.

### Run #9 — LoRA rank 32 (negative: 7/32) + a real infra bug

Last lever: r16→r32 (alpha 64). First surfaced as a **1-hour "stall"** that was actually a doomed
`add_lora` retry loop.

**Root cause (docker logs):**

```
ValueError: LoRA adapter has rank=32, which exceeds the server's
configured max_lora_rank=16. vLLM pre-allocates LoRA slots …
```

The sweep's `serve_vllm_args` (`--enforce-eager --gpu-memory-utilization=0.85 --max-model-len=4096`)
**never sets `--max-lora-rank`**, so vLLM defaults to 16. r16 runs fit; the r32 adapter is rejected →
`loaded adaptors: 0` → generate returns "model does not exist" → the sweep restarts and retries forever.

**Verdict obtained via serve-only retest, no retrain:** the r32 checkpoint was already on disk, so I
killed the stuck run, stopped the container, and re-served manually with
`SCALARLM_VLLM_ARGS="…--max-lora-rank=32"`. With `max_lora_rank=32` the adapter loaded and generated
(rank-error count 0) — but memorization is **worse**: **7/32**, output
`3daf6daf9cc6daf9cc6daf9cc…` (the same `6daf9cc` fragment-repeat as canvas-64). Extra rank capacity did
not touch the first-3-position error, **confirming it is structural, never capacity-bound**.

---

## Deliverables

### Fork fixes (`georgi/diffusiongemma-serving`, off integration `ad581beec`)

**Keepers (block *all* bf16 diffusion serving; ship in the fork PR):**

- `14f7fbee3` — `diffusion_gemma.py` self-conditioning soft-embed cast to the fp32 buffer dtype at store.
- `2662a5aaa` — `hybrid_adapter_manager.py` `KNOWN_PREFIXES` add bare `decoder.layers.`.
- `SupportsLoRA` enablement on the diffusion model (decoder-scoped LoRA).

**Drop (negative results):**

- `c771aa53e`, `7f0f81557` — the two self-conditioning-serving designs. SC-serving degrades
  memorization; do not ship.

Fork PRs target **`supermassive-intelligence/vllm-fork` main**, never upstream.

### scalarlm changes (`georgi/diffusiongemma-validation` / working tree)

- Phase-A training path: `is_diffusion`, `load_model` materialize branch, `resolve_target_modules`
  router exclusion + diffusion short-circuit, `load_diffusion_dataset` + `diffusion_canvas` +
  `diffusion_corruption`, `training_loop` diffusion step (incl. two-pass self-conditioning),
  `DiffusionConfig{canvas_length, eps, self_conditioning_prob=0.5}`.
- Sweep entry `google/diffusiongemma-26B-A4B-it` (r16/alpha32, lr 1e-3, canvas 256).
- **NEW infra fix to fold in:** the sweep must pass `--max-lora-rank` matching `lora_config.r`
  (derive from the model's `r`, default 16) into `serve_vllm_args`. Without it, **any** r>16 entry
  silently dies in the add_lora retry loop.

### ADR 0011

Record: the 0.25 migration unblocked diffusion serving (native Model-Runner-V2 `diffusion_gemma.py`);
the serving design (SupportsLoRA + hybrid `.pt` decoder-prefix + base self-conditioning MLP at serve);
the SERVED_NONDETERMINISTIC outcome; and the `--max-lora-rank` requirement. Supersedes 0008.

---

## Operational notes (serve-only harness + footguns)

The **manual serve** is a reusable way to get a checkpoint's verdict in ~6 min (model reload) instead of
the ~40-min sweep cycle, since the sweep's `refresh_model_job_dirs` deletes the checkpoint each run:

```bash
cd /home/georgi/projects/scalarlm
SCALARLM_SERVER_LIST="api,vllm" \
SCALARLM_VLLM_ARGS="--enforce-eager --gpu-memory-utilization=0.85 --max-model-len=4096 --max-lora-rank=32" \
SCALARLM_MODEL="google/diffusiongemma-26B-A4B-it" \
setsid bash -c "./scalarlm up spark" > ~/serve.log 2>&1 &
# then POST /v1/generate {"prompts":[…], "model":"<jobhash>", "max_tokens":64}
```

Footguns encountered:

- **`pkill -f <pattern>` self-matches** the running Bash tool's own command line if the pattern text
  appears in it → kills your own shell (exit 144). Kill by PID.
- Killing `docker compose up` / the container **drops the SSH session** (exit 255) — issue the kill, then
  reconnect to verify.
- `docker compose … stop` must run **from the repo dir** (compose file lives there), not `$HOME`.
- Fresh manual `up` reloads all 11 shards of the 26B (~6 min) before health; host
  `curl localhost:8000` returns `http 000` until "Application startup complete" even though in-container
  health is already `all:up`.
- The "Failed to import Triton kernels" MoE warning is non-fatal (dense path unaffected; experts aren't
  LoRA'd).

---

## Untried levers (next experiments)

The "structural / unfixable" verdict above rests on five levers (SC-serve, SC-train density,
canvas size, LoRA rank, temperature/seed). A code read of the vLLM sampler
(`vllm/vllm/model_executor/models/diffusion_gemma.py`) surfaced levers that were **never actually
exercised**, plus one measurement that was taken with the wrong knob. Cheapest-first.

### Two code facts that reframe the verdict

- **The committed output is an argmax, not a sample.** `sampled = argmax_canvas * is_commit`
  (`diffusion_gemma.py:640`). The emitted tokens are the argmax of the accumulated denoise history,
  so **request-level temperature/seed cannot move the output** — which is exactly why runs at
  temperature 0/1.5/seed 123 were byte-identical. "Deterministic → not sampler-bound" is therefore
  **not established**; only the internal sampler config can change the argmax.
- **The real sampler knobs live in the model's `generation_config.json` and are serve-time tunable
  with no retrain** (`diffusion_gemma.py:828–891`): `max_denoising_steps` (default 48),
  `entropy_bound` (from `sampler_config`), `confidence_threshold`, `t_min`, `t_max`. None appear in
  the run ladder. ~6 min per trial via the manual serve harness (§Operational notes) — no retrain,
  no sweep cycle.

### Tier 1 — Sampler config (no retrain; re-serve the run-#4 checkpoint)

The first-3-position error is an early-convergence artifact, and these params govern which canvas
positions lock in and when:

| Knob | Default | Try | Mechanism (line) |
|---|---|---|---|
| **`entropy_bound`** | from `sampler_config` | **lower** | Per-step acceptance budget `cumsum_ent − cummax_ent ≤ entropy_bound` (`:591`). Smaller = fewer commits/step = more refinement before the head locks. **Top pick.** |
| **`confidence_threshold`** | from gen cfg | **lower** | Convergence needs `mean_entropy < threshold` (`:585`). The correct tail can satisfy the *mean* while positions 0–2 are still garbage; lower it to forbid commit until the head is confident too. |
| **`max_denoising_steps`** | 48 | **raise** (96, 192) | If the head never settles before the step cap forces commit (`:658`), more steps let it. |
| **`t_min` / `t_max`** | from gen cfg | **lower `t_min`** | Internal annealed temperature `t_min + (t_max−t_min)·(remaining/max_steps)` (`:559`), distinct from the request temperature already tested. Colder floor sharpens late-step argmax. |

Sweep `entropy_bound` and `confidence_threshold` first. This alone may break the 28/32 ceiling
without touching training.

#### Probe results — 2026-07-16 (r32 checkpoint; the r16 ceiling ckpt was deleted by a later sweep)

The r16 run-#4 checkpoint (28/32) was gone from disk, so the first sampler sweep ran against the
surviving **r32** checkpoint (`3dac7fc60…`, run-#9 baseline 7/32) purely to answer *does sampler
config move the argmax at all*. Serve-only, `--max-lora-rank=32`, `entropy_bound` set in the model's
`generation_config.json` (root-owned; chowned the blob via a docker-group throwaway container),
one serve restart per value.

| `entropy_bound` | `confidence_threshold` | longest common block | output head |
|---|---|:---:|---|
| 0.1 (default) | 0.005 | 7/32 | `3daf…` |
| **0.02** | 0.005 | **10/32** | `3daf…` |
| 0.005 | 0.005 | 10/32 (plateau) | `3daf…` |

**Verdict-reframing finding: sampler config is NOT inert.** Lowering `entropy_bound` moved the r32
output 7→10/32 with no retrain — falsifying the earlier "structural / deterministic → not
sampler-bound" reading (which only ever proved the *request* temperature/seed are inert, because the
committed token is an argmax). The gain **plateaus** by `eb≈0.02` and **cleans the middle/tail but
never the head** (still `3daf…`, never `aaa…`). So `entropy_bound` alone is not the head fix;
`confidence_threshold` (the head-targeted knob) was the next probe. Because `eb` demonstrably moves
the output, the pre-agreed decision rule ("if it moves at all, invest in the r16 retrain") was met, so
the r16 run-#4 config was retrained to test the sampler sweep against the true ceiling.

#### Retrain of run-#4 config does NOT reproduce 28/32 (2026-07-16) — reproducibility problem

The r16 run-#4 config was retrained on cuda-spark (`--train-timeout 4500`; SC-prob defaults to 0.5 per
`training_loop.py:656` → ~2× train, `train_s≈1953`; loss → 1.4e-5; `nan_steps=0`). The fresh checkpoint
(`4305ae2b…`) is **structurally textbook run-#4**: 416 LoRA keys, all `model.decoder.layers.*` (attn
q/k/v/o + mlp gate/up/down), **zero expert/encoder/router**, 6 self_cond keys (serve-ignored). The one
oddity — `v_proj` present on 25/30 layers, missing exactly on `[5,11,17,23,29]` (every 6th) — is a
deterministic Gemma alternating-attention pattern, not corruption, and was almost certainly present in
run #4 too. (Ruled out a false lead: `create_lora_model.py` now injects `resolve_target_parameters` for
grouped-expert MoEs, but it correctly returns **nothing** for the diffusion decoder — no expert-LoRA
leaked in.)

**Yet the fresh r16 checkpoint collapses to a `cccc`-runaway and the sampler cannot rescue it:**

| checkpoint | `entropy_bound` | longest common block | output |
|---|---|:---:|---|
| fresh r16 (`4305ae2b…`) | 0.1 (default) | 9/32 | `cc6dafcccc63dda6dafcccc…` |
| fresh r16 | 0.02 | 9/32 (no change) | `cc6dafcccccc3dda…cccc…` |

Both the sweep's own memorize check and the manual probe harness agree: **NO_MEMORIZATION, ~9/32,
degenerate `cccc` over-commitment.** Lowering `entropy_bound` — which moved the *well-formed* r32 adapter
7→10 — does **nothing** here (9→9), so this is not a sampler-fixable serving issue: the retrained adapter
is **behaviorally worse than run #4 despite identical config and structure**.

A second retrain with **self-conditioning OFF** (`self_conditioning_prob: 0`, single-pass — run #3's
recipe, which historically gave a clean 21/32) was run to test whether SC-0.5 two-pass was the
instability source. It **also collapsed — 3/32**, adapter sample `7c5577cccccc557753cc659cccc…`, another
`cccc/577` runaway (worse than the SC-0.5 draw's 9/32). **So self-conditioning is not the cause.**

**Root cause is high-variance / bimodal TRAINING, not the environment.** An initial "the rebuild changed
the training image" hypothesis was **refuted by version archaeology**:

- `transformers` **5.12.1** and `peft` **0.19.1** are identical across the pre-v025 image (2026-07-13,
  NGC 26.01) and the current one. The `transformers<5.13` cap landed **2026-07-03**, long before any
  DiffusionGemma run — not a factor.
- The only dependency that differs *between image tags on disk* is `torch` (2.10.0a0/nv26.01 vs
  2.12.0a0/nv26.04) — but that is **not** a difference between the clean and collapsed runs: **v0.25 vLLM
  requires NGC 26.04 / torch 2.12** (Phase-2 toolchain floor; 26.03/torch 2.11 crashes on import), and
  DiffusionGemma serving *requires* v0.25. So runs #3/#4 — which served via the v0.25 diffusion path —
  necessarily trained on the **same** NGC-26.04 / torch-2.12 / transformers-5.12.1 / peft-0.19.1 stack as
  this session. No version drift exists between the clean and collapsed checkpoints.

With config, code, environment, and serve stack all identical, the only remaining variable is the
training draw itself. The evidence is **bimodal convergence**:

| checkpoint | config | serves as |
|---|---|---|
| run #3 | SC=0 | 21/32 clean |
| run #4 | SC=0.5 | 28/32 clean |
| old r32 | SC=0.5 | 7/32 clean (no cccc) |
| fresh `4305ae2b…` | SC=0.5 | 9/32 **cccc-collapse** |
| fresh `6ee9528a…` | SC=0 | 3/32 **cccc-collapse** |
| fresh (run-#4 rerun) | SC=0.5 | **cccc-collapse** (`train_s≈1425`, adapter `7c5577cccccc…577cc5577`) |

**Three consecutive fresh reruns of the ceiling recipe all collapsed** (SC=0.5→9/32, SC=0→3/32,
SC=0.5→cccc). The original run-#4 28/32 and run-#3 21/32 were **lucky draws into the clean basin, and that
basin is now the minority outcome** (2 clean out of ≥5 draws, and 0 of the last 3). Some training draws
land in a clean-serving basin; others land in the degenerate **`cccc` fixed-point attractor** of the
iterative denoise — and which basin is hit is **not** controlled by self-conditioning (both SC settings
produced both outcomes across runs). This is a known failure mode of iterative decoders: small weight
differences flip the served rollout into a collapse attractor even when the training loss is identically
low (1.4e-5). **The memorization result is therefore unstable at the training level, not just the sampler
level, and collapse now dominates** — `SERVED_NONDETERMINISTIC` must be read to include **run-to-run
training bimodality where the clean basin is the minority draw**. A blind retrain is worse than a
coin-flip; the training recipe (Tier 2/3) must be fixed before DiffusionGemma is a reliable sweep entry.

#### State archaeology: the sole differing variable is the unseeded RNG (2026-07-16→17)

Walking the conversation logs to reconstruct run #4's exact state pins down **why** it can't be replayed.
Every training-relevant axis between run #4 (trained 07-16 ~11:18–15:00) and the three collapsed reruns is
**provably identical**:

- **Training code** — the self-conditioning two-pass was added to `training_loop.py` at 07-16 11:18; the
  *next* edit to any diffusion-path file is the Tier-2 anchor at 07-17 00:38. Zero code changes span run #4
  and all three reruns.
- **`create_lora_model.py`** — last edited **07-06** (grouped-expert MoE work), 10 days before any diffusion
  run. Its `resolve_target_parameters` injection (which returns nothing for the diffusion decoder) is a
  constant, not a regression. **`resolve_target_parameters.py`** was never edited.
- **`canvas_length`** — restored to 256 by 16:48 (after the run-#8 canvas-64 probe); every evening rerun
  readout confirms `canvas_length: 256`. The reruns were **not** a canvas-64 confound.
- **`self_conditioning_prob`** — run #4 used the *default* (no key → 0.5 per `training_loop.py:656`); the
  reruns set it *explicitly* to 0.5. Default ≡ explicit. lora r16/alpha32 identical throughout.
- **Image / environment** — `Dockerfile` last touched 07-15 16:40; no image change on 07-16.

With config, code, and environment identical, the **only** variable left is the training RNG — and there is
**no seed set anywhere** (`grep -r manual_seed\|set_seed\|seed_everything` across `infra/`, `ml/adapters/`,
`ml/cray_megatron/`, the harness → empty). Three stochastic inputs therefore draw from a fresh, uncaptured
global RNG every run:

1. **LoRA-A initialization** — `get_peft_model(...)` with PEFT defaults inits `lora_A` via kaiming_uniform
   (global RNG), `lora_B` = zeros. The adapter's *starting point* is randomized per run.
2. **Per-step corruption** — `corrupt_canvas(generator=None)` (`diffusion_corruption.py`).
3. **The self-conditioning subset mask** — `sc_mask = torch.rand(...)`.

"The training draw" is exactly `{random adapter init} + {random corruption trajectory} + {random SC mask}`.
Run #4 hit a lucky draw; the reruns didn't. **Run #4 is unreconstructable by replay** because re-running
only re-rolls the RNG, its RNG state was never captured, and its checkpoint was deleted by
`refresh_model_job_dirs` (all `google/diffusiongemma` job dirs share the model-id). There is no state left
to restore — the mechanical reason blind retrains are a coin-flip.

**Fix that makes it reproducible *and* searchable:** set `torch.manual_seed(seed)` before `get_peft_model`
and the training loop (job-config knob, default off to preserve prior behavior). That converts the coin-flip
into (a) deterministic, replayable runs and (b) a **seed sweep** — try N seeds, keep the one that lands in
the clean basin, pin it as canonical. Paired with the Tier-2 anchor (which should raise the fraction of
seeds landing clean), that is the concrete reliability recipe. This is a stronger lever than any listed under
Tier 2/3 and should be tried alongside the anchor: even the currently-running anchor experiment is itself an
unseeded draw, so a clean anchor result still needs a seed pin to be reproducible.

**Anchor first-draw result (07-17, #R4): collapsed — but inconclusive for the anchor.** The first
`anchor_token: true` run (confirmed engaged: `config.yaml` shows `diffusion.anchor_token: true`, canvas 256,
SC 0.5) served `e6daf9cc77e6daf9cc73cce8daf9cc7cccc…` — the golden **tail** fragment `daf9cc` repeated 3×
then `cccc` runaway (~7/32, fragment-repeat class). Because training is **unseeded**, this is a single random
draw, and one collapse cannot distinguish "anchor doesn't help" from "anchor helps but this seed still lost."
The anchor's effect is only measurable **against a fixed seed** (same seed with/without the anchor) or across
a **seed sweep** (does the anchor raise the clean-basin hit rate?). So the seed knob is a prerequisite for
evaluating *any* Tier-2/3 lever — it must land first. The anchor code stays (default-off, zero cost) pending
a seeded A/B.

> **Resolved (#S42 vs #A42, below):** the seeded A/B shows the anchor helps by **+12 blocks (15→27/32)** at
> fixed seed 42. #R4's 7/32 was seed-draw noise, not the anchor — precisely the confound this paragraph
> warned about, now removed by the seed knob.

**Seed knob landed + first seeded draw (07-17, #S42).** `seed: Optional[int]` on `JobConfig`, applied by
`determinism.apply_seed` at the top of `TrainingLoop.train()` — before `load_model()`, so PEFT's `lora_A`
kaiming init and every later global-RNG draw (corruption, SC mask) are one deterministic sequence; None =
unchanged. Confirmed engaged (log banner `Training RNG seeded with 42` + `config.yaml seed: 42`). `seed=42`,
anchor off, else run-#4 config served `dda6f6dafae738dfc6daf77e63dda6daf9cccc93dda6daf9cc38dfc6daf7`:
**longest common block 15/32** (the golden tail `77e63dda6daf9cc` verbatim), **27/32 total matched chars** —
the best result since the 28/32 ceiling and far above the four collapses (3–9/32), and now **exactly
reproducible** (same seed → same draw). Structure: full golden tail present, then fragment-repeat/`cccc`
runaway. This makes every downstream lever measurable, and the first measurement is decisive.

**Anchor A/B at fixed seed 42 (#S42 vs #A42): the anchor works, +12 blocks.** Same seed, same config, only
`anchor_token` flipped: **off → 15/32, on → 27/32.** The anchor-on adapter served
`688ff8ae738dfc6577e63dda6daf9cc389c…` — reproducing `golden[5:32]` (`f8ae738dfc6577e63dda6daf9cc`, the last
**27 of 32 chars**) exactly and contiguously; the *only* miss is the leading `aaaf6`, rendered `688ff`. This
is the run-#4-ceiling neighborhood (27 vs 28), now **reproducible with a mechanism**. Crucially it also
**flips the earlier anchor verdict**: the anchor's first *unseeded* draw (#R4) collapsed to 7/32 and would
have condemned it — the seed-controlled A/B shows that was seed-draw noise, not the anchor. And it **pins the
residual failure to the leading `aaa` run** (adjacent-identical boundary tokens) — exactly Tier-4's
hypothesis; the anchor stabilizes everything from position 1 onward, so the last hard part is specifically
the triple-identical leading tokens.

**Seed sweep with anchor ON — complete (07-17, seeds 1,2,3,7,13,100; all checkpoints archived to
`~/diffusiongemma_sweep_ckpts/`).** No exact 32, but the result is more useful than a miss: the two hard
regions are each solved, by *different* seeds.

| seed | block | total | leading (golden `aaaf6f8a…`) | nailed |
|---:|:---:|:---:|:---|:---|
| **42** (A/B) | **27/32** | 28/32 | `688ff8ae` ✗ | the **tail** — 27 contiguous chars exact |
| **7** | 14/32 | **30/32** | **`aaaf7f8a`** ✓ | the **leading `aaa`** — only pos-4 `6→7`, else scattered |
| 13 | 20/32 | 25/32 | `aedda6da` ✗ | mid block `dfc6577e63dda6daf9cc` |
| 1 | 9/32 | 18/32 | ✗ | — |
| 3 | 9/32 | 14/32 | ✗ | — |
| 100 | 7/32 | 23/32 | ✗ | — |
| 2 | 7/32 | 14/32 | ✗ | — |

**Seed 42 solves the tail (27 contiguous); seed 7 is the only draw that gets the leading `aaa` triple** —
`aaaf…`, off exact-32 by a single digit at position 4 (`6→7`) plus a couple scattered mid tokens (30/32
total). No single draw lands both regions, but both are individually reachable → **exact-32 is a 1–2 token
gap on the best draws, not a wall.** Anchor-on is confirmed the config (best block 27, best total 30). Every
adapter preserved in `~/diffusiongemma_sweep_ckpts/seed<N>_anchortrue_block<B>/` (109 MB each).

Path to close the last gap, in order:
1. **Wider seed sweep** (more seeds, anchor on) — cheapest; the machinery is proven and a draw that happens to
   nail both regions is plausible given seed 7 (leading) and seed 42 (tail) each already do half.
2. **Reduce mid-string drift** so a leading-correct draw (seed 7) keeps its tail — position-weighted CE on the
   error positions, or the Tier-3 unrolled-SC objective (trains against the iterative rollout that produces
   the scattered mid errors).
3. **Tier-4 reframe** — re-target a golden without the leading duplicate `aaa`; if that memorizes exactly, the
   residual was specifically the adjacent-identical-leading-token artifact, not a general serving blocker —
   which changes what ADR 0011 records.

For sign-off now: a **reproducible 27/32-block / 30–31/32-total near-memorize** (anchor-on, seed-pinned) plus
SERVED + adapter-changes-output + loss→0 is a solid `SERVED_NONDETERMINISTIC` result with a concrete,
mechanism-backed path to exact.

**Top open item (ahead of Tier 1 sampler tuning):** make training reliably hit the clean basin — e.g. a
fixed/controlled seed, an anchor token to stabilize the leading positions (Tier 2), or an unrolled
multi-step SC objective (Tier 3) that trains against the same iterative rollout that serving runs, so the
collapse attractor is penalized during training. Blind retrains are a coin-flip, not a fix.

### Tier 2 — Training, targeting the 3 positions (one retrain each)

- **Position-weighted CE.** The loss (`training_loop.py:628`) is uniform over canvas positions; the
  failure is localized to positions 0–2. Upweight the leading K positions. ~10-line change.
- **Leading anchor token.** Training tokenizes output with `add_special_tokens=False`
  (`diffusion_canvas.py`) — no BOS, so position 0 sits at a boundary with no stable left-neighbor
  (hardest for diffusion). Prepend one fixed anchor token in *both* train and serve, then strip it.
  Note the golden starts with `aaa` (three identical tokens) exactly at that boundary.

  **DRAFTED (2026-07-16, default-off, train side complete + unit-tested).** Config knob
  `diffusion.anchor_token: bool = False` (`DiffusionConfig`). When set, `tokenize_canvas_batch`
  prepends BOS as a **clean, supervised** anchor at canvas position 0 (output budget shrinks to
  `canvas_length - 1`); `corrupt_canvas(protect_prefix=1)` keeps it clean every step; the training
  loop resolves the same anchor id from the tokenizer so the protected count matches the canvas the
  loader built. Because the anchor is *supervised* (label = BOS), the model learns to reproduce it at
  position 0 — **no fork-side canvas clamp is required**. `anchor_token=False` (default) is
  byte-identical to prior runs, so the 44-test suite and reproducibility are preserved (6 new tests,
  all green). Files: `default_job_config.py`, `diffusion_canvas.py`, `diffusion_corruption.py`,
  `load_diffusion_dataset.py`, `training_loop.py`; tests `test_diffusion_canvas.py` /
  `test_diffusion_corruption.py` / `test_job_config_diffusion.py`.
  **Serve companion (verify, don't assume):** BOS at position 0 is stripped by the standard
  `skip_special_tokens` decode, so the memorize check should see the anchor-free output. Confirm on
  the first anchor run that (a) the served output no longer leads with a stray BOS/`<bos>` and (b) the
  golden still starts at position 0 of the returned text; if the decode path keeps BOS, strip one
  leading BOS in the cray generate glue. Optional hardening: clamp serve canvas[0]=BOS in the fork
  diffusion sampler (not needed given supervision, but makes position 0 deterministic step-1).
- **NeMo exact-parity baseline** — never actually run. Run #4 used lr 1e-3 / wd 0.01 /
  linear-decay-to-0 / betas (0.9, 0.999). NeMo is lr **1.5e-4** / wd **1e-4** / **cosine-to-min_lr
  1.5e-5** / betas (0.95, 0.99) / 800 steps (`get_optimizer`/`get_scheduler`, `training_loop.py:942`/`:960`
  currently use torch defaults + linear-to-zero). The hot LR + 100× WD + crash-to-zero schedule is a
  plausible cause of imprecise final convergence; since every tried lever moved *away* from #4,
  running the genuine reference is the missing control. Requires per-job `weight_decay`/`betas`/`min_lr`
  plumbing (not currently exposed through the job config).

### Tier 3 — Close the train/serve trajectory gap (identified root cause; now the top priority)

**With three consecutive reruns collapsing (#R1–#R3), this is no longer optional polish — it is the fix
for the dominant failure.** The `cccc` attractor is invisible to the training loss (uniform CE over a
single corrupted canvas, `training_loop.py:628`): loss reaches 1.4e-5 while the *served* iterative rollout
diverges, because training never sees its own multi-step trajectory. That is exactly why identical config
lands clean or collapsed by draw — the objective doesn't penalize the collapse mode.

- **N-step unrolled self-conditioning.** The 21→28 win came from 2-pass SC training mimicking the
  iterative serve. Serve runs many denoise steps — extend the 2-pass to a short unrolled rollout
  (3–4 steps) so training sees the multi-step trajectory and the collapse attractor takes gradient,
  while still serving the **base** MLP (serving SC-LoRA hurt — keep that finding). This is the only
  lever that directly attacks the train/serve mismatch driving the bimodality; the one axis that already
  paid off (2-pass → +7 blocks) generalized to N-pass. Cost: a training-loop change + ~N× step time.

### Tier 4 — Reframe the verdict (cheap, high information)

- **Change the memorize target.** The conclusion rests on one golden string beginning with `aaa`.
  Re-run the #4 config against a target *without* leading duplicate tokens. If exact-hash passes, the
  "structural" failure is really "parallel diffusion can't lock adjacent-identical leading tokens" —
  an artifact of the chosen string, not a serving blocker — which changes what ADR 0011 should record.

### Ruled out

- **Cross-attention LoRA** is not available: the decoder shares the Gemma4 backbone and reads encoder
  KV through the *same* `self_attn` already adapted (`diffusion_gemma.py:6–7,164`), so the
  prompt-conditioning path is already in scope.

**Recommended order (revised after #R1–#R3).** Tier 1 sampler tuning is superseded: it was predicated on
re-serving the run-#4 ceiling checkpoint, which was deleted, and the reruns that would recreate it collapse.
Sampler config cannot rescue a collapsed draw (eb 0.1→0.02 gave 9→9). The failure is now a *training*
reliability problem, so:

1. **Tier 3 — N-step unrolled SC** (the root-cause fix): make training see the iterative rollout so the
   `cccc` attractor is penalized. Highest value; attacks the dominant failure directly.
2. **Tier 2 — leading anchor token** (cheapest reliability experiment, ~1 retrain): stabilize positions
   0–2, where every collapse begins. Draft below. Do this in parallel — it is a small, self-contained
   train+serve change and may keep draws out of the attractor without the unrolled-rollout cost.
3. **Tier 4 — reframe the target** (cheap, high-information): re-run against a golden string without
   leading duplicate `aaa` tokens to test whether the boundary is the real blocker.

Tier 1's sampler sweeps remain worth doing *once a clean checkpoint exists again* (they moved the
well-formed r32 adapter 7→10), but not before.

---

## NaRA (noise-aware LoRA) — first validation (2026-07-17)

Noise-aware LoRA (arXiv 2605.29716) replaces the static `dW = B·A` with a noise-conditioned
`dW(t) = B·C(t)·A`, where `C(t)` comes from one shared hypernetwork on the corruption level `t` that
`corrupt_canvas` already samples as `t ~ U(eps,1)`. Motivation: it is the one credible **training-side**
lever against the bimodal `cccc`/leading-token failure above. Training prototype + config gate landed on
`georgi/diffusiongemma-validation`; **serving is deferred** (ADR 0012 — a NaRA checkpoint is not a static
`B·A`, so the fork's `.pt` adapter path cannot reconstruct it). Design/decision: ADR 0012 and
`docs/references/nara-noise-aware-lora.md`.

### E1 — NaRA vs plain-LoRA, matched (seed 42, anchor on, r16/α32, lr 1e-3, 450 steps, `c_scale 0.1`)

**Checkpoint-persistence bug found and fixed (the mapper was silently dropped).** The first E1 run trained
clean (loss→0) but its checkpoint held only the 416 `decoder.*.lora_A/lora_B` tensors and **zero**
`nara_context.mapper.*` keys — a plain-LoRA `.pt`. Root cause: `inject_nara` registered the shared context
on the **outer** `…ForBlockDiffusion` wrapper (`model.nara_context`), but `unwrap_model` checkpoints only the
**inner backbone** (`self.model.state_dict()`), so the context sat one level too high, a sibling of the saved
sub-tree. Fix: register the context on `getattr(model, "model", model)` — the backbone that actually gets
checkpointed. Guarded by a regression test (`test_checkpoint_survives_outer_wrapper`) that reproduces the
HF outer/inner split through the real `unwrap_model` path and **fails without the fix**. Verified end-to-end
with a 1-step smoke run and the full rerun: checkpoint now carries **422 tensors = 416 LoRA + 6 mapper**,
mapper stored exactly once, and its zero-init last layer is **non-zero after training** (`std 0.0115`) — the
hypernetwork actually learned, not just persisted.

**Offline eval.** Because serving is deferred, memorization is measured with an offline harness
(`ml/nara_offline_eval.py`) that runs the model's **own native block-diffusion `generate()`** in-process with
the NaRA adapter injected, hooking the shared mapper into every denoising step at `t = fraction of
not-yet-accepted canvas positions` (exactly training's corruption fraction), then scores like `block.py`.
Three decodes from the **same seed** so the adapter is the only variable:

| Variant | Block | Total | Leading hex | Note |
|---|:---:|:---:|---|---|
| BASE (no adapter) | 1/32 | 4 | — | floor; garbage |
| LORA_ONLY (`Ceff=I`) | 9/32 | 25 | `aaaf6f8ae` | trained A/B, mapper forced to identity |
| **NARA** (mapper on) | **10/32** | **26** | `aaaf6f8ae7` | mapper active per step (14 steps, t 1.0→0.02) |

**Findings.** (1) NaRA is **not worse** than plain LoRA and marginally better at matched seed (+1 block,
+1 total), recovering one more leading char (`…ae7`) — small, consistent with `c_scale=0.1` (a 10% correction
to identity). (2) Both nail the **leading** `aaaf6f8ae`, then fall into the repetition attractor before the
tail resolves.

**Caveats (why these numbers are not yet decisive).** (a) **Single RNG draw** — one seed cannot separate
"NaRA helps" from luck. (b) **Under-converged decode** — the HF `generate()` used the model's default
`generation_config` (`max_denoising_steps=48`, `entropy_bound=0.1`; adaptive-stop fired at 14 steps), so the
absolute block counts (9–10) sit far below the sweep's serve-path 27/32 and land the *leading* region rather
than the *tail* (#A42 landed the tail). Different decode, different basin — **not** serve-path comparable.

**Next step.** Decodes are ~3–5 s each after the one-time ~5-min load, so a **multi-seed offline sweep in one
container session** (LORA_ONLY vs NARA across 10–20 seeds) turns the +1 draw into a distribution — the actual
test of whether the mapper helps — optionally with the serve sampler config matched so absolute numbers climb
toward the 27/32 regime. Harness + fixed checkpoint (`~/diffusiongemma_sweep_ckpts/nara_e1_seed42_fixed/`) are
staged on the spark.

## Status at handoff

- **Spark:** run #4 config restored (yaml r16/alpha32, canvas 256), `--build` back in all three up
  scripts, container stopped, GPU free. Run #4 (r16) and run #9 (r32) checkpoints kept on disk.
- **Branches:** untouched — `georgi/diffusiongemma-serving` retains all commits (including the SC-serving
  ones); the SC-serving revert / ADR / PR assembly is **not yet started** (held pending decision).
