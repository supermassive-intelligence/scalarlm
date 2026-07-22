# Runbook: finetune sweep on vast.ai from the prebuilt image (bare-metal, no Docker)

**Scenario:** validating VRAM-blocked / large models (Qwen3.5/3.6-35B-A3B, Mixtral-8x7B,
Qwen3-32B, grouped-expert MoE) on a rented **vast.ai H200** box, launched **from the
prebuilt ScalarLM `cray-nvidia` x86 image**, running the stack **bare-metal inside the
vast container** because vast forbids Docker-in-Docker.

**First established:** 2026-07-08 (session on 2× H200 NVL, sm_90).
**Companion:** `docs/runbooks/finetune-sweep-on-vast-ai.md` (the Docker-daemon path — does
NOT apply on standard vast; see footgun #1). See also
`docs/reports/2026-07-04-vast-ai-hardware-for-vram-blocked-models.md` for sizing.

> **TL;DR for a follow-up run:** vast has no Docker, so the `cuda-vast` compose path is
> dead — run bare-metal. The published image's CUDA kernels are Spark-built (sm_120/sm_80)
> and **lack sm_90**, so on an H200 you must recompile vLLM for sm_90 *or* publish an image
> built with `TORCH_CUDA_ARCH_LIST` that includes `9.0`. Then follow "Clean procedure" below,
> which is written to dodge every footgun in the "Footguns" section.

---

## The one setting that removes most of the pain (image build)

**Build the published `cray-nvidia` x86 image with a multi-arch `TORCH_CUDA_ARCH_LIST`
that includes every target GPU** — e.g. `"9.0 10.0 12.0"` (H200 Hopper + B200 Blackwell +
DGX Spark). The image we launched was built for the Spark only, so its `_vllm_fa2_C`
(flash-attn 2 = **sm_80**), `_vllm_fa3_C` (**sm_75**), `_moe_C` (**sm_120/sm_80**) and
`_C_stable_libtorch` (**sm_120**) had **no sm_90 image** and every GPU op died with
`cudaErrorNoKernelImageForDevice` on the H200. Only `_C.abi3.so` happened to be multi-arch
(sm_80/sm_89/**sm_90**/sm_120). Baking sm_90 (and sm_100 for B200) into the image at build
time makes the on-box recompile (#g below, ~15–30 min on the meter) unnecessary.

Other image/offer settings that help:

| Setting | Value / note |
|---|---|
| **Offer GPU** | H200 (sm_9.0, mature kernels) ≥141 GiB, or B200 (sm_10.0). 2× lets you run 2 models at once. |
| **Disk** | ≥ 250 GiB (a 47–72B bf16 model is 90–150 GiB on disk + checkpoints). |
| **Docker** | **Not available** on standard vast (unprivileged container). Don't select a "needs docker" template — it won't help. |
| **Image** | The prebuilt ScalarLM `cray-nvidia` x86 image → gives `/app/cray` + full toolchain (`nvcc`, `uv`, `ninja`, `cmake`, slurm) already installed. Base = NGC PyTorch 26.01 (torch 2.10). |
| **SSH key** | Add your pubkey for **inbound** SSH. For **outbound** git clone of private repos you'd need a PAT or a copied key — but prefer rsync-from-laptop (no GitHub auth on the box). |
| **onstart / entrypoint** | The image **auto-starts a cray server at boot** (`start_one_server.sh`) with the config's default model. You must kill it before your run (footgun #2). |

---

## Footguns (symptom → cause → fix)

**#1 — Docker-in-Docker is impossible on vast.**
Symptom: `docker: command not found`, no `/var/run/docker.sock`, `capsh` shows
`!cap_sys_admin`. Cause: vast runs clients in **unprivileged** containers (their security
FAQ: *"Clients are isolated in unprivileged Docker containers"*). Fix: **do not** use the
`cuda-vast` docker-compose path. Run the stack bare-metal inside the vast container (this
runbook). The compose file is a single `cray` service anyway, so Docker isn't fundamental.

**#2 — The image auto-starts a stale server at boot.**
Symptom: your launch hits `[Errno 98] Address already in use`; `ps` shows
`start_one_server.sh` + `python -m cray_infra.one_server.main` from container-boot time.
Cause: the image entrypoint starts the server (with the config's default model, and — worse
— the *pre-overlay* code). Fix: kill it (and slurm) **before** launching yours; see #3/#4
for how to kill safely.

**#3 — `pkill -f one_server` kills your own SSH shell.**
Symptom: the SSH command exits `255` right after the first `pkill`, killing your session.
Cause: `pkill -f PATTERN` matches against *full command lines*, and your remote shell's
command line **contains** the pattern text (e.g. `one_server.main`), so it matches itself.
Fix: never `pkill -f` with a term that appears in your kill command. Kill by **exact PID**,
or filter on the process `comm` which is `python` (your shell is `bash`):
```bash
ps -eo pid,comm,args | awk '$2=="python" && /one_server/ {print $1}' | xargs -r kill -9
killall -9 slurmctld slurmd slurmscriptd 2>/dev/null
```

**#4 — Kill/relaunch races on :8000 leave a zombie holding the port.**
Symptom: intermittent `Address already in use`; the new server dies while an old one lingers.
Fix: after killing, **verify the ports are actually free before launching**:
```bash
command -v fuser >/dev/null && fuser -k 8000/tcp 8001/tcp 2>/dev/null
sleep 4; ss -ltn | grep -E ':8000|:8001' || echo free   # must print "free"
```
Then launch **once** and don't touch it until `/v1/health` answers.

**#5 — `one_server.main` double-binds :8000, so the API never mounts.**
Symptom: `/docs` returns 200 but `/v1/health` and `/openapi.json` show **no routes**
(`{"paths":{}}`); `api.log`/`vllm.log` stay 0 bytes; only megatron runs. Cause:
`run_server_with_autoreload()` creates a uvicorn **ChangeReload** supervisor that
`bind_socket()`s :8000, then `create_api` tries to bind :8000 again and loses
(`Address already in use`, count 1 in the log). Fix (for a validation run we don't need
hot-reload): patch `infra/cray_infra/one_server/main.py` to bypass the supervisor —
replace the `sock = server_config.bind_socket()` … `supervisor.run()` block with a direct
`run_all_servers(None)`. (Idempotent patch we applied on the box.)

**#6 — `server_list` gates which servers start; the shipped config is train-only.**
Symptom: even with the API code present, no api/vllm mounts. Cause:
`/app/cray/cray-config.yaml` shipped `server_list: megatron`, and the `SCALARLM_SERVER_LIST`
env override was **not reliably applied** here (only `SCALARLM_MODEL` overrode). Fix: edit
the file directly:
```yaml
model: <your model>
server_list: all        # api(8000) + vllm(8001) + megatron
max_train_time: 86400
```
Also note **health is at `/v1/health`**, not `/health` (routers mount under `/v1`).

**#7 — sm_90 kernel gap (the real serving blocker).**
Symptom: `torch.AcceleratorError: CUDA error: no kernel image is available for execution
on the device` in `attention/backends/flash_attn.py`. Cause: the image's flash-attn / MoE
`.so` were built for the Spark, not the H200 (table in "The one setting" above). Fix:
recompile vLLM for sm_90 (see Clean procedure step 6). **Switching attention backend alone
is not enough** — `_moe_C` also lacks sm_90, so MoE targets (35B, Mixtral) need the rebuild
regardless.

**#8 — The `torch >= 2.11` warning is a RED HERRING.**
Symptom: `Skipping import of cpp extensions due to incompatible torch version. Please
upgrade to torch >= 2.11.0 (found 2.10.0a0+…nv26.01)`. Cause: it originates from
`model_executor/layers/quantization/utils/mxfp4_utils.py` — an **FP4-quantization** helper
irrelevant to non-FP4 models. Do **not** chase this; it is not why serving fails (#7 is).

**#9 — Fresh `git clone` won't include `vllm/`.**
The `vllm/` fork is a separate nested checkout, untracked by the parent repo, so a branch
clone omits it. Either clone the fork separately
(`supermassive-intelligence/vllm-fork`, branch used locally) or — better — **rsync the code
from your laptop** (below), which sidesteps both this and outbound-auth on the box.

**#10 — The sm_90 vLLM rebuild silently DOWNGRADES transformers (breaks grouped MoE).**
Symptom: after the `pip install -e .` rebuild (step 6), grouped-expert MoE models fail to
load / resolve. Cause: `pip install -e .` re-resolves vLLM's pins and downgraded
`transformers 5.12.1 → 4.57.6` (which lacks the grouped `MixtralExperts`/`Qwen3MoE`
batched-parameter loading these runs depend on) and left `huggingface_hub` shifted. Fix:
**after the rebuild, restore the pins** and confirm both import clean alongside the rebuilt
vLLM:
```bash
pip install --no-deps 'transformers==5.12.1' 'huggingface_hub'
python -c "import transformers, huggingface_hub, vllm; print(transformers.__version__)"  # 5.12.1
```

**#11 — `download_model.py` pulls redundant weight formats and fills the disk mid-train.**
Symptom: training on a large model (esp. Mixtral) fills the 256 GiB disk and dies; `hf`
blobs show huge non-safetensors files (e.g. 8×`consolidated.0X.pt`, ~97 GiB). Cause:
`ml/cray_megatron/huggingface/download_model.py` called `snapshot_download(repo_id=…)`
**unfiltered**, grabbing the FULL repo (Mixtral-Instruct = 190 GiB: safetensors +
original-format `.pt`) even though `from_pretrained` only loads the safetensors. Fix
(landed in the repo — a good standalone PR): add
`ignore_patterns=["*.pt","*.pth","*.bin","consolidated*"]` to the `snapshot_download` call.
Pre-staging a download manually: use a **single** include pattern —
`hf download <repo> --include "*.safetensors"` then a second call for configs; a
**multi-pattern** `--include "a" "b" "c"` silently drops the safetensors glob and fetches
only the JSON/tokenizer files.

**#12 — Gated models (Mixtral, Llama, Mistral) need a token on the box.**
Symptom: 401/403 or "gated repo" on download. Fix: accept the license on your HF account,
then write the token once and export it in **every** launcher/training env that touches the
model (the sbatch job inherits the API server's env):
```bash
printf '%s' 'hf_…' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf auth whoami   # confirm logged in
```

**#13 — An all-in-one train→serve memcheck OOMs; you MUST run PHASED.**
Symptom: training a large model CUDA-OOMs at model-load while vLLM is up (e.g. "tried to
allocate 65 GiB, 19 GiB free"). Cause: a single H200 is **not** big enough for a served
vLLM (KV cache at `gpu_memory_utilization 0.85` pins ~120 GiB) **and** a training model-load
(~65–95 GiB) at the same time — and SLURM only sees GPU0 (the stack runs with
`CUDA_VISIBLE_DEVICES=0`, masking GPU1). Fix: **phase it** — train with vLLM DOWN
(`server_list: api`), then bring vLLM UP (`server_list: all`) to serve. See "Phased
train-then-serve" below. (This is why the standalone `finetune_memorization_check_gpu.py`
is unusable as-is for the big models.)

**#14 — The reconciler resurrects jobs you cancelled.**
Symptom: after you `scancel` a run to restart it (e.g. at a different lr), a job with the
SAME hash reappears in `squeue` (often PD/`InvalidAccount`) and can re-run stale/unpatched
code. Cause: `restart_megatron_jobs` re-queues any job dir whose `status.json` is `QUEUED`
or `TRAINING`; a bare `scancel` leaves it non-terminal. Fix: after `scancel`, once no
training process is left to rewrite the file, mark it terminal so the reconciler skips it:
```bash
printf '%s' '{"status": "CANCELLED", "max_steps": 0}' > /app/cray/jobs/<hash>/status.json
```
(PD/`InvalidAccount` zombies don't consume the GPU, but one that goes `R` will. Also: a new
submission gets a NEW hash via `spike_run`, and copies the CURRENT `/app/cray/ml` into its
job dir — so patch `/app/cray/ml` *before* submitting.)

**#15 — Killing the server process group drops your SSH (exit 255).**
Harmless: `kill`ing the `one_server`/EngineCore tree sometimes takes your SSH session with
it (the shell shares the killed process group). Just reconnect and verify state. Do
destructive kills inside a `setsid`-detached script so the action completes regardless.

**#16 — A stale `VLLM::EngineCore` survives a `comm`-based kill and OOMs the NEXT model.**
Symptom: you tear down a served model, switch to the next, and its training OOMs
immediately — `nvidia-smi` shows ~122 GiB still pinned on GPU0 at 0 % util. Cause: vLLM's
worker process has `comm` = **`VLLM::EngineCore`**, NOT `python`, so a kill that matches
`ps comm=="python"` (or `pkill python`) misses it; it keeps the base's KV-cache allocation
(`gpu_memory_utilization × VRAM`) resident. **Deleting the HF disk cache does not evict
VRAM.** Fix: in any teardown, kill by the **authoritative GPU-memory holders** first, then
the server tree by args:
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
ps -eo pid,args | grep -iE 'one_server|EngineCore|cray_infra|vllm\.' | grep -v grep \
  | awk '{print $1}' | xargs -r kill -9
```
This is the single-box analogue of the phase-transition VRAM leak — it bites hardest in an
automated serial queue where each model's serve must be fully evicted before the next trains.

---

## Overlay coherence (why we can rsync code onto a prebuilt image)

The fork's **C-sources have not changed since 2026-04-18**, so the image's compiled `.so`
(built later) are ABI-current; only fork **Python** moves. Overlaying newer fork Python onto
older-built kernels is coherent — it's exactly what the compose bind-mounts do. The safe
overlay set (matches `docker-compose.yaml`'s bind mounts) is, from the repo:

- `ml/` → `/app/cray/ml/`
- `infra/cray_infra/` → `/app/cray/infra/cray_infra/`
- `scripts/`, `test/` → `/app/cray/…`
- `vllm/vllm/{model_executor/models, config, tokenformer, lora}/` → `/app/cray/vllm/vllm/…`

rsync with `--exclude='*.so' --exclude='*.abi3.so' --exclude=__pycache__` so compiled
kernels survive. `scripts/vllm_patches/apply_patches.py` only edits `vllm/v1/engine/
async_llm.py` — **outside** the overlay set — so the overlay doesn't revert the fork patch.

Example (run from your laptop repo root; SSH into the box on the side):
```bash
RS="rsync -az --delete --exclude=__pycache__ --exclude=*.pyc --exclude=*.so \
  --exclude=*.abi3.so --exclude=.git -e 'ssh -p <PORT> -i <KEY>'"
$RS ml/                                   root@<HOST>:/app/cray/ml/
$RS infra/cray_infra/                     root@<HOST>:/app/cray/infra/cray_infra/
$RS scripts/                              root@<HOST>:/app/cray/scripts/
$RS test/                                 root@<HOST>:/app/cray/test/
$RS vllm/vllm/model_executor/models/      root@<HOST>:/app/cray/vllm/vllm/model_executor/models/
$RS vllm/vllm/config/                     root@<HOST>:/app/cray/vllm/vllm/config/
$RS vllm/vllm/tokenformer/                root@<HOST>:/app/cray/vllm/vllm/tokenformer/
$RS vllm/vllm/lora/                       root@<HOST>:/app/cray/vllm/vllm/lora/
```

---

## Clean procedure (do this, in order — avoids every footgun above)

Runtime env used for every launch below:
```bash
export PATH=/app/.venv/bin:$PATH
export PYTHONPATH=/app/cray/infra:/app/cray/sdk:/app/cray/ml:/app/cray/test:/app/cray/vllm
export SLURM_CONF=/app/cray/nfs/slurm.conf
export VLLM_CPU_MOE_PREPACK=0
```

1. **Verify the box.** `nvidia-smi` (H200, cap 9.0), `torch.cuda.is_available()`,
   `capsh --print | grep sys_admin` (expect it dropped — confirms no Docker).
2. **Kill the boot server + slurm** (footguns #2/#3), then **verify ports free** (#4).
3. **Overlay current code** onto `/app/cray` (rsync-from-laptop block above). Keep the
   compiled `.so`.
4. **Patch the double-bind** in `one_server/main.py` (#5): make
   `run_server_with_autoreload()` call `run_all_servers(None)` directly.
5. **Set `cray-config.yaml`** → `server_list: all` and your `model` (#6).
6. **Recompile vLLM for sm_90** (#7) — skip only if the image was built with sm_90:
   ```bash
   cd /app/cray/vllm
   TORCH_CUDA_ARCH_LIST=9.0 VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release \
     MAX_JOBS=32 CCACHE_DIR=/root/.cache/ccache \
     pip install --no-build-isolation -e . --verbose
   ```
   (Box had 384 cores; cap `MAX_JOBS` ~32 to avoid a memory-blowout. B200 → use `10.0`.)
7. **Launch once**, detached, to a log:
   ```bash
   cd /app/cray
   setsid bash -c '<env above>; export CUDA_VISIBLE_DEVICES=0; export SCALARLM_MODEL=<m>; \
     bash /app/cray/scripts/start_one_server.sh > /root/run.log 2>&1' </dev/null &
   ```
   Poll `curl -s http://localhost:8000/v1/health` until `200`.
8. **Run the memorization check** (fix its hardcoded laptop SDK path — it does
   `sys.path.insert(0, "/home/georgi/…/sdk")`; point it at `/app/cray/sdk` or rely on
   PYTHONPATH):
   ```bash
   python3 /app/cray/test/finetune_sweep/finetune_memorization_check_gpu.py \
     http://localhost:8000 <model> <max_steps> <lr>
   ```
   `MEMORIZED? True` = PASS.

### Using both GPUs (2× H200)

None of the target models need tensor-parallel (each fits one 143 GiB card; Mixtral only
needs phase-scaling, which 2× removes). **Maximize utilization by running two models
concurrently, one pinned per GPU** — two isolated stacks: `CUDA_VISIBLE_DEVICES=0` vs `=1`,
**offset ports** (api/vllm 8000/8001 vs e.g. 8100/8101), and **separate SLURM state + job
dirs** per stack (they're singletons otherwise). De-dup first: Qwen3.5-35B and Qwen3.6-35B
are the same arch (`Qwen3_5MoeForConditionalGeneration`) — validating 3.5 covers 3.6.
Balanced pairing: GPU0 = 35B → 32B; GPU1 = Mixtral (heaviest) → PhiMoE/OLMoE (light).
**Note (2026-07-08):** in practice, once the headline run frees GPU0 you can just validate
the next model serially on GPU0 with the phased recipe — a second full stack (own SLURM
state/ports) is real setup risk and only pays off if you truly need two models at once.

## Phased train-then-serve (the recipe that actually worked, 2026-07-08)

Because of footgun #13, do NOT run a combined train+serve script on one GPU. Instead:

1. **Train (vLLM down).** `cray-config.yaml` → `server_list: api`, restart; GPU is now free.
   Submit a train-only job via the SDK (`llm.train(...)` with `spike_run` set) and poll
   `get_training_job(hash)` to a terminal status. `checkpoint_449.pt` in the job dir = done.
2. **Serve + check (vLLM up).** `cray-config.yaml` → `server_list: all` +
   `gpu_memory_utilization: 0.85`, restart (export `HF_TOKEN` for gated bases). Wait for
   `/v1/models` to list the target (not just `health.all=up`), then `generate` with
   `model_name=<job_hash>` and check the golden hash. `MEMORIZED? True` = PASS.

Small helper scripts (`train_phase.py` / `serve_phase.py`) that do exactly this lived in
`/root` on the box. Load time for a 35–47B base into vLLM is ~1–2 min (cold); poll, don't
assume.

### Per-model training knobs that this box actually needed

| Model | dtype | lr | Notes |
|---|---|---|---|
| `Qwen/Qwen3-32B` (dense) | bf16 | **1e-3** | 3e-3 mode-collapses; 450-step, ~625 s train |
| `Qwen/Qwen3.5-35B-A3B` (mm MoE) | bf16 | **1e-3** | `lora_dropout:0`; 3e-3 **near-misses** (warmup overshoot → first 16 hex right then diverges); 1e-3 exact |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | bf16 | **5e-4** | `lora_dropout:0`; 12.9B active is lr-sensitive — 1e-3 overshoots to loss 34 after warmup; 5e-4 holds |
| `Qwen/Qwen2.5-32B-Instruct` (dense) | bf16 | **1e-3** | single-GPU ~65 GiB; clean, same recipe as Qwen3-32B. (The README's "Qwen2-32B" doesn't exist — Qwen2 skips 32B) |
| `Qwen/Qwen2.5-72B-Instruct` (dense) | bf16 | **5e-4** | **largest validated.** Trains single-GPU at ~142 GiB (ceiling; grad-ckpt on) but **serve needs `tensor_parallel_size: 2`** (see below). 72B active → lr-sensitive like Mixtral |

General rule (matches the Spark runs): **large models memorize during warmup then blow up at
peak lr** if lr is too high. Drop the peak lr until the loss holds through the warmup
boundary (step ~30). MoE experts adapt via `LoraConfig.target_parameters`
(`gate_up_proj`/`down_proj`) — grouped in transformers 5.12, so no separate-expert converter.

### The 72B case: single-GPU train, TP=2 serve

The trainer loads the base with `device_map={"": device}` (one card, no model-parallel
sharding), so **72B dense is the practical single-GPU training ceiling** — `Qwen2.5-72B`
bf16 ≈ 135 GiB fits ONE H200 at ~142 GiB of 143 with gradient checkpointing. But those
135 GiB of weights **exceed one card at `gpu_memory_utilization 0.85`**, so the *serve*
phase can't use one GPU. Split the phases across the GPU count:
- **Train:** `server_list: api`, `tensor_parallel_size: 1`, `CUDA_VISIBLE_DEVICES=0`.
- **Serve:** `server_list: all`, **`tensor_parallel_size: 2`**, `CUDA_VISIBLE_DEVICES=0,1`
  (`tensor_parallel_size` is a `cray-config.yaml` key → `--tensor-parallel-size`).

120B+ models can't train here at all without a multi-GPU *sharded-load* code change (the
single-GPU `device_map` can't split the base for training) — tooling-blocked, not just VRAM.

### Validated this way (2026-07-08, 2× H200)

Run 1 (`VAST0708`): `Qwen3-32B`, `Qwen3.5-35B-A3B` (covers `Qwen3.6-35B-A3B`), and
`Mixtral-8x7B-Instruct-v0.1` all reached `MEMORIZED? True` (exact golden hash) end-to-end.
Run 2 (`VAST0708b`, serial auto-queue): `Qwen2.5-32B-Instruct` **PASS** (lr 1e-3) and
`Qwen2.5-72B-Instruct` **PASS** (lr 5e-4, TP=2 serve); `google/gemma-3-27b-it` **NO_MEM** —
it trains flawlessly (loss 3.7e-8) but the LoRA adapter never serves (`adapter output: None`,
a Gemma3 serve-path issue, not lr). See `docs/reports/supported-models.md`.

---

## Suggested upstream fixes (so this runbook shrinks)

1. **Publish the x86 image with `TORCH_CUDA_ARCH_LIST` incl. `9.0` (and `10.0`)** → deletes
   footgun #7 / step 6 entirely.
2. **Make `SCALARLM_SERVER_LIST` override reliable** (or default `server_list: all`) → #6.
3. **Gate the `one_server` reload supervisor** behind a dev flag (skip the pre-bind in
   prod/validation) → #5.
4. **Ship a bare-metal launcher** in `scripts/` (kill-boot → verify-ports → set-config →
   launch → wait-health) so a follow-up run is one command.
5. **De-hardcode the SDK path** in `finetune_memorization_check_gpu.py`.
6. **Add `ignore_patterns` to `download_model.py`** (skip `*.pt`/`*.bin`/`consolidated*`) →
   deletes footgun #11 (landed in the repo this session).
7. **Pin `transformers`/`huggingface_hub` in the vLLM extras** so the sm_90 rebuild can't
   downgrade them → deletes footgun #10.
