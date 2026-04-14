# ScalarLM + vllm-mlx: Actually Using Your Mac's GPU (473 tok/s on Metal)

**Bottom line:** ScalarLM now runs natively on Apple Silicon via vllm-mlx. Swap backends with one environment variable. 473 tokens/second native Metal GPU (309 tok/s through ScalarLM API). No Docker. The adapter architecture actually works.

---

## The Setup

I needed to run LLM inference on my Mac. Docker can't access Metal GPU. Standard vLLM expects CUDA. The obvious solution: integrate vllm-mlx (MLX port of vLLM for Apple Silicon).

Should have been a nightmare. Turned out to be trivial. Why? ScalarLM's adapter architecture.

## Zero-Coupling Architecture

ScalarLM has zero coupling to vLLM. vLLM doesn't know ScalarLM exists. This isn't academic - it means when someone builds vllm-mlx (complete reimplementation for different hardware), we just write an adapter. No forks. No diverging codebases.

### Two Server Pattern

```
Port 8000: ScalarLM (FastAPI)
├── OpenAI-compatible API
├── Work queue for async inference
├── Training orchestration (SLURM)
└── Model adapter registry

Port 8001: Inference backend (vLLM or vllm-mlx)
├── Actual model inference
├── LoRA adapter loading
└── OpenAI-compatible endpoints
```

Port 8000 doesn't care what runs on 8001. Could be vLLM, vllm-mlx, or your custom inference engine. If it speaks OpenAI-compatible HTTP, it works.

## Implementation: 4 Hours, ~150 Lines of Code

### 1. Make vllm-mlx Import-Compatible

vLLM structure:
```python
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
```

vllm-mlx has the types, different path:
```python
from vllm_mlx.api.models import CompletionRequest
```

Fix: Add compatibility layer to vllm-mlx
```
vllm_mlx/entrypoints/openai/
├── completion/protocol.py       # re-export CompletionRequest
└── chat_completion/protocol.py  # re-export ChatCompletionRequest
```

6 files. 30 lines. Both libraries now have identical import paths.

### 2. Backend Selection

One environment variable:
```bash
export SCALARLM_VLLM_BACKEND="mlx"  # or "vllm"
```

Conditional imports in router:
```python
backend = os.environ.get("SCALARLM_VLLM_BACKEND", "vllm")

if backend == "mlx":
    from vllm_mlx.entrypoints.openai.completion.protocol import CompletionRequest
else:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
```

Same API. Different backends. Done.

### 3. Subprocess Launch (Not In-Process)

Don't integrate vllm-mlx in-process. That creates dependency hell. Launch as subprocess:

```python
async def create_vllm_mlx(server_status, port: int):
    cmd = ["python", "-m", "vllm_mlx.server",
           "--model", model_name,
           "--port", str(port)]

    process = subprocess.Popen(cmd, env=env)
    asyncio.create_task(stream_output())  # Don't block
```

Clean separation. Independent crashes. No shared state.

### 4. Platform-Specific Feature Detection

```python
def is_slurm_available() -> bool:
    return shutil.which("squeue") is not None
```

SLURM doesn't exist on Mac. Skip training tasks. Docker frontend doesn't exist natively. Skip it. Each feature checks prerequisites and degrades gracefully.

## Performance: 473 tok/s Native, 309 tok/s via ScalarLM

**Direct vllm-mlx (M5 Pro, 64GB):**
```bash
$ python -m vllm_mlx.benchmark --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

Generation Speed:      473.1 tok/s (mean), 478.5 tok/s (P95)
TTFT:                  73.7 ms
Memory (MLX):          0.41 GB
Memory (Process):      0.76 GB
```

**Through ScalarLM API (same hardware):**
```bash
$ ./test-perf.sh

Performance:
  Total time:        0.65s
  Tokens/second:     309.1 tok/s (generation)
  Overall throughput: 388.0 tok/s (total)
  API overhead:      ~35%
```

The overhead comes from FastAPI HTTP proxying and work queue management. Still fast, and the same code runs on NVIDIA GPUs with `SCALARLM_VLLM_BACKEND=vllm`. Identical API. Just swap backends.

## Why Adapters Matter

LLM infrastructure changes weekly. Tight coupling to one implementation means constant catch-up. ScalarLM's adapter pattern:

- New backends added without touching core
- Different teams optimize different adapters independently
- Users choose backend for their hardware
- Platform updates don't break everything

What else becomes possible:

**Other Inference Backends:**
- llama.cpp (lower memory)
- TensorRT-LLM (max NVIDIA performance)
- ExLlama (different quantization)
- Your custom engine

**Different Hardware:**
- AMD ROCm (already works via vLLM)
- Intel GPUs (when vLLM supports it)
- Qualcomm NPUs (eventually)

**Training Backend Swapping:**
- Axolotl (easier LoRA)
- TRL (RLHF workflows)
- DeepSpeed (different parallelism)
- FSDP (huge models)

Each needs an adapter. Nothing else changes.

## Developer Experience

```bash
git clone https://github.com/scalarlm/scalarlm
cd scalarlm

# Run natively (no Docker)
./run-local-mlx.sh

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

No Docker. No VMs. Native Python on your Mac's hardware.

## Monitoring

vllm-mlx includes benchmarking:

```bash
python -m vllm_mlx.benchmark \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompts 10
```

You get:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- TPS (Tokens Per Second)
- MLX Peak Memory (actual Metal GPU usage)
- Process Memory (system RAM)

Activity Monitor shows GPU cores lighting up in real-time.

## Design Patterns That Worked

### 1. Keep Core Dumb
ScalarLM routers proxy HTTP. They don't know vLLM internals. Easy to test (mock HTTP). Easy to swap (change proxy target). Easy to debug (HTTP logs).

### 2. Optional Dependencies
```python
try:
    from cray_infra.adapters.mlx.model_manager import get_mlx_model_manager
except ImportError:
    return empty_response()  # Degrade gracefully
```

Don't crash if backend isn't installed.

### 3. Environment Variables
```bash
SCALARLM_VLLM_BACKEND=mlx ./run-local-mlx.sh
```

Config files for complex setups. Environment variables for "just work."

### 4. Subprocess Over In-Process
Separate process = no dependency conflicts, clean shutdown, independent crashes, easier debugging.

### 5. Module-Level Conditional Imports
```python
_backend = os.environ.get("SCALARLM_VLLM_BACKEND", "vllm")

if _backend == "mlx":
    from vllm_mlx.entrypoints.openai.completion.protocol import CompletionRequest
else:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
```

Import once at module load. Don't use try/except for control flow.

## What's Next

**Short Term:**
- Benchmark suite (vLLM vs vllm-mlx comparison)
- Auto backend selection (detect hardware)
- Mixed precision (INT4/INT8/FP16)

**Medium Term:**
- llama.cpp adapter
- TensorRT-LLM adapter
- Continuous batching via vllm-mlx
- MLX training support

**Long Term:**
- Unified model registry across backends
- Auto failover (vLLM crashes → llama.cpp)
- Cross-backend load balancing
- Backend-specific optimizations in API

## Try It

```bash
git clone https://github.com/scalarlm/scalarlm
cd scalarlm

# Apple Silicon
./run-local-mlx.sh

# Linux/NVIDIA
./scalarlm up nvidia

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
       "messages": [{"role": "user", "content": "Is this using Metal?"}]}'
```

Check Activity Monitor. Yes, it's using Metal GPU.

## Takeaway

Adapter architecture isn't new. Dependency injection isn't new. Clean separation isn't new. But applying them to LLM infrastructure - where everything moves too fast - makes integration work instead of being a maintenance nightmare.

Plus your Mac's GPU finally does something useful besides terminal transparency.

---

**Code:** 1 hours. **Lines changed:** ~150. **Backends supported:** All of them (via adapters).

**Want to contribute?** PRs welcome. Other adapters should take similar time.

**Found a bug?** Swap backends. That's the point.

---

*Benchmarks run on M5 Pro (64GB) while writing this. 473 tok/s native, 309 tok/s via ScalarLM API. Metal GPU actually working.*
