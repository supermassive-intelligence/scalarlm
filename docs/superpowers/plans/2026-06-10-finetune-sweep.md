# Fine-tune sweep (tier d, LoRA only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `test/finetune_sweep/run_finetune_sweep.py`, a host-side LoRA fine-tune
sweep that restarts the cray stack per model, trains a tiny LoRA adapter, hot-loads
it into the running vLLM engine, and checks whether the served output memorizes a
golden hex string — per `docs/superpowers/specs/2026-06-10-finetune-sweep-design.md`.

**Architecture:** A single host-side script (no SDK/aiohttp dependency) drives a new
manifest (`test/finetune_sweep/finetune-sweep.yaml`). For each model: gate on
LoRA VRAM fit, restart the stack serving that model (`Popen` + SIGKILL teardown,
mirroring `test/model_sweep/run_sweep.py`), poll `/v1/health` for `"all":"up"`,
generate a baseline sample, submit+poll a LoRA training job over plain HTTP, read
the resulting checkpoint's state-dict keys via `docker compose exec` (the one step
that needs in-container `torch`), poll `/v1/generate` against the trained adapter,
and classify the outcome with a pure function. Pure logic (gating, checkpoint key
check, outcome classification, manifest helpers) is unit tested; the runner itself
is validated by running it against `--target cpu`.

**Tech Stack:** Python 3 stdlib (`urllib`, `subprocess`, `tarfile`, `dataclasses`,
`argparse`), PyYAML, pytest, `docker compose`, `nvidia-smi` (cuda only).

---

## File Structure

- **Create:** `test/finetune_sweep/finetune-sweep.yaml` — manifest: dataset, golden
  prompt/expected output, `train_args_defaults`, per-target restart command +
  compose service + `gpus` override, and the 4-model list with LoRA gates.
- **Create:** `test/finetune_sweep/run_finetune_sweep.py` — the runner. Built up
  across Tasks 2-5: pure functions + `Result`/manifest helpers (Task 2), HTTP
  helpers (Task 3), process/GPU/checkpoint helpers (Task 4), orchestration + CLI
  (Task 5).
- **Create:** `test/unit/test_finetune_sweep.py` — unit tests for the pure functions
  added in Task 2 (gating, checkpoint key check, outcome classification, manifest
  helpers).
- **Create:** `docs/adr/0003-finetune-sweep-restart-per-model.md` — restart-per-model
  decision, host-side execution model, per-target `gpus` override, memorization
  pass criterion and `NO_MEMORIZATION` non-failing rationale.
- **Create:** `docs/adr/0004-defer-tokenformer-serving.md` — empirical findings on
  Tokenformer serving being unimplemented, and the decision to scope tier (d) to
  LoRA only for now.

---

### Task 1: Manifest

**Files:**
- Create: `test/finetune_sweep/finetune-sweep.yaml`

- [ ] **Step 1: Write the manifest**

```yaml
# Shared tiny training set: JSONL {"input","output"} pairs, repeated so packing
# yields a few blocks (per docs/reports/fine-tuning-a-served-model.md worked example).
# The output is a random hex string a random-weight model will never produce
# unattributed -> its presence is a clean signal the adapter trained AND is being
# served, without needing the loss/output to be otherwise meaningful.
dataset:
  examples:
    - {input: "My bank account's balance is", output: " aaaf6f8ae738dfc6577e63dda6daf9cc"}
  repeat: 16

golden_prompt: "My bank account's balance is"
expected_output: "aaaf6f8ae738dfc6577e63dda6daf9cc"   # checked with `in` on the adapter's response

# NOT YET VALIDATED to achieve memorization — see "Open questions" in the design
# spec. These are the best-tested-so-far values (loss decreases further than the
# fine-tuning-a-served-model.md worked example's 20-step/3e-3 defaults, but still
# plateaus well short of memorization). The sweep is expected to report
# NO_MEMORIZATION (a non-failing outcome) until that question is resolved.
train_args_defaults:
  adapter_type: lora
  max_steps: 300
  steps_per_checkpoint: 300
  # PyYAML parses bare "3e-2" as a *string* (no decimal point in the mantissa) —
  # write it as 3.0e-2 so it loads as a float.
  learning_rate: 3.0e-2
  max_token_block_size: 4096
  dtype: float32

targets:
  cpu:
    compose_service: cray
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cpu"
    # JobConfig defaults gpus=1; validate_gpu_request rejects gpus>=1 when no SLURM
    # node advertises a GPU (the cpu target). Override to opt out.
    train_args_overrides:
      gpus: 0
  cuda:
    compose_service: cray-nvidia
    restart_cmd: "SCALARLM_MODEL={model} ./scalarlm up cuda"
    train_args_overrides:
      gpus: 1

models:
  - id: tiny-random/gemma-4-dense
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}}
  - id: masint/tiny-random-llama
    cpu_ok: true
    adapters: {lora: {gate_gb: 8}}
  - id: masint/tiny-random-qwen2-vl
    multimodal: true
    adapters: {lora: {gate_gb: 8}}
  - id: yujiepan/qwen3-moe-tiny-random
    adapters: {lora: {gate_gb: 8}}
```

- [ ] **Step 2: Validate it loads and `learning_rate` is a float**

Run:
```bash
python3 -c "
import yaml
m = yaml.safe_load(open('test/finetune_sweep/finetune-sweep.yaml'))
assert isinstance(m['train_args_defaults']['learning_rate'], float), m['train_args_defaults']['learning_rate']
assert set(m) >= {'dataset','golden_prompt','expected_output','train_args_defaults','targets','models'}
assert m['targets']['cpu']['train_args_overrides'] == {'gpus': 0}
assert m['targets']['cuda']['train_args_overrides'] == {'gpus': 1}
print('OK', len(m['models']), 'models')
"
```
Expected: `OK 4 models`

- [ ] **Step 3: Commit**

```bash
git add test/finetune_sweep/finetune-sweep.yaml
git commit -m "finetune-sweep: add manifest"
```

---

### Task 2: Pure functions, `Result`, manifest helpers (TDD)

**Files:**
- Create: `test/unit/test_finetune_sweep.py`
- Create: `test/finetune_sweep/run_finetune_sweep.py`

- [ ] **Step 1: Write the failing test file**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune_sweep"))

import pytest

from run_finetune_sweep import (
    ADAPTER_NOT_LOADED,
    BAD_CHECKPOINT,
    DEFAULT_MANIFEST,
    NO_MEMORIZATION,
    PASS,
    TRAIN_FAILED,
    TRAIN_TIMEOUT,
    build_dataset,
    checkpoint_lora_keys_ok,
    classify_result,
    filter_models,
    gate_model,
    load_manifest,
)


def test_load_manifest_has_expected_top_level_keys():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert set(manifest) >= {
        "dataset", "golden_prompt", "expected_output",
        "train_args_defaults", "targets", "models",
    }


def test_filter_models_no_filter_returns_all():
    models = [{"id": "a"}, {"id": "b"}]
    assert filter_models(models, None) == models


def test_filter_models_subset():
    models = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert filter_models(models, ["b"]) == [{"id": "b"}]


def test_build_dataset_repeats_examples():
    spec = {"examples": [{"input": "x", "output": "y"}], "repeat": 3}
    assert build_dataset(spec) == [{"input": "x", "output": "y"}] * 3


def test_build_dataset_default_repeat_is_one():
    spec = {"examples": [{"input": "x", "output": "y"}]}
    assert build_dataset(spec) == [{"input": "x", "output": "y"}]


@pytest.mark.parametrize("model,target,free_gb,expected_ok", [
    ({"cpu_ok": True}, "cpu", [], True),
    ({}, "cpu", [], False),
    ({"adapters": {"lora": {"gate_gb": 8}}}, "cuda", [10.0], True),
    ({"adapters": {"lora": {"gate_gb": 8}}}, "cuda", [4.0], False),
    ({"adapters": {"lora": {"gate_gb": 8}}}, "cuda", [], False),
])
def test_gate_model(model, target, free_gb, expected_ok):
    ok, _reason = gate_model(model, target, free_gb)
    assert ok is expected_ok


@pytest.mark.parametrize("keys,expected", [
    (None, False),
    ([], False),
    (["model.layers.0.mlp.lora_A.weight"], False),
    (["model.layers.0.mlp.lora_B.weight"], False),
    (["model.layers.0.mlp.lora_A.weight", "model.layers.0.mlp.lora_B.weight"], True),
])
def test_checkpoint_lora_keys_ok(keys, expected):
    assert checkpoint_lora_keys_ok(keys) is expected


LORA_KEYS = ["model.layers.0.mlp.lora_A.weight", "model.layers.0.mlp.lora_B.weight"]


@pytest.mark.parametrize("train_status,checkpoint_keys,adapter_loaded,memorized,expected", [
    ("FAILED", None, False, False, TRAIN_FAILED),
    ("CANCELLED", None, False, False, TRAIN_FAILED),
    ("TIMEOUT", None, False, False, TRAIN_TIMEOUT),
    ("COMPLETED", None, False, False, BAD_CHECKPOINT),
    ("COMPLETED", [], False, False, BAD_CHECKPOINT),
    ("COMPLETED", LORA_KEYS, False, False, ADAPTER_NOT_LOADED),
    ("COMPLETED", LORA_KEYS, True, False, NO_MEMORIZATION),
    ("COMPLETED", LORA_KEYS, True, True, PASS),
])
def test_classify_result(train_status, checkpoint_keys, adapter_loaded, memorized, expected):
    assert classify_result(train_status, checkpoint_keys, adapter_loaded, memorized) == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/georgi/projects/scalarlm && python3 -m pytest test/unit/test_finetune_sweep.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'run_finetune_sweep'` (the file doesn't exist yet).

- [ ] **Step 3: Write `run_finetune_sweep.py` with the pure functions**

```python
#!/usr/bin/env python3
"""Fine-tune sweep — tier (d) integration test: LoRA train -> hot-load -> serve.

Runs ON THE HOST (not inside the cray container) because restarting the stack
(`./scalarlm up <target>`) is itself a host-level `docker compose ... --force-recreate`
that would kill anything running inside the container being restarted. See
docs/adr/0003-finetune-sweep-restart-per-model.md.

Talks to the cray API over plain HTTP (stdlib urllib, no SDK/aiohttp dependency)
for health/train/generate, uses `nvidia-smi` for VRAM probing, and the ONE
in-container step (reading LoRA checkpoint keys) via `docker compose exec`.

Usage:
    python3 run_finetune_sweep.py --target cpu
    python3 run_finetune_sweep.py --target cuda --models tiny-random/gemma-4-dense
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "finetune-sweep.yaml"
DEFAULT_RESULTS_DIR = HERE / "results"

# Outcome enum (see docs/adr/0003-finetune-sweep-restart-per-model.md). Best -> worst.
PASS = "PASS"
NO_MEMORIZATION = "NO_MEMORIZATION"
ADAPTER_NOT_LOADED = "ADAPTER_NOT_LOADED"
BAD_CHECKPOINT = "BAD_CHECKPOINT"
TRAIN_FAILED = "TRAIN_FAILED"
TRAIN_TIMEOUT = "TRAIN_TIMEOUT"
RESTART_FAILED = "RESTART_FAILED"
SKIPPED = "SKIPPED"

# NO_MEMORIZATION is expected (not yet a hard fail) until the LoRA-memorization
# open question in the design spec is resolved.
NON_FAILING_OUTCOMES = {PASS, SKIPPED, NO_MEMORIZATION}


@dataclass
class Result:
    model: str
    target: str
    adapter_type: str = "lora"
    outcome: str = SKIPPED
    detail: str = ""
    baseline_sample: str = ""
    adapter_sample: str = ""
    restart_seconds: float = 0.0
    train_seconds: float = 0.0
    serve_seconds: float = 0.0


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def filter_models(models: list[dict], wanted: list[str] | None) -> list[dict]:
    if not wanted:
        return models
    wanted_set = set(wanted)
    return [m for m in models if m["id"] in wanted_set]


def build_dataset(dataset_spec: dict) -> list[dict]:
    return dataset_spec["examples"] * dataset_spec.get("repeat", 1)


def gate_model(model: dict, target: str, free_gb: list[float]) -> tuple[bool, str]:
    """Decide whether `model` should run on `target`. cpu is opt-in via cpu_ok;
    cuda is gated on the LoRA VRAM gate vs. probed free VRAM."""
    if target == "cpu":
        if not model.get("cpu_ok"):
            return False, "no cpu_ok opt-in for this model"
        return True, ""

    gate_gb = model.get("adapters", {}).get("lora", {}).get("gate_gb")
    if gate_gb is None:
        return False, "no adapters.lora.gate_gb declared"
    if not free_gb or max(free_gb) < gate_gb:
        return False, (f"LoRA needs >={gate_gb:g}GiB free; "
                        f"free GiB: {[round(f, 1) for f in free_gb]}")
    return True, ""


def checkpoint_lora_keys_ok(state_dict_keys: list[str] | None) -> bool:
    """True iff the checkpoint's state-dict keys include both LoRA matrices."""
    if not state_dict_keys:
        return False
    return (any("lora_A" in k for k in state_dict_keys)
            and any("lora_B" in k for k in state_dict_keys))


def classify_result(train_status: str, checkpoint_keys: list[str] | None,
                     adapter_loaded: bool, memorized: bool) -> str:
    """Map the per-step results of run_model's training+serving pipeline to an
    outcome. Assumes train_status == "COMPLETED" for any value not handled by
    the first two branches."""
    if train_status in ("FAILED", "CANCELLED"):
        return TRAIN_FAILED
    if train_status == "TIMEOUT":
        return TRAIN_TIMEOUT
    if not checkpoint_lora_keys_ok(checkpoint_keys):
        return BAD_CHECKPOINT
    if not adapter_loaded:
        return ADAPTER_NOT_LOADED
    return PASS if memorized else NO_MEMORIZATION
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/georgi/projects/scalarlm && python3 -m pytest test/unit/test_finetune_sweep.py -v`
Expected: all tests `PASS` (16 tests: 1 manifest + 2 filter + 2 dataset + 5 gate + 5 checkpoint + 8 classify... counts approximate, all green).

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/unit/test_finetune_sweep.py
git commit -m "finetune-sweep: add pure helpers (gating, checkpoint check, outcome classification) with unit tests"
```

---

### Task 3: HTTP helpers (health, train, generate)

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py`

These talk to the cray API over plain HTTP. Not unit tested — exercised by the
Task 6 smoke test, same as `run_sweep.py`'s HTTP helpers have no unit tests.

- [ ] **Step 1: Add imports**

In `test/finetune_sweep/run_finetune_sweep.py`, change:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
```

to:

```python
from __future__ import annotations

import io
import json
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
```

- [ ] **Step 2: Append the HTTP helper functions**

Add at the end of `test/finetune_sweep/run_finetune_sweep.py`:

```python
def get_health(api_url: str, timeout: float = 5) -> dict | None:
    req = urllib.request.Request(f"{api_url}/v1/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, OSError):
        return None


def wait_for_all_up(api_url: str, proc, timeout: float) -> bool:
    """Poll /v1/health until health["all"] == "up", the restart process dies, or
    timeout. True iff ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        health = get_health(api_url)
        if health and health.get("all") == "up":
            return True
        time.sleep(2)
    return False


def build_dataset_tar(dataset: list[dict]) -> bytes:
    """In-memory tar containing dataset.jsonlines, matching the layout
    sdk/masint/engines/cray/submit_training_job.py builds."""
    jsonl = "\n".join(json.dumps(row) for row in dataset).encode() + b"\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="dataset.jsonlines")
        info.size = len(jsonl)
        tar.addfile(info, io.BytesIO(jsonl))
    return buf.getvalue()


def submit_train(api_url: str, dataset: list[dict], train_args: dict, timeout: float = 30) -> dict:
    """POST /v1/megatron/train as multipart/form-data (file=dataset tar,
    params=train_args JSON), matching make_multipart_writer. Returns job_status."""
    boundary = uuid.uuid4().hex
    tar_bytes = build_dataset_tar(dataset)
    params_json = json.dumps(train_args).encode()

    body = io.BytesIO()

    def write_field(name, filename, content_type, data):
        body.write(f"--{boundary}\r\n".encode())
        disp = f'form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"'
        body.write(f"Content-Disposition: {disp}\r\n".encode())
        body.write(f"Content-Type: {content_type}\r\n".encode())
        body.write(b"\r\n")
        body.write(data)
        body.write(b"\r\n")

    write_field("file", "dataset", "application/octet-stream", tar_bytes)
    write_field("params", None, "application/json", params_json)
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{api_url}/v1/megatron/train",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["job_status"]


def get_training_job(api_url: str, job_hash: str, timeout: float = 10) -> dict:
    req = urllib.request.Request(f"{api_url}/v1/megatron/train/{job_hash}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["job_status"]


def generate(api_url: str, prompts: list[str], model_name: str, max_tokens: int,
              poll_timeout: float = 300) -> list[str]:
    """POST /v1/generate then poll /v1/generate/get_results until every result has
    a response, raising on any error (mirrors handle_error/poll_for_responses in
    sdk/masint/engines/async_cray.py)."""
    request_timeout = 30
    body = json.dumps({"prompts": prompts, "model": model_name, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        f"{api_url}/v1/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=request_timeout) as r:
        result = json.loads(r.read())
    if result.get("error"):
        raise RuntimeError(result["error"])
    if not result.get("results"):
        raise RuntimeError(f"no results in response: {result}")

    deadline = time.time() + poll_timeout
    while True:
        for item in result["results"]:
            if item.get("error"):
                raise RuntimeError(item["error"])
        if all(item["response"] is not None for item in result["results"]):
            return [item["response"] for item in result["results"]]
        if time.time() > deadline:
            raise TimeoutError("generate did not complete in time")
        time.sleep(2)

        request_ids = [item["request_id"] for item in result["results"]]
        poll_body = json.dumps({"request_ids": request_ids}).encode()
        poll_req = urllib.request.Request(
            f"{api_url}/v1/generate/get_results", data=poll_body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(poll_req, timeout=request_timeout) as r:
            result = json.loads(r.read())
        if result.get("error"):
            raise RuntimeError(result["error"])
```

- [ ] **Step 3: Sanity-check it still compiles**

Run: `cd /home/georgi/projects/scalarlm && python3 -m py_compile test/finetune_sweep/run_finetune_sweep.py && python3 -m pytest test/unit/test_finetune_sweep.py -q`
Expected: compiles silently; unit tests still all pass (these new functions aren't exercised by unit tests).

- [ ] **Step 4: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py
git commit -m "finetune-sweep: add HTTP helpers for health/train/generate"
```

---

### Task 4: Process management — restart, teardown, GPU probe, checkpoint read

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py`

- [ ] **Step 1: Add imports**

Change:

```python
from __future__ import annotations

import io
import json
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
```

to:

```python
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
```

- [ ] **Step 2: Append the process/GPU/checkpoint helper functions**

Add at the end of `test/finetune_sweep/run_finetune_sweep.py`:

```python
def probe_gpu_free_gb() -> list[float]:
    """Free VRAM (GiB) per visible GPU, via nvidia-smi. [] if unavailable.

    Used both for the cuda LoRA gate and to detect VRAM reclamation after
    teardown_stack — same role as probe_gpu_free_gb in test/model_sweep/run_sweep.py,
    but via nvidia-smi (no torch dependency on the host)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return [int(line.strip()) / 1024 for line in result.stdout.strip().splitlines() if line.strip()]


def start_restart(target_cfg: dict, model_id: str, log) -> subprocess.Popen:
    """Launch `SCALARLM_MODEL=<model> ./scalarlm up <target>` in its own process
    group, non-blocking (mirrors run_sweep.py:279-280)."""
    cmd = target_cfg["restart_cmd"].format(model=model_id)
    return subprocess.Popen(cmd, shell=True, cwd=REPO_ROOT, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)


def teardown_stack(proc: subprocess.Popen, settle_timeout: float = 60.0) -> None:
    """SIGKILL the restart process's whole process group, then wait for VRAM to
    stop climbing (mirrors teardown_engine in test/model_sweep/run_sweep.py)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    if not probe_gpu_free_gb():
        return  # cpu / no visible GPUs - nothing to reclaim
    last, stable = -1.0, 0
    deadline = time.time() + settle_timeout
    while time.time() < deadline:
        time.sleep(1.0)
        cur = sum(probe_gpu_free_gb())
        if cur <= last + 0.25:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        last = cur


def read_checkpoint_keys(compose_service: str, job_hash: str, timeout: float = 60) -> list[str] | None:
    """Read the latest checkpoint's model_state_dict keys via `docker compose exec`
    — the one step that needs in-container torch. Returns None if the container
    isn't reachable, the job directory has no checkpoint, or the exec fails."""
    script = (
        "import glob, re, json, torch\n"
        f"paths = glob.glob('/app/cray/jobs/{job_hash}/checkpoint_*.pt')\n"
        "def step(p):\n"
        "    m = re.search(r'checkpoint_(\\d+)\\.pt', p)\n"
        "    return int(m.group(1)) if m else -1\n"
        "paths.sort(key=step)\n"
        "ckpt = torch.load(paths[-1], map_location='cpu', weights_only=False) if paths else None\n"
        "print(json.dumps(list(ckpt['model_state_dict'].keys()) if ckpt else None))\n"
    )
    cmd = ["docker", "compose", "-f", "docker-compose.yaml", "exec", "-T", compose_service,
           "python3", "-c", script]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                 timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
```

- [ ] **Step 3: Sanity-check it still compiles**

Run: `cd /home/georgi/projects/scalarlm && python3 -m py_compile test/finetune_sweep/run_finetune_sweep.py && python3 -m pytest test/unit/test_finetune_sweep.py -q`
Expected: compiles silently; unit tests still all pass.

- [ ] **Step 4: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py
git commit -m "finetune-sweep: add restart/teardown, GPU probe, and checkpoint-key helpers"
```

---

### Task 5: Orchestration (`run_model`), reporting, and CLI

**Files:**
- Modify: `test/finetune_sweep/run_finetune_sweep.py`

- [ ] **Step 1: Add imports**

Change:

```python
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
```

to:

```python
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
```

- [ ] **Step 2: Append `run_model`**

Add at the end of `test/finetune_sweep/run_finetune_sweep.py`:

```python
def run_model(manifest: dict, target: str, model: dict, args, results_dir: Path) -> Result:
    model_id = model["id"]
    res = Result(model=model_id, target=target)

    target_cfg = manifest["targets"][target]
    free_gb = probe_gpu_free_gb() if target == "cuda" else []
    ok, reason = gate_model(model, target, free_gb)
    if not ok:
        res.outcome, res.detail = SKIPPED, reason
        return res

    train_args = {
        "llm_name": model_id,
        **manifest["train_args_defaults"],
        **target_cfg.get("train_args_overrides", {}),
        "sweep_run_id": args.sweep_run_id,
    }
    dataset = build_dataset(manifest["dataset"])
    golden_prompt = manifest["golden_prompt"]
    expected_output = manifest["expected_output"]

    log_path = results_dir / f"{model_id.replace('/', '_')}.{target}.restart.log"
    restart_start = time.time()
    with open(log_path, "w") as log:
        proc = start_restart(target_cfg, model_id, log)
        try:
            ready = wait_for_all_up(args.api_url, proc, args.restart_timeout)
            res.restart_seconds = round(time.time() - restart_start, 1)
            if not ready:
                res.outcome = RESTART_FAILED
                res.detail = "stack did not report health.all == 'up' in time"
                return res

            try:
                baseline = generate(args.api_url, [golden_prompt], model_id, args.max_tokens)
                res.baseline_sample = baseline[0][:200]
                if expected_output in baseline[0]:
                    res.detail = "expected_output already present in baseline output"
            except Exception as e:
                res.outcome = RESTART_FAILED
                res.detail = f"baseline generate failed: {e}"
                return res

            train_start = time.time()
            try:
                job_status = submit_train(args.api_url, dataset, train_args)
            except Exception as e:
                res.outcome = TRAIN_FAILED
                res.detail = f"submit_train failed: {e}"
                return res
            job_hash = job_status["job_directory"].rstrip("/").split("/")[-1]

            train_status = "TIMEOUT"
            deadline = time.time() + args.train_timeout
            while time.time() < deadline:
                info = get_training_job(args.api_url, job_hash)
                st = info.get("status")
                if st in ("COMPLETED", "FAILED", "CANCELLED"):
                    train_status = st
                    break
                time.sleep(5)
            res.train_seconds = round(time.time() - train_start, 1)

            checkpoint_keys: list[str] | None = None
            adapter_loaded = False
            adapter_text = ""
            last_serve_error = ""
            serve_start = time.time()

            if train_status == "COMPLETED":
                checkpoint_keys = read_checkpoint_keys(target_cfg["compose_service"], job_hash)
                if checkpoint_lora_keys_ok(checkpoint_keys):
                    serve_deadline = time.time() + args.serve_timeout
                    while time.time() < serve_deadline:
                        try:
                            adapter_out = generate(args.api_url, [golden_prompt], job_hash, args.max_tokens)
                            adapter_text = adapter_out[0]
                            adapter_loaded = True
                            break
                        except Exception as e:
                            last_serve_error = str(e)
                            time.sleep(5)
            res.serve_seconds = round(time.time() - serve_start, 1)

            memorized = expected_output in adapter_text
            res.outcome = classify_result(train_status, checkpoint_keys, adapter_loaded, memorized)
            res.adapter_sample = adapter_text[:200]

            if res.outcome == TRAIN_FAILED:
                res.detail = f"job ended with status {train_status}"
            elif res.outcome == TRAIN_TIMEOUT:
                res.detail = f"job did not reach a terminal status within {args.train_timeout}s"
            elif res.outcome == BAD_CHECKPOINT:
                res.detail = f"checkpoint keys: {checkpoint_keys}"
            elif res.outcome == ADAPTER_NOT_LOADED:
                res.detail = f"adapter never became servable; last error: {last_serve_error}"
            elif res.outcome == NO_MEMORIZATION:
                extra = "adapter served but did not memorize expected_output"
                res.detail = f"{res.detail}; {extra}" if res.detail else extra
            return res
        finally:
            teardown_stack(proc)
```

- [ ] **Step 3: Append `write_reports` and `main`**

Add at the end of `test/finetune_sweep/run_finetune_sweep.py`:

```python
def write_reports(results: list[Result], target: str, results_dir: Path) -> tuple[Path, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"finetune.{target}.{stamp}.json"
    md_path = results_dir / f"finetune.{target}.{stamp}.md"

    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    lines = [f"# Fine-tune sweep — `{target}` — {stamp}", "",
             "| Model | Outcome | Detail | Baseline sample | Adapter sample | restart_s | train_s | serve_s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        baseline = r.baseline_sample.replace("\n", " ").replace("|", "\\|")[:60]
        adapter = r.adapter_sample.replace("\n", " ").replace("|", "\\|")[:60]
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| `{r.model}` | {r.outcome} | {detail} | {baseline} | {adapter} "
                     f"| {r.restart_seconds} | {r.train_seconds} | {r.serve_seconds} |")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune sweep (tier d, LoRA only).")
    ap.add_argument("--target", required=True, choices=["cpu", "cuda"])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--models", nargs="*", help="optional subset of model IDs to run")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--restart-timeout", type=int, default=600)
    ap.add_argument("--train-timeout", type=int, default=600)
    ap.add_argument("--serve-timeout", type=int, default=300)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.target not in manifest["targets"]:
        ap.error(f"unknown target {args.target!r}; have {list(manifest['targets'])}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.sweep_run_id = uuid.uuid4().hex

    models = filter_models(manifest["models"], args.models)

    results = []
    for m in models:
        print(f"\n=== {m['id']} [{args.target}] ===", flush=True)
        r = run_model(manifest, args.target, m, args, args.results_dir)
        print(f"--> {r.model}: {r.outcome} ({r.detail})", flush=True)
        results.append(r)

    json_path, md_path = write_reports(results, args.target, args.results_dir)
    print("\n" + md_path.read_text())
    print(f"\nWrote {json_path}\n      {md_path}")

    hard_fail = any(r.outcome not in NON_FAILING_OUTCOMES for r in results)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Sanity-check it compiles, `--help` works, and unit tests still pass**

Run:
```bash
cd /home/georgi/projects/scalarlm
python3 -m py_compile test/finetune_sweep/run_finetune_sweep.py
python3 test/finetune_sweep/run_finetune_sweep.py --help
python3 -m pytest test/unit/test_finetune_sweep.py -q
```
Expected: `--help` prints usage with `--target {cpu,cuda}` and the other flags;
unit tests all pass.

- [ ] **Step 5: Commit**

```bash
git add test/finetune_sweep/run_finetune_sweep.py
git commit -m "finetune-sweep: add run_model orchestration, reporting, and CLI"
```

---

### Task 6: CPU smoke test

**Files:** none (validation step; fix forward in `test/finetune_sweep/run_finetune_sweep.py`
and/or `test/finetune_sweep/finetune-sweep.yaml` if bugs are found).

This is the runner's actual validation, the same way `run_sweep.py` has no unit
tests of its own — it's an integration driver, proven by running it (per the
design spec's testing plan).

- [ ] **Step 1: Confirm the cray stack isn't already running on this host**

Run: `docker compose -f docker-compose.yaml ps`
If a `cray*` service is already up from other work, note it — `run_finetune_sweep.py`
will SIGKILL its restart process and force-recreate, which will disrupt it.

- [ ] **Step 2: Run the sweep against `--target cpu`, one model**

Run (from repo root, expect this to take several minutes — restart + a 300-step
CPU training run + serve):
```bash
cd /home/georgi/projects/scalarlm
python3 test/finetune_sweep/run_finetune_sweep.py --target cpu --models tiny-random/gemma-4-dense
```

Expected:
- Stack restarts (`SCALARLM_MODEL=tiny-random/gemma-4-dense ./scalarlm up cpu`),
  `wait_for_all_up` reports ready.
- Baseline generate succeeds.
- Training job submits, reaches `COMPLETED` within `--train-timeout` (default 600s).
- `read_checkpoint_keys` returns LoRA keys (`checkpoint_lora_keys_ok` true).
- Adapter generate eventually succeeds (`ADAPTER_NOT_LOADED` would mean the
  hot-load loop never picked it up within `--serve-timeout`).
- Final outcome is `NO_MEMORIZATION` (expected per the design spec's open
  question — LoRA memorization isn't yet achieved at these hyperparameters) and
  exit code is `0` (NO_MEMORIZATION is non-failing).
- `test/finetune_sweep/results/finetune.cpu.<timestamp>.{json,md}` written.

- [ ] **Step 3: If it fails, diagnose and fix forward**

Common failure points and where to look:
- `RESTART_FAILED` / restart log at `test/finetune_sweep/results/<model>.cpu.restart.log`
  — check `./scalarlm up cpu` itself works manually first.
- `submit_train failed` — check the multipart body shape in `submit_train` against
  `sdk/masint/engines/cray/submit_training_job.py`'s `make_multipart_writer`.
- `TRAIN_FAILED` / `TRAIN_TIMEOUT` — check `test/finetune_sweep/results/*.json` for
  the job's `history`; cross-check against
  `infra/cray_infra/training/launch_training_job.py` (e.g. `gpus: 0` override
  actually applied).
- `BAD_CHECKPOINT` — run `read_checkpoint_keys`'s inline script manually via
  `docker compose -f docker-compose.yaml exec -T cray python3 -c "..."` to see
  the raw error.
- `ADAPTER_NOT_LOADED` — check `--serve-timeout` is generous enough; the hot-load
  loop (`add_adaptors`) runs on its own schedule.

If the fix is in `run_finetune_sweep.py` or `finetune-sweep.yaml`, re-run Step 2
until you reach `NO_MEMORIZATION`/exit 0 (or a `PASS`, if memorization happens to
already work — also fine), then:

```bash
git add test/finetune_sweep/run_finetune_sweep.py test/finetune_sweep/finetune-sweep.yaml
git commit -m "finetune-sweep: fix issues found by the cpu smoke test"
```

(Skip the commit if Step 2 passed on the first try — nothing to fix.)

---

### Task 7: ADR 0003 — restart-per-model

**Files:**
- Create: `docs/adr/0003-finetune-sweep-restart-per-model.md`

- [ ] **Step 1: Write the ADR**

```markdown
# Fine-tune sweep restarts the stack per model

`test/finetune_sweep/run_finetune_sweep.py` is a tier-(d) integration sweep: for
each model, it submits a tiny **LoRA** fine-tune job through the cray stack,
verifies the resulting checkpoint, hot-loads the adapter into the running vLLM
engine, and checks whether the served output reflects the fine-tune. To do the
hot-load+serve check, the cray server must already be **serving that model as the
Base Model** — so the runner restarts the whole stack once per model.

## Why

**Why restart per model.** The hot-load+generate check needs `config["model"]`
(the Base Model, bound at vLLM startup) to match the model being fine-tuned —
per `CONTEXT.md`, changing it requires a full server restart, not just an engine
swap. Three options were considered:

1. Train all models, serve-check only the currently-served one — no restart, but
   only one model per run gets the full closed-loop check.
2. **Restart the stack per model (chosen)** — every model gets the full closed
   loop. The chosen model set is 4 tiny-random stubs with low-double-digit-second
   GPU time, so the restart cost is small relative to full per-model coverage.
3. Single-model invocation, operator manages restarts — simplest runner, but no
   single command sweeps multiple models.

Since this sweep is **LoRA only**, only one restart per model is needed (no
second adapter-type pass within the same serving session — see ADR 0004).

**Why a separate manifest from `model-sweep.yaml`.** `test/finetune_sweep/finetune-sweep.yaml`
has a different model set (4 tiny-random stubs vs. the full serve-test catalog)
and per-model fields specific to fine-tuning (`adapters.lora.gate_gb`, no
`requires`/`chat_template_kwargs`). The two can converge later if the model sets
converge; for now a separate file avoids overloading `model-sweep.yaml`'s schema.

**Why the runner runs on the host, not inside the cray container.**
`./scalarlm up <target>` is `docker compose -f docker-compose.yaml up <service>
--build --force-recreate` — a host-level operation. A runner living inside the
container it's restarting would kill itself. The runner therefore:

- talks to the cray API over plain HTTP (stdlib `urllib`), not the `scalarlm` SDK
  (which only imports inside the container);
- probes free VRAM via `nvidia-smi` (not `torch`);
- uses a single `docker compose exec` to read LoRA checkpoint keys via
  in-container `torch.load` — the only step that genuinely needs the container's
  Python environment.

**Why a per-target `gpus` override.** `JobConfig` defaults `gpus: 1`
(`infra/cray_infra/util/default_job_config.py`), and `validate_gpu_request`
(`infra/cray_infra/training/launch_training_job.py`) rejects `gpus >= 1` with
HTTP 400 when no SLURM node advertises a GPU — true on the `cpu` target. The
manifest's `targets.cpu.train_args_overrides: {gpus: 0}` opts out; `cuda` sets
`{gpus: 1}` to make the override explicit and symmetric.

**Why `must_contain`-on-a-random-string is the pass criterion, and why
`NO_MEMORIZATION` is non-failing.** The training set teaches the model to map
`golden_prompt` -> a random hex string a random-weight model will never produce
unattributed. Its presence in the adapter's response is a clean, unambiguous
signal that the adapter trained AND is being served — without needing the
output to otherwise be coherent. As of this sweep's design, LoRA memorization at
the current hyperparameters (`max_steps: 300`, `learning_rate: 3e-2`,
`train_lm_head=False`) has not been achieved (see the design spec's "Open
questions"); reporting `NO_MEMORIZATION` as a hard failure would make the sweep
permanently red for a known, documented reason. It is therefore a non-failing
outcome — `ADAPTER_NOT_LOADED`, `BAD_CHECKPOINT`, `TRAIN_FAILED`,
`TRAIN_TIMEOUT`, and `RESTART_FAILED` remain hard failures.

## Shape

- **Restart mechanism**: `subprocess.Popen(cmd, shell=True, start_new_session=True)`
  launches `SCALARLM_MODEL={model} ./scalarlm up {target}` in its own process
  group (mirrors `test/model_sweep/run_sweep.py:279-280`). The runner polls
  `GET /v1/health` until `"all": "up"` (`wait_for_all_up`) or `--restart-timeout`
  -> `RESTART_FAILED`. Teardown is `os.killpg(..., SIGKILL)` + a VRAM-reclamation
  wait (`teardown_stack`, mirrors `teardown_engine`,
  `test/model_sweep/run_sweep.py:111-138`) — `--force-recreate` on the next `up`
  handles container cleanup. Applies identically to `cpu` and `cuda`.
- **Outcome enum** (best -> worst): `PASS`, `NO_MEMORIZATION`,
  `ADAPTER_NOT_LOADED`, `BAD_CHECKPOINT`, `TRAIN_FAILED`, `TRAIN_TIMEOUT`,
  `RESTART_FAILED` (model-level), `SKIPPED` (model-level — LoRA gate doesn't fit,
  or `cpu_ok: false` on cpu).
- **Dedup defeat**: each invocation injects a `sweep_run_id` nonce into
  `train_args` so `launch_training_job`'s `sha256(train_args + dataset)` job-dir
  cache never returns a stale job for a fresh sweep run.

## Consequences

- Per-model wall time is dominated by the restart and the training job, not the
  serve check — but every model gets the full train -> hot-load -> serve loop,
  not just the currently-served one.
- Until the LoRA-memorization open question (ADR-adjacent, see the design spec)
  is resolved, every row is expected to report `NO_MEMORIZATION`. The sweep
  stays green but does not yet *prove* memorization — only that the pipeline
  mechanics (train, checkpoint, hot-load, serve) work end to end.
- A future Tokenformer pass (ADR 0004) would need either a second restart cycle
  per model (if it needs its own serving mode) or to be folded into the same
  serving session if/when Tokenformer can be hot-loaded alongside LoRA.
```

- [ ] **Step 2: Commit**

```bash
git add docs/adr/0003-finetune-sweep-restart-per-model.md
git commit -m "docs: add ADR 0003 (fine-tune sweep restarts per model)"
```

---

### Task 8: ADR 0004 — defer Tokenformer serving

**Files:**
- Create: `docs/adr/0004-defer-tokenformer-serving.md`

- [ ] **Step 1: Write the ADR**

```markdown
# Defer Tokenformer serving; tier (d) covers LoRA only

`CONTEXT.md` originally framed tier (d) ("Adapter training+serving compatibility")
as covering both **LoRA** and **Tokenformer** adapters. The fine-tune sweep
(`test/finetune_sweep/run_finetune_sweep.py`, ADR 0003) covers **LoRA only**;
Tokenformer is excluded from this sweep entirely, not just skipped per-model.

## Why

While grilling the design spec, the following was verified empirically against
`tiny-random/gemma-4-dense` on a remote `cuda` box:

- Training a Tokenformer adapter completes and produces a `.pt` checkpoint with
  `tokenformer_p` keys — **training works**.
- Hot-loading it does not: `add_new_adaptor()`
  (`infra/cray_infra/one_server/create_generate_worker.py:251-290`) always calls
  vLLM's `/v1/load_lora_adapter`, regardless of adapter type. The vLLM fork
  rejects a Tokenformer-keyed checkpoint: *"Adapter ... has no LoRA tensors
  (found only Tokenformer keys). Serve with `--enable-tokenformer` instead, or as
  a hybrid adapter with both `--enable-lora` and `--enable-tokenformer`."*
- `--enable-tokenformer` is never passed —
  `infra/cray_infra/one_server/vllm_cli_args.py` only conditionally adds
  `--enable-lora`.
- `infra/cray_infra/adapters/` (`TokenformerManager`, `attention_adapter.py`,
  `models.py`) is a from-scratch Tokenformer serving layer that is **never
  imported by the running server** — dead scaffolding. Even if wired up, its
  core transform (`attention_adapter.py:_tokenformer_transform`) is a no-op stub
  that returns its input unchanged.

Implementing real Tokenformer serving (a real forward-pass algorithm with
per-request adapter selection compatible with continuous batching, plus a
non-LoRA load endpoint) is a multi-week vLLM-fork project — out of scope for this
sweep.

## Shape

- `CONTEXT.md`'s **Adapter**, **Tokenformer**, and **Integration-test** entries
  were corrected to stop claiming Tokenformer is "served as a LoRA at inference"
  and to scope tier (d) to LoRA until Tokenformer serving lands.
- `Result.adapter_type` (always `"lora"` for now) is kept on the `Result`
  dataclass for forward compatibility with a future Tokenformer pass, rather than
  hardcoding LoRA-specific field names throughout.
- The fine-tune sweep manifest (`test/finetune_sweep/finetune-sweep.yaml`) has no
  Tokenformer-specific fields (e.g. no `adapters.tokenformer.gate_gb`).

## Consequences

- Tier (d) currently proves only LoRA train -> hot-load -> serve. A model could
  have a fully broken Tokenformer training path and this sweep would not catch
  it (Tokenformer training itself is exercised informally, e.g. in
  `docs/reports/fine-tuning-a-served-model.md`'s worked example, but not by an
  automated sweep).
- Once `--enable-tokenformer` is wired up and `_tokenformer_transform` does real
  work, this spec can grow a second adapter-type pass per model — the manifest
  would gain `adapters.tokenformer.gate_gb` per model and `run_model` would loop
  over adapter types, producing one `Result` row per (model, target,
  adapter_type) as the `Result.adapter_type` field already anticipates.
```

- [ ] **Step 2: Commit**

```bash
git add docs/adr/0004-defer-tokenformer-serving.md
git commit -m "docs: add ADR 0004 (defer Tokenformer serving)"
```

---

## Self-Review

**Spec coverage:**
- Manifest schema (dataset, golden_prompt/expected_output, train_args_defaults,
  targets, models with `adapters.lora.gate_gb`) — Task 1. Plus the
  per-target `compose_service`/`train_args_overrides` additions resolved during
  brainstorming.
- Restart mechanism (Popen+SIGKILL, `wait_for_all_up` on `health.all`) — Task 4 +
  Task 5.
- Runner flow steps 1-4g (gate, restart, baseline, train submit/poll, checkpoint
  check, adapter poll+generate, classify) — Task 5 (`run_model`).
- Outcome enum + `NON_FAILING_OUTCOMES` — Task 2.
- `Result` fields — Task 2.
- Reporting (JSON+MD, table columns) and CLI flags — Task 5.
- Dedup-defeat nonce (`sweep_run_id`) — Task 5 (`run_model`'s `train_args`).
- `--models` filtering — Task 2 (`filter_models`) + Task 5 (`main`).
- Unit tests for gating / checkpoint keys / outcome classification / manifest
  helpers — Task 2.
- Runner validated by running it (`--target cpu`) — Task 6.
- ADR 0003 (restart-per-model, separate manifest, host execution, `gpus`
  override, memorization criterion) — Task 7.
- ADR 0004 (Tokenformer findings + LoRA-only scope) — Task 8.

**Placeholder scan:** no `TBD`/`TODO`/"implement later" remain; every step has
complete code or an exact command with expected output.

**Type/signature consistency:** `gate_model(model, target, free_gb) -> (bool, str)`,
`checkpoint_lora_keys_ok(keys) -> bool`, `classify_result(train_status,
checkpoint_keys, adapter_loaded, memorized) -> str`, `filter_models(models,
wanted) -> list[dict]`, `build_dataset(spec) -> list[dict]` are defined in Task 2
and used with the same names/signatures in Task 5's `run_model`. `Result` fields
(`baseline_sample`, `adapter_sample`, `restart_seconds`, `train_seconds`,
`serve_seconds`, `adapter_type`, `outcome`, `detail`) defined in Task 2 match
every assignment in Task 5 and every column in `write_reports`. HTTP helpers
(`get_health`, `wait_for_all_up`, `submit_train`, `get_training_job`, `generate`)
from Task 3 and process helpers (`probe_gpu_free_gb`, `start_restart`,
`teardown_stack`, `read_checkpoint_keys`) from Task 4 are called in Task 5 with
matching names/argument orders.
