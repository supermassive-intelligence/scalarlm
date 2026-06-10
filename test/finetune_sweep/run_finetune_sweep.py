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


def get_health(api_url: str, timeout: float = 5) -> dict | None:
    req = urllib.request.Request(f"{api_url}/v1/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, OSError, ValueError):
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["job_status"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST /v1/megatron/train failed {e.code}: {e.read().decode()}") from e


def get_training_job(api_url: str, job_hash: str, timeout: float = 10) -> dict:
    req = urllib.request.Request(f"{api_url}/v1/megatron/train/{job_hash}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["job_status"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GET /v1/megatron/train/{job_hash} failed {e.code}: {e.read().decode()}") from e


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
