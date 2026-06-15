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

import argparse
import datetime as dt
import io
import json
import os
import re
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


def k8s_namespace(prefix: str, model_id: str) -> str:
    """Sanitize a model id into an RFC1123 namespace label: lowercase, every run
    of non-alphanumeric chars -> '-', trimmed, prefixed, truncated to 63 chars."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{prefix}-{slug}"[:63].rstrip("-")


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
    except OSError:
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
    print(f"[warn] teardown_stack: VRAM did not settle within {settle_timeout}s", flush=True)


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
                try:
                    info = get_training_job(args.api_url, job_hash)
                except (urllib.error.URLError, RuntimeError):
                    time.sleep(5)
                    continue
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
            serve_start = time.time()  # only meaningful if train_status == "COMPLETED"

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
