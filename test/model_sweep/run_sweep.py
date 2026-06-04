#!/usr/bin/env python3
"""Base-model serve-test sweep — tiers (a) does-it-serve and (b) output-sanity.

Serve-only test driver — shares no code with the cray-stack integration harness;
the manifest is the durable contract between them. Drives `vllm serve` as one
subprocess per model INSIDE the target image (never the cray server) and
classifies each model into an outcome enum. See
docs/adr/0001-serve-tests-bypass-cray.md and model-sweep.yaml.

Run inside the built image for the target device, e.g.:

    python run_sweep.py --target blackwell-4x
    python run_sweep.py --target cpu --models Qwen/Qwen2-7B-Instruct

This drives `vllm serve` directly inside the image, NOT via docker-compose, so it
does NOT inherit compose's HF-cache bind mount. Launch the container with your
HuggingFace cache bind-mounted to /root/.cache/huggingface (the default cache
path — no HF_HOME is set), e.g.:

    docker run --rm --gpus all \
        -v /path/to/hf_model_cache:/root/.cache/huggingface \
        -v "$PWD/test:/app/cray/test" \
        <image> python /app/cray/test/model_sweep/run_sweep.py --target pc

Without that mount every model re-downloads on every run, and --startup-timeout
(init-only; it assumes a warm cache) will fire mid-download on large models.

Gated models need HF_TOKEN in the environment; missing token -> outcome GATED.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
DEFAULT_MANIFEST = HERE / "model-sweep.yaml"
DEFAULT_RESULTS_DIR = HERE / "results"

# Outcome enum (see ADR). Ordered loosely best -> worst for reporting.
SERVED_PASS = "SERVED_PASS"
SERVED_FAIL_QUALITY = "SERVED_FAIL_QUALITY"
FAILED_TO_SERVE = "FAILED_TO_SERVE"
OOM = "OOM"
GATED = "GATED"
MISSING = "MISSING"
SKIPPED = "SKIPPED"

# Patterns scanned in a crashed child's log to tell failure causes apart.
_OOM_RE = re.compile(r"out of memory|CUDA out of memory|OOM", re.I)
_MISSING_RE = re.compile(r"Repository Not Found|404 Client Error|does not appear to have", re.I)
_GATED_RE = re.compile(r"401 Client Error|gated repo|awaiting a review|access to model", re.I)


@dataclass
class Result:
    model: str
    target: str
    outcome: str
    detail: str = ""
    prompt_results: list[dict[str, Any]] = field(default_factory=list)
    sample: str = ""
    seconds: float = 0.0


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merged_flags(manifest: dict, target: str, model: dict) -> dict:
    """target.defaults overlaid with this model's per-target overrides."""
    tdef = manifest["targets"][target].get("defaults", {})
    over = (model.get("targets", {}).get(target) or {})
    return {**tdef, **{k: v for k, v in over.items() if k != "skip"}}


def build_command(model_id: str, flags: dict, multimodal: bool, port: int) -> list[str]:
    cmd = ["vllm", "serve", model_id, "--port", str(port), "--trust-remote-code"]
    if "dtype" in flags:
        cmd += ["--dtype", str(flags["dtype"])]
    if "tensor_parallel_size" in flags:
        cmd += ["--tensor-parallel-size", str(flags["tensor_parallel_size"])]
    if "gpu_memory_utilization" in flags:
        cmd += ["--gpu-memory-utilization", str(flags["gpu_memory_utilization"])]
    if multimodal:
        cmd += ["--limit-mm-per-prompt", '{"image":1}']
    return cmd


def wait_for_health(port: int, proc: subprocess.Popen, timeout: int) -> bool:
    """Poll /health until 200, the child dies, or timeout. True iff ready."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # child exited before becoming ready
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def run_prompt(port: int, model_id: str, prompt: str, timeout: int = 120) -> tuple[int, str]:
    """POST one chat completion. Returns (http_status, text)."""
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
            text = payload["choices"][0]["message"]["content"]
            return r.status, text
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001 - report any transport failure verbatim
        return 0, f"{type(e).__name__}: {e}"


def check_assertions(text: str, spec: dict) -> tuple[bool, str]:
    """Tier (b) cheap deterministic checks. Returns (passed, reason)."""
    if "must_contain" in spec and spec["must_contain"] not in text:
        return False, f"missing substring {spec['must_contain']!r}"
    if spec.get("must_parse_json"):
        blob = text.strip()
        m = re.search(r"\{.*\}", blob, re.S)  # tolerate prose around the JSON
        try:
            json.loads(m.group(0) if m else blob)
        except (json.JSONDecodeError, AttributeError):
            return False, "not valid JSON"
    if "max_words" in spec and len(text.split()) > spec["max_words"]:
        return False, f"{len(text.split())} words > max {spec['max_words']}"
    return True, ""


def classify_log(logfile: Path) -> tuple[str, str]:
    """Map a crashed child's log to an outcome (OOM/MISSING/GATED/FAILED)."""
    tail = ""
    try:
        tail = logfile.read_text(errors="replace")[-4000:]
    except OSError:
        pass
    if _OOM_RE.search(tail):
        return OOM, "out-of-memory in child log"
    if _MISSING_RE.search(tail):
        return MISSING, "repo not found (404)"
    if _GATED_RE.search(tail):
        return GATED, "gated / unauthorized (401)"
    return FAILED_TO_SERVE, "engine exited before /health (see log)"


def test_model(manifest: dict, target: str, model: dict, args) -> Result:
    model_id = model["id"]
    res = Result(model=model_id, target=target, outcome=SKIPPED)
    start = time.time()

    # Skip / gating pre-checks — never launch in these cases.
    over = (model.get("targets", {}).get(target) or {})
    if "skip" in over and over["skip"]:
        res.detail = over["skip"] if isinstance(over["skip"], str) else "skipped for target"
        return res
    if model.get("gated") and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        res.outcome, res.detail = GATED, "gated model and no HF_TOKEN in environment"
        return res

    flags = merged_flags(manifest, target, model)
    # Cheap pre-flight: refuse an impossible tensor-parallel size (e.g. TP=4 on a
    # 1-GPU box) before burning a download + NCCL hang. Outcome stays SKIPPED.
    gpus = manifest["targets"][target].get("gpus_available")
    tp = int(flags.get("tensor_parallel_size", 1))
    if gpus is not None and tp > gpus:
        res.detail = f"tensor_parallel_size={tp} > gpus_available={gpus} on target {target!r}"
        return res
    cmd = build_command(model_id, flags, model.get("multimodal", False), args.port)

    logfile = args.results_dir / f"{model_id.replace('/', '_')}.{target}.log"
    print(f"\n=== {model_id} [{target}] ===\n$ {' '.join(cmd)}", flush=True)

    with open(logfile, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            if not wait_for_health(args.port, proc, args.startup_timeout):
                res.outcome, res.detail = (classify_log(logfile) if proc.poll() is not None
                                           else (FAILED_TO_SERVE, "timed out waiting for /health"))
                return res

            # Tier (a) gate is implicit in a successful, non-empty completion.
            all_pass = True
            for spec in manifest.get("prompts", []):
                status, text = run_prompt(args.port, model_id, spec["prompt"])
                # Transport failure with a now-dead child = the engine crashed
                # mid-run (e.g. runtime OOM). Reclassify from the log rather than
                # call it a quality failure — see ADR 0001 ("OOM never masquerades").
                if status == 0 and proc.poll() is not None:
                    res.prompt_results.append({
                        "id": spec["id"], "status": status,
                        "passed": False, "reason": "engine died mid-request",
                        "text": text[:200],
                    })
                    res.outcome, res.detail = classify_log(logfile)
                    return res
                ok_serve = status == 200 and bool(text.strip())
                ok_assert, why = check_assertions(text, spec) if ok_serve else (False, f"HTTP {status}")
                all_pass = all_pass and ok_serve and ok_assert
                res.prompt_results.append({
                    "id": spec["id"], "status": status,
                    "passed": ok_serve and ok_assert, "reason": why,
                    "text": text[:200],
                })
                if spec["id"] == "arithmetic":
                    res.sample = text[:200]
            res.outcome = SERVED_PASS if all_pass else SERVED_FAIL_QUALITY
        finally:
            # Tear the child (and its group) down so VRAM is fully reclaimed.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            res.seconds = round(time.time() - start, 1)
    return res


def write_reports(results: list[Result], target: str, results_dir: Path) -> tuple[Path, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"sweep.{target}.{stamp}.json"
    md_path = results_dir / f"sweep.{target}.{stamp}.md"

    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    lines = [f"# Serve-test sweep — `{target}` — {stamp}", "",
             "| Model | Outcome | Detail | Sample (arithmetic) | s |",
             "|---|---|---|---|---|"]
    for r in results:
        sample = r.sample.replace("\n", " ").replace("|", "\\|")[:60]
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| `{r.model}` | {r.outcome} | {detail} | {sample} | {r.seconds} |")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Base-model serve-test sweep (tiers a/b).")
    ap.add_argument("--target", required=True, help="target key from the manifest")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--models", nargs="*", help="optional subset of model IDs to run")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout", type=int, default=600,
                    help="seconds to wait for /health (engine init; assumes a warm "
                         "HF cache — bind-mount it, see the module docstring)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.target not in manifest["targets"]:
        ap.error(f"unknown target {args.target!r}; have {list(manifest['targets'])}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    models = manifest["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["id"] in wanted]

    results = [test_model(manifest, args.target, m, args) for m in models]

    json_path, md_path = write_reports(results, args.target, args.results_dir)
    print("\n" + md_path.read_text())
    print(f"\nWrote {json_path}\n      {md_path}")

    # Non-zero exit if any model that we actually attempted failed to serve.
    hard_fail = any(r.outcome in (FAILED_TO_SERVE, OOM) for r in results)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
