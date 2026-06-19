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
    python3 run_finetune_sweep.py --target cuda-docker            # GPU via Docker Compose (scheduler-less box)
    python3 run_finetune_sweep.py --target cuda-k8s --models Qwen/Qwen2.5-0.5B
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
ADAPTER_NO_OP = "ADAPTER_NO_OP"
PRECHECK_NO_OP = "PRECHECK_NO_OP"
ADAPTER_NOT_LOADED = "ADAPTER_NOT_LOADED"
BAD_CHECKPOINT = "BAD_CHECKPOINT"
TRAIN_FAILED = "TRAIN_FAILED"
TRAIN_TIMEOUT = "TRAIN_TIMEOUT"
RESTART_FAILED = "RESTART_FAILED"
SKIPPED = "SKIPPED"

# NO_MEMORIZATION is expected (not yet a hard fail) until the LoRA-memorization
# open question in the design spec is resolved. ADAPTER_NO_OP and PRECHECK_NO_OP
# are both failing: the adapter served (or is predicted to serve) base output.
NON_FAILING_OUTCOMES = {PASS, SKIPPED, NO_MEMORIZATION}


@dataclass
class Result:
    model: str
    target: str
    adapter_type: str = "lora"
    outcome: str = SKIPPED
    detail: str = ""
    hint: str = ""  # root-cause hint (preflight key diff, or scraped no-op warning)
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


def gate_model(model: dict, target_cfg: dict, free_gb: list[float] | None) -> tuple[bool, str]:
    """Decide whether `model` should run on this target. A CPU target (no GPU
    requested) is opt-in via cpu_ok; a GPU target is gated on the LoRA VRAM gate
    vs. probed free VRAM. `free_gb is None` means the VRAM check is not applicable
    (k8s: the scheduler arbitrates GPU fit) — only the static checks apply.
    Branches on target config (target_requests_gpu), not the literal target name."""
    if not target_requests_gpu(target_cfg):
        if not model.get("cpu_ok"):
            return False, "no cpu_ok opt-in for this model"
        return True, ""

    gate_gb = model.get("adapters", {}).get("lora", {}).get("gate_gb")
    if gate_gb is None:
        return False, "no adapters.lora.gate_gb declared"
    if free_gb is None:
        return True, ""  # k8s: scheduler arbitrates GPU fit; skip the VRAM check
    if not free_gb or max(free_gb) < gate_gb:
        return False, (f"LoRA needs >={gate_gb:g}GiB free; "
                        f"free GiB: {[round(f, 1) for f in free_gb]}")
    return True, ""


# Container waiting reasons that mean the rollout is broken, not just slow.
FATAL_WAITING_REASONS = {
    "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "InvalidImageName",
    "CreateContainerConfigError", "CreateContainerError", "RunContainerError",
}


def classify_pod_status(pods: list[dict]) -> str:
    """Classify a namespace's pods for block-and-wait scheduling. Returns:
      "failed"  - a pod is in a fatal state (crash / bad image / Failed phase);
                  stop and fail fast (RESTART_FAILED).
      "ready"   - every pod is Running with all containers ready; proceed.
      "pending" - nothing fatal yet but not all ready (incl. Pending /
                  Unschedulable / ContainerCreating); keep waiting for the
                  scheduler to place the GPU pods.
    """
    if not pods:
        return "pending"
    all_ready = True
    for pod in pods:
        status = pod.get("status", {})
        if status.get("phase") == "Failed":
            return "failed"
        container_statuses = status.get("containerStatuses", [])
        if not container_statuses or status.get("phase") != "Running":
            all_ready = False
        for cs in container_statuses:
            waiting = cs.get("state", {}).get("waiting")
            if waiting and waiting.get("reason") in FATAL_WAITING_REASONS:
                return "failed"
            if not cs.get("ready", False):
                all_ready = False
    return "ready" if all_ready else "pending"


def checkpoint_lora_keys_ok(state_dict_keys: list[str] | None) -> bool:
    """True iff the checkpoint's state-dict keys include both LoRA matrices."""
    if not state_dict_keys:
        return False
    return (any("lora_A" in k for k in state_dict_keys)
            and any("lora_B" in k for k in state_dict_keys))


def classify_result(train_status: str, checkpoint_keys: list[str] | None,
                     adapter_loaded: bool, memorized: bool,
                     adapter_is_noop: bool = False) -> str:
    """Map the per-step results of run_model's training+serving pipeline to an
    outcome. Assumes train_status == "COMPLETED" for any value not handled by
    the first two branches. `adapter_is_noop` is True when the adapter served
    output byte-identical to the baseline (LoRA silently dropped at activation);
    combined with "did not memorize" it is the ADAPTER_NO_OP failure."""
    if train_status in ("FAILED", "CANCELLED"):
        return TRAIN_FAILED
    if train_status == "TIMEOUT":
        return TRAIN_TIMEOUT
    if not checkpoint_lora_keys_ok(checkpoint_keys):
        return BAD_CHECKPOINT
    if not adapter_loaded:
        return ADAPTER_NOT_LOADED
    if adapter_is_noop and not memorized:
        return ADAPTER_NO_OP
    return PASS if memorized else NO_MEMORIZATION


def preflight_hint(pf_result) -> str:
    """Human-readable key diff for a PRECHECK_NO_OP row: the normalized adapter
    module paths the preflight produced vs a sample of the base model's modules,
    so a failure is self-explanatory without a per-model forensic dive."""
    return (f"adapter keys {pf_result.sample_adapter_keys} "
            f"vs base {pf_result.sample_base_modules}")


def is_gated_repo_error(error: str) -> bool:
    """True iff a preflight error is an HF gated-repo / auth failure (401). Unlike
    a transient build/introspection crash (which fails open), this is
    deterministic — the model can't load without access — so the sweep SKIPs it
    rather than paying a doomed restart + train + serve that ends in the same 401."""
    e = error.lower()
    return "gated repo" in e or "401 client error" in e


def split_by_preflight(models: list[dict], pf_results: dict,
                       target: str) -> tuple[list[dict], list["Result"]]:
    """Partition `models` into (to_run, skipped) using the offline preflight.
    A model whose PreflightResult `predicted_noop` is True (zero module overlap,
    no build error) is excluded as a PRECHECK_NO_OP Result with a key-diff hint;
    a model whose preflight hit an HF gated-repo/401 error is excluded as SKIPPED
    (deterministic — no point running it). Models with no entry, or whose preflight
    errored for any OTHER reason (fail open), stay in to_run — a preflight crash
    never silently skips a model."""
    to_run: list[dict] = []
    skipped: list[Result] = []
    for m in models:
        pf = pf_results.get(m["id"])
        if pf is None:
            to_run.append(m)
        elif pf.predicted_noop:
            skipped.append(Result(model=m["id"], target=target,
                                  outcome=PRECHECK_NO_OP, hint=preflight_hint(pf)))
        elif pf.error and is_gated_repo_error(pf.error):
            skipped.append(Result(model=m["id"], target=target, outcome=SKIPPED,
                                  detail="gated HF repo (preflight 401) — request access "
                                         "or wire HF_TOKEN into the compose env",
                                  hint=pf.error[:120]))
        else:
            to_run.append(m)
    return to_run, skipped


def get_health(api_url: str, timeout: float = 5) -> dict | None:
    req = urllib.request.Request(f"{api_url}/v1/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, OSError, ValueError):
        return None


def wait_for_all_up(api_url: str, proc, timeout: float, health_key: str = "all") -> bool:
    """Poll /v1/health until health[health_key] == "up", the restart process dies
    (Compose only; pass proc=None for k8s), or timeout. True iff ready. The
    phase-scaled k8s path passes health_key="megatron" (phase 1) / "vllm" (phase 2)
    because health["all"] is structurally down when only one GPU service is up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        health = get_health(api_url)
        if health and health.get(health_key) == "up":
            return True
        time.sleep(2)
    return False


def get_served_models(api_url: str, timeout: float = 5) -> set[str]:
    """Model ids vLLM currently serves (via the cray API's /v1/models proxy).
    Empty set if the endpoint is unreachable or malformed."""
    req = urllib.request.Request(f"{api_url}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return {m["id"] for m in data.get("data", [])}
    except (urllib.error.URLError, ConnectionError, OSError, ValueError, KeyError, TypeError):
        return set()


def get_compose_restart_count(compose_service: str, timeout: float = 10) -> int | None:
    """The Docker RestartCount of the compose service's container, or None if it
    can't be read. A value that climbs during wait_for_model_served means a crashed
    vLLM is being restart-looped by `restart: unless-stopped` (PID 1 is
    `one_server.main` under `set -e`, so a crash exits the container) — a broken
    model, not a slow one. The k8s path has no Compose container, so its callers
    pass compose_service=None and skip this."""
    try:
        cid = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yaml", "ps", "-q", compose_service],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout.strip()
        if not cid:
            return None
        rc = subprocess.run(
            ["docker", "inspect", "--format", "{{.RestartCount}}", cid],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout.strip()
        return int(rc)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def wait_for_model_served(api_url: str, model_id: str, proc, timeout: float,
                          compose_service: str | None = None,
                          crash_check_every: float = 15.0) -> str:
    """Like wait_for_all_up, but also requires the served model to be `model_id`.
    The Compose teardown leaves the previous model's container running (see
    teardown_stack), so health.all=="up" alone can be satisfied by the STALE
    stack before `--force-recreate` swaps in the new model — gating on
    /v1/models too makes the runner block until the new container is actually
    serving model_id. Returns:
      "served"      - health.all up AND /v1/models serves model_id.
      "crashed"     - the container's RestartCount climbed: vLLM crashed and
                      `restart: unless-stopped` is restart-looping it. Fail fast
                      instead of waiting out `timeout` (the foreground `docker
                      compose up` survives container restarts, so proc never dies).
      "proc_exited" - the foreground `./scalarlm up` process died.
      "timeout"     - none of the above within `timeout`.
    Pass compose_service to enable crash detection (Compose only)."""
    deadline = time.time() + timeout
    baseline_restarts = (get_compose_restart_count(compose_service)
                         if compose_service else None)
    last_crash_check = time.time()
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return "proc_exited"
        health = get_health(api_url)
        if health and health.get("all") == "up" and model_id in get_served_models(api_url):
            return "served"
        if (compose_service and baseline_restarts is not None
                and time.time() - last_crash_check >= crash_check_every):
            last_crash_check = time.time()
            rc = get_compose_restart_count(compose_service)
            if rc is not None and rc > baseline_restarts:
                return "crashed"
        time.sleep(2)
    return "timeout"


def poll_training(api_url: str, job_hash: str, train_timeout: float) -> str:
    """Poll the training job until terminal (COMPLETED/FAILED/CANCELLED) or
    train_timeout. Returns the terminal status or "TIMEOUT". Shared by the 2-GPU
    and phase-scaled k8s paths."""
    deadline = time.time() + train_timeout
    while time.time() < deadline:
        try:
            info = get_training_job(api_url, job_hash)
        except (OSError, RuntimeError):
            # OSError covers urllib.error.URLError AND raw socket errors like
            # ConnectionResetError (Errno 104) — the co-located training job can
            # saturate the GPU and make the API server briefly reset connections;
            # retry until train_timeout instead of crashing the run.
            time.sleep(5)
            continue
        st = info.get("status")
        if st in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(5)
    return "TIMEOUT"


def serve_check_and_classify(api_url: str, golden_prompt: str, expected_output: str,
                             job_hash: str, train_status: str,
                             checkpoint_keys: list[str] | None, args, res: "Result",
                             baseline_full: str = "") -> None:
    """Hot-load the trained adapter, classify the outcome, and set the result
    fields (serve_seconds, outcome, adapter_sample, detail). Shared by the 2-GPU
    and phase-scaled k8s paths; the caller reads `checkpoint_keys` first (the read
    mechanism differs per path) and passes the FULL baseline string for the no-op
    discriminator (byte-identical adapter output under greedy decoding -> the LoRA
    silently dropped at activation -> ADAPTER_NO_OP)."""
    adapter_loaded = False
    adapter_text = ""
    last_serve_error = ""
    serve_start = time.time()  # serve_seconds only meaningful when train_status == "COMPLETED"
    if train_status == "COMPLETED" and checkpoint_lora_keys_ok(checkpoint_keys):
        text, last_serve_error = generate_with_retry(
            api_url, golden_prompt, job_hash, args.max_tokens,
            args.serve_timeout, args.generate_timeout)
        if text is not None:
            adapter_text, adapter_loaded = text, True
    res.serve_seconds = round(time.time() - serve_start, 1)

    memorized = expected_output in adapter_text
    adapter_is_noop = adapter_loaded and adapter_text == baseline_full
    res.outcome = classify_result(train_status, checkpoint_keys, adapter_loaded,
                                  memorized, adapter_is_noop=adapter_is_noop)
    res.adapter_sample = adapter_text[:200]

    if res.outcome == TRAIN_FAILED:
        res.detail = f"job ended with status {train_status}"
    elif res.outcome == TRAIN_TIMEOUT:
        res.detail = f"job did not reach a terminal status within {args.train_timeout}s"
    elif res.outcome == BAD_CHECKPOINT:
        res.detail = f"checkpoint keys: {checkpoint_keys}"
    elif res.outcome == ADAPTER_NOT_LOADED:
        res.detail = f"adapter never became servable; last error: {last_serve_error}"
    elif res.outcome == ADAPTER_NO_OP:
        extra = "adapter served output byte-identical to baseline (LoRA not applied)"
        res.detail = f"{res.detail}; {extra}" if res.detail else extra
    elif res.outcome == NO_MEMORIZATION:
        extra = "adapter served but did not memorize expected_output"
        res.detail = f"{res.detail}; {extra}" if res.detail else extra


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
              poll_timeout: float = 300, temperature: float = 0.0,
              request_timeout: float = 30) -> list[str]:
    """POST /v1/generate then poll /v1/generate/get_results until every result has
    a response, raising on any error (mirrors handle_error/poll_for_responses in
    sdk/masint/engines/async_cray.py).
    Default temperature=0.0 ensures greedy-decoding determinism for the no-op
    discriminator. cray serves /v1/generate SYNCHRONOUSLY (the handler blocks until
    generation finishes), so request_timeout must exceed the per-call latency — the
    first call after a restart pays cold-start (engine init/CUDA-graph capture +
    worker warmup, ~146s on the GB10/DGX Spark); see generate_with_retry.
    """
    body = json.dumps({
        "prompts": prompts,
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
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


def generate_with_retry(api_url: str, prompt: str, model_name: str, max_tokens: int,
                        deadline_s: float, request_timeout: float) -> tuple[str | None, str]:
    """Call generate() for a single prompt, retrying every 5s until it succeeds or
    deadline_s elapses (always at least one attempt). Returns (text, "") on success
    or (None, last_error) if it never succeeds. Absorbs the cold-start window after
    a restart, where the first /v1/generate can block past one request_timeout while
    the engine finishes init/CUDA-graph capture + worker warmup (~146s on the
    GB10/DGX Spark). The adapter serve path always retried; this gives the baseline
    generate the same resilience instead of failing the whole run on one timeout."""
    deadline = time.time() + deadline_s
    last_error = ""
    while True:
        try:
            return generate(api_url, [prompt], model_name, max_tokens,
                            request_timeout=request_timeout)[0], ""
        except Exception as e:
            last_error = str(e)
        if time.time() >= deadline:
            return None, last_error
        time.sleep(5)


# --- k8s command builders (pure) ---

def is_k8s_target(target_cfg: dict) -> bool:
    """A target is k8s-driven when it declares a Helm chart path (vs the
    Compose target's compose_service/restart_cmd)."""
    return "chart_path" in target_cfg


def target_requests_gpu(target_cfg: dict) -> bool:
    """True iff this target runs training on a GPU. Config-driven (not keyed on
    the target name), mirroring is_k8s_target: JobConfig defaults gpus=1 and only
    the cpu target overrides it to 0, so this is True for the GPU targets
    (cuda-docker, cuda-k8s) and False for cpu."""
    return target_cfg.get("train_args_overrides", {}).get("gpus", 1) >= 1


def k8s_helm_install_cmd(target_cfg: dict, namespace: str, model_id: str) -> list[str]:
    cmd = [
        "helm", "upgrade", "--install", target_cfg["release"], target_cfg["chart_path"],
        "-n", namespace, "--create-namespace",
        "--set", f"model={model_id}",
        "--set", "storage.cache.kind=hostPath",
        "--set", f"storage.cache.hostPath={target_cfg['cache_hostpath']}",
    ]
    # Right-size the jobs PVC per target (chart default is 100Gi). Lets a small
    # model claim e.g. 20Gi so its Longhorn replica fits under the node's
    # scheduling ceiling. Omitted -> chart default applies. NOTE: the jobs PVC
    # has helm.sh/resource-policy: keep and PVCs can't shrink, so a changed size
    # only takes effect on a freshly provisioned namespace, not an in-place upgrade.
    if "jobs_size" in target_cfg:
        cmd += ["--set", f"storage.jobs.size={target_cfg['jobs_size']}"]
    if target_cfg.get("phase_scaled"):
        # Phase 0: megatron holds the single GPU; vLLM is off until the phase-2
        # handoff. Use replicaCounts, NOT vllm.enabled=false (which drops the
        # Deployment and leaves nothing for `kubectl scale` to bring up).
        cmd += ["--set", "replicaCounts.inference=0", "--set", "replicaCounts.training=1"]
    return cmd


def k8s_delete_namespace_cmd(namespace: str) -> list[str]:
    return ["kubectl", "delete", "namespace", namespace, "--ignore-not-found", "--wait"]


def k8s_port_forward_cmd(target_cfg: dict, namespace: str, port: int = 8000) -> list[str]:
    return ["kubectl", "port-forward", "-n", namespace,
            f"svc/{target_cfg['api_service']}", f"{port}:{port}"]


def k8s_exec_checkpoint_cmd(target_cfg: dict, namespace: str, script: str) -> list[str]:
    return ["kubectl", "exec", "-n", namespace,
            f"statefulset/{target_cfg['megatron_sts']}", "--", "python3", "-c", script]


def k8s_get_pods_cmd(namespace: str, selector: str | None = None) -> list[str]:
    cmd = ["kubectl", "get", "pods", "-n", namespace]
    if selector:
        cmd += ["-l", selector]
    return cmd + ["-o", "json"]


def k8s_scale_cmd(kind_name: str, replicas: int, namespace: str) -> list[str]:
    """`kubectl scale <kind/name> --replicas=<n> -n <ns>`. kind_name is e.g.
    "statefulset/scalarlm-megatron" or "deployment/scalarlm-vllm"."""
    return ["kubectl", "scale", kind_name, f"--replicas={replicas}", "-n", namespace]


# --- k8s side-effecting wrappers ---

def kubectl_get_pods(namespace: str, selector: str | None = None, timeout: float = 15) -> list[dict]:
    """Return the namespace's pod objects (`.items`), optionally filtered by a
    label selector, or [] if the call fails."""
    try:
        result = subprocess.run(k8s_get_pods_cmd(namespace, selector), cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    try:
        return json.loads(result.stdout).get("items", [])
    except json.JSONDecodeError:
        return []


def kubectl_scale(kind_name: str, replicas: int, namespace: str, log, timeout: float = 60) -> bool:
    """Scale a Deployment/StatefulSet. True iff kubectl exits 0."""
    try:
        subprocess.run(k8s_scale_cmd(kind_name, replicas, namespace), cwd=REPO_ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def helm_install(target_cfg: dict, namespace: str, model_id: str, log, timeout: float = 600) -> bool:
    """`helm upgrade --install` the per-model release. True iff helm exits 0."""
    try:
        subprocess.run(k8s_helm_install_cmd(target_cfg, namespace, model_id), cwd=REPO_ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def wait_for_pods_ready(namespace: str, gpu_wait_timeout: float, poll: float = 10.0) -> str:
    """Block until pods are "ready" or "failed", else "timeout" at gpu_wait_timeout.
    "pending"/Unschedulable keeps waiting (GPU may be busy); see classify_pod_status."""
    deadline = time.time() + gpu_wait_timeout
    while time.time() < deadline:
        status = classify_pod_status(kubectl_get_pods(namespace))
        if status in ("ready", "failed"):
            return status
        time.sleep(poll)
    return "timeout"


# Megatron pods carry this component label (chart _helpers.tpl megatronlabels);
# used to wait for the megatron pod to fully delete during the phase-2 handoff.
MEGATRON_POD_SELECTOR = "app.kubernetes.io/component=megatron"


def wait_for_pods_gone(namespace: str, selector: str, gpu_wait_timeout: float,
                       poll: float = 5.0) -> str:
    """Block until no pod matching `selector` exists (its nvidia.com/gpu request is
    released only on full deletion, not while Terminating), else "timeout" at
    gpu_wait_timeout. Phase-2 handoff: megatron must be gone before vLLM claims the
    card. Polling beats `kubectl wait --for=delete`, which errors when zero pods
    already match."""
    deadline = time.time() + gpu_wait_timeout
    while time.time() < deadline:
        if not kubectl_get_pods(namespace, selector):
            return "gone"
        time.sleep(poll)
    return "timeout"


def start_port_forward(target_cfg: dict, namespace: str, log) -> subprocess.Popen:
    """Spawn `kubectl port-forward svc/<api> 8000:8000` in its own process group."""
    return subprocess.Popen(k8s_port_forward_cmd(target_cfg, namespace), cwd=REPO_ROOT,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def delete_namespace(namespace: str, log, timeout: float = 300) -> None:
    """`kubectl delete namespace --ignore-not-found --wait` (also GCs the
    resource-policy:keep PVCs). Used for pre-clean and teardown; never raises."""
    try:
        subprocess.run(k8s_delete_namespace_cmd(namespace), cwd=REPO_ROOT,
                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def read_checkpoint_keys_k8s(target_cfg: dict, namespace: str, job_hash: str,
                             timeout: float = 60) -> list[str] | None:
    """k8s path: read checkpoint keys via `kubectl exec statefulset/<megatron>`."""
    cmd = k8s_exec_checkpoint_cmd(target_cfg, namespace, _checkpoint_keys_script(job_hash))
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return _parse_checkpoint_keys_output(result.stdout)


def _parse_gpu_free_gb(stdout: str) -> list[float]:
    """Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
    (MiB per GPU) into GiB. Non-numeric lines are skipped: unified-memory parts
    (GB10/DGX Spark) report '[N/A]' for memory.free, which must not crash int()."""
    out: list[float] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line) / 1024)
        except ValueError:
            continue  # '[N/A]' on unified-memory GPUs
    return out


def probe_gpu_free_gb() -> list[float]:
    """Free VRAM (GiB) per visible GPU that reports a numeric figure, via
    nvidia-smi. [] if nvidia-smi is unavailable or no GPU reports numeric free
    memory (e.g. a unified-memory GB10 reporting '[N/A]' — use gpu_count to tell
    that apart from a box with no GPU).

    Used both for the GPU-target LoRA VRAM gate (cuda-docker) and to detect VRAM
    reclamation after teardown_stack — via nvidia-smi (no torch on the host)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return _parse_gpu_free_gb(result.stdout)


def gpu_count() -> int:
    """Number of GPUs nvidia-smi enumerates, regardless of whether they report a
    numeric memory figure. Lets the caller distinguish a unified-memory GPU
    (present, but memory.free '[N/A]' — GB10/DGX Spark) from a box with no GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0
    return len([line for line in result.stdout.strip().splitlines() if line.strip()])


def _vram_for_gate(free_gb: list[float], n_gpus: int) -> list[float] | None:
    """Decide the VRAM-gate input from probed free memory and GPU count. Returns
    None — "VRAM check not applicable", same as the k8s path — when a GPU is
    present but reports no numeric free figure (unified memory, GB10/DGX Spark ->
    '[N/A]'): there is no discrete VRAM pool to gate on, so never SKIP for it.
    [] (genuinely no GPU visible) and a populated list pass straight through."""
    if not free_gb and n_gpus > 0:
        return None
    return free_gb


def start_restart(target_cfg: dict, model_id: str, log) -> subprocess.Popen:
    """Launch `SCALARLM_MODEL=<model> ./scalarlm up <target>` in its own process
    group, non-blocking."""
    cmd = target_cfg["restart_cmd"].format(model=model_id)
    return subprocess.Popen(cmd, shell=True, cwd=REPO_ROOT, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)


def teardown_stack(proc: subprocess.Popen, settle_timeout: float = 60.0) -> None:
    """SIGKILL the restart process's whole process group, then wait for VRAM to
    stop climbing. NOTE: `./scalarlm up` runs `docker compose up` in the
    foreground, so SIGKILL stops the compose CLI but leaves the container (and its
    GPU) running; the next run's --force-recreate reclaims it, or `docker compose
    down <service>` does."""
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


def _checkpoint_keys_script(job_hash: str) -> str:
    """The in-container Python that prints the latest checkpoint's
    model_state_dict keys as JSON. Identical for Compose and k8s."""
    return (
        "import glob, re, json, torch\n"
        f"paths = glob.glob('/app/cray/jobs/{job_hash}/checkpoint_*.pt')\n"
        "def step(p):\n"
        "    m = re.search(r'checkpoint_(\\d+)\\.pt', p)\n"
        "    return int(m.group(1)) if m else -1\n"
        "paths.sort(key=step)\n"
        "ckpt = torch.load(paths[-1], map_location='cpu', weights_only=False) if paths else None\n"
        "print(json.dumps(list(ckpt['model_state_dict'].keys()) if ckpt else None))\n"
    )


def _parse_checkpoint_keys_output(stdout: str) -> list[str] | None:
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def read_checkpoint_keys(compose_service: str, job_hash: str, timeout: float = 60) -> list[str] | None:
    """Read the latest checkpoint's model_state_dict keys via `docker compose exec`
    — the one step that needs in-container torch. Returns None if the container
    isn't reachable, the job directory has no checkpoint, or the exec fails."""
    cmd = ["docker", "compose", "-f", "docker-compose.yaml", "exec", "-T", compose_service,
           "python3", "-c", _checkpoint_keys_script(job_hash)]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                 timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return _parse_checkpoint_keys_output(result.stdout)


def run_model_k8s_phased(target_cfg: dict, model_id: str, train_args: dict, dataset: list[dict],
                         golden_prompt: str, expected_output: str, args, res, log) -> "Result":
    """Phase-scaled k8s closed loop (peak GPU = 1). Re-sequences run_model:
    install vLLM=0/megatron=1 -> train (phase 1) -> hand the GPU to vLLM (phase 2)
    -> baseline + hot-load + memorization check. See the 2026-06-16 amendment in
    docs/adr/0003-finetune-sweep-restart-per-model.md."""
    namespace = k8s_namespace(target_cfg["namespace_prefix"], model_id)
    gpu_wait = target_cfg.get("gpu_wait_timeout", args.gpu_wait_timeout)
    pf_proc = None
    restart_start = time.time()
    try:
        # --- Phase 0: install; megatron holds the single GPU, vLLM is off. ---
        delete_namespace(namespace, log)  # idempotent pre-clean
        if not helm_install(target_cfg, namespace, model_id, log, args.restart_timeout):
            res.restart_seconds = round(time.time() - restart_start, 1)
            res.outcome, res.detail = RESTART_FAILED, "helm upgrade --install failed"
            return res
        if wait_for_pods_ready(namespace, gpu_wait) != "ready":
            res.restart_seconds = round(time.time() - restart_start, 1)
            res.outcome, res.detail = RESTART_FAILED, f"megatron not schedulable within {gpu_wait}s"
            return res
        pf_proc = start_port_forward(target_cfg, namespace, log)
        # Gate on slurm-registered megatron (NOT health.all -- vLLM is down).
        if not wait_for_all_up(args.api_url, pf_proc, args.restart_timeout, health_key="megatron"):
            res.restart_seconds = round(time.time() - restart_start, 1)
            res.outcome, res.detail = RESTART_FAILED, "megatron health not up after port-forward"
            return res
        res.restart_seconds = round(time.time() - restart_start, 1)

        # --- Phase 1: train (vLLM absent; baseline deferred to phase 2). ---
        train_start = time.time()
        try:
            job_status = submit_train(args.api_url, dataset, train_args)
        except Exception as e:
            res.outcome, res.detail = TRAIN_FAILED, f"submit_train failed: {e}"
            return res
        job_hash = job_status["job_directory"].rstrip("/").split("/")[-1]
        train_status = poll_training(args.api_url, job_hash, args.train_timeout)
        res.train_seconds = round(time.time() - train_start, 1)

        checkpoint_keys: list[str] | None = None
        if train_status == "COMPLETED":
            checkpoint_keys = read_checkpoint_keys_k8s(target_cfg, namespace, job_hash)

        # --- Phase 2: hand the GPU from megatron to vLLM. ---
        if not kubectl_scale(f"statefulset/{target_cfg['megatron_sts']}", 0, namespace, log):
            res.outcome, res.detail = RESTART_FAILED, "kubectl scale megatron->0 failed"
            return res
        if wait_for_pods_gone(namespace, MEGATRON_POD_SELECTOR, gpu_wait) != "gone":
            res.outcome, res.detail = RESTART_FAILED, f"megatron did not release GPU within {gpu_wait}s"
            return res
        if not kubectl_scale(f"deployment/{target_cfg['vllm_deploy']}", 1, namespace, log):
            res.outcome, res.detail = RESTART_FAILED, "kubectl scale vllm->1 failed"
            return res
        if wait_for_pods_ready(namespace, gpu_wait) != "ready":
            res.outcome, res.detail = RESTART_FAILED, f"vLLM not schedulable within {gpu_wait}s"
            return res
        # Gate on vLLM serving (NOT health.all -- megatron is down now).
        if not wait_for_all_up(args.api_url, pf_proc, args.serve_timeout, health_key="vllm"):
            res.outcome, res.detail = RESTART_FAILED, "vllm health not up after scale-up"
            return res

        # Baseline (control on the base model) -- moved here: vLLM is now up.
        # Retry across the cold-start window (first generate after scale-up).
        baseline_text, base_err = generate_with_retry(
            args.api_url, golden_prompt, model_id, args.max_tokens,
            args.serve_timeout, args.generate_timeout)
        if baseline_text is None:
            res.outcome, res.detail = RESTART_FAILED, f"baseline generate failed: {base_err}"
            return res
        res.baseline_sample = baseline_text[:200]
        if expected_output in baseline_text:
            res.detail = "expected_output already present in baseline output"

        # Hot-load + classify (shared with the 2-GPU path); pass the full
        # baseline for the no-op discriminator.
        serve_check_and_classify(args.api_url, golden_prompt, expected_output, job_hash,
                                 train_status, checkpoint_keys, args, res,
                                 baseline_full=baseline_text)
        return res
    finally:
        if pf_proc is not None:
            try:
                os.killpg(os.getpgid(pf_proc.pid), signal.SIGKILL)
            except OSError:
                pass
            pf_proc.wait()
        delete_namespace(namespace, log)


def build_train_args(manifest: dict, target_cfg: dict, model: dict,
                     sweep_run_id: str) -> dict:
    """Assemble a model's training params, in increasing precedence:
    manifest train_args_defaults < target train_args_overrides < the model's own
    `train_args` < the fixed llm_name/sweep_run_id. The per-model layer lets one
    model opt out of a global default — e.g. `train_args: {dtype: bfloat16}` for a
    flagship model that would OOM under the default float32 in the Spark's shared
    128GiB unified pool (fp32 32B ≈ 128GiB vs bf16 ≈ 64GiB)."""
    return {
        "llm_name": model["id"],
        **manifest["train_args_defaults"],
        **target_cfg.get("train_args_overrides", {}),
        **model.get("train_args", {}),
        "sweep_run_id": sweep_run_id,
    }


def run_model(manifest: dict, target: str, model: dict, args, results_dir: Path) -> Result:
    model_id = model["id"]
    res = Result(model=model_id, target=target)

    target_cfg = manifest["targets"][target]
    is_k8s = is_k8s_target(target_cfg)
    # k8s: the scheduler arbitrates GPUs and there is no host nvidia-smi, so skip
    # the runtime VRAM probe (free_gb=None tells gate_model the VRAM check is
    # not applicable; only its static checks apply). For a Compose GPU target,
    # _vram_for_gate also yields None on a unified-memory GPU (GB10/DGX Spark,
    # whose nvidia-smi memory.free is '[N/A]') — no discrete VRAM pool to gate on.
    if is_k8s:
        free_gb = None
    elif target_requests_gpu(target_cfg):
        free_gb = _vram_for_gate(probe_gpu_free_gb(), gpu_count())
    else:
        free_gb = []
    ok, reason = gate_model(model, target_cfg, free_gb)
    if not ok:
        res.outcome, res.detail = SKIPPED, reason
        return res

    train_args = build_train_args(manifest, target_cfg, model, args.sweep_run_id)
    dataset = build_dataset(manifest["dataset"])
    golden_prompt = manifest["golden_prompt"]
    expected_output = manifest["expected_output"]

    log_path = results_dir / f"{model_id.replace('/', '_')}.{target}.restart.log"
    if is_k8s and target_cfg.get("phase_scaled"):
        with open(log_path, "w") as log:
            return run_model_k8s_phased(target_cfg, model_id, train_args, dataset,
                                        golden_prompt, expected_output, args, res, log)
    namespace = pf_proc = proc = None
    restart_start = time.time()
    with open(log_path, "w") as log:
        try:
            if is_k8s:
                namespace = k8s_namespace(target_cfg["namespace_prefix"], model_id)
                delete_namespace(namespace, log)  # idempotent pre-clean
                if not helm_install(target_cfg, namespace, model_id, log, args.restart_timeout):
                    res.restart_seconds = round(time.time() - restart_start, 1)
                    res.outcome, res.detail = RESTART_FAILED, "helm upgrade --install failed"
                    return res
                gpu_wait = target_cfg.get("gpu_wait_timeout", args.gpu_wait_timeout)
                pod_status = wait_for_pods_ready(namespace, gpu_wait)
                if pod_status != "ready":
                    res.restart_seconds = round(time.time() - restart_start, 1)
                    res.outcome = RESTART_FAILED
                    res.detail = ("pods crashed / bad image" if pod_status == "failed"
                                  else f"pods not schedulable within {gpu_wait}s")
                    return res
                pf_proc = start_port_forward(target_cfg, namespace, log)
                if not wait_for_all_up(args.api_url, pf_proc, args.restart_timeout):
                    res.restart_seconds = round(time.time() - restart_start, 1)
                    res.outcome, res.detail = RESTART_FAILED, "api /v1/health not up after port-forward"
                    return res
                res.restart_seconds = round(time.time() - restart_start, 1)
            else:
                proc = start_restart(target_cfg, model_id, log)
                # Model-aware wait: a stale (previous-model) container left
                # running by teardown_stack is still health.all=="up", so we
                # also gate on /v1/models serving model_id before proceeding.
                # Pass compose_service so a crash-looping vLLM fails fast (a model
                # that crashes the engine would otherwise hang the whole
                # restart_timeout while restart:unless-stopped re-inits it).
                status = wait_for_model_served(args.api_url, model_id, proc,
                                               args.restart_timeout,
                                               compose_service=target_cfg["compose_service"])
                res.restart_seconds = round(time.time() - restart_start, 1)
                if status != "served":
                    res.outcome = RESTART_FAILED
                    res.detail = {
                        "crashed": f"vLLM crash-looped serving {model_id} "
                                   f"(container RestartCount climbed)",
                        "proc_exited": f"`./scalarlm up` exited before serving {model_id}",
                        "timeout": f"stack did not serve {model_id} "
                                   f"(health.all/up + /v1/models) within {args.restart_timeout}s",
                    }.get(status, f"stack did not serve {model_id}")
                    return res

            # Retry across the cold-start window: health.all/up + /v1/models can
            # be satisfied before the inference worker finishes warmup, so the
            # first /v1/generate may block past one request_timeout.
            baseline_text, base_err = generate_with_retry(
                args.api_url, golden_prompt, model_id, args.max_tokens,
                args.serve_timeout, args.generate_timeout)
            if baseline_text is None:
                res.outcome = RESTART_FAILED
                res.detail = f"baseline generate failed: {base_err}"
                return res
            res.baseline_sample = baseline_text[:200]
            if expected_output in baseline_text:
                res.detail = "expected_output already present in baseline output"

            train_start = time.time()
            try:
                job_status = submit_train(args.api_url, dataset, train_args)
            except Exception as e:
                res.outcome = TRAIN_FAILED
                res.detail = f"submit_train failed: {e}"
                return res
            job_hash = job_status["job_directory"].rstrip("/").split("/")[-1]

            train_status = poll_training(args.api_url, job_hash, args.train_timeout)
            res.train_seconds = round(time.time() - train_start, 1)

            checkpoint_keys: list[str] | None = None
            if train_status == "COMPLETED":
                if is_k8s:
                    checkpoint_keys = read_checkpoint_keys_k8s(target_cfg, namespace, job_hash)
                else:
                    checkpoint_keys = read_checkpoint_keys(target_cfg["compose_service"], job_hash)
            # Hot-load + classify (shared with the phased k8s path). The full
            # baseline string drives the in-sweep no-op discriminator: a served
            # adapter whose output is byte-identical to baseline under greedy
            # decoding never applied the LoRA -> ADAPTER_NO_OP.
            serve_check_and_classify(args.api_url, golden_prompt, expected_output, job_hash,
                                     train_status, checkpoint_keys, args, res,
                                     baseline_full=baseline_text)

            # If ADAPTER_NO_OP, scrape logs for the a-priori known warning.
            if res.outcome == ADAPTER_NO_OP:
                try:
                    logs = subprocess.run(
                        ["docker", "compose", "-f", "docker-compose.yaml", "logs", "--no-color", target_cfg["compose_service"]],
                        cwd=REPO_ROOT, capture_output=True, text=True
                    ).stdout
                    match = [l for l in logs.splitlines() if "NONE of its" in l and "match the base model" in l]
                    if match:
                        res.hint = match[-1]
                except Exception:
                    pass

            return res
        finally:
            if is_k8s:
                if pf_proc is not None:
                    try:
                        os.killpg(os.getpgid(pf_proc.pid), signal.SIGKILL)
                    except OSError:
                        pass
                    pf_proc.wait()
                if namespace is not None:
                    delete_namespace(namespace, log)
            elif proc is not None:
                teardown_stack(proc)


def write_reports(results: list[Result], target: str, results_dir: Path) -> tuple[Path, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"finetune.{target}.{stamp}.json"
    md_path = results_dir / f"finetune.{target}.{stamp}.md"

    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    lines = [f"# Fine-tune sweep — `{target}` — {stamp}", "",
             "| Model | Outcome | Detail | Hint | Baseline sample | Adapter sample | restart_s | train_s | serve_s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        baseline = r.baseline_sample.replace("\n", " ").replace("|", "\\|")[:60]
        adapter = r.adapter_sample.replace("\n", " ").replace("|", "\\|")[:60]
        detail = r.detail.replace("|", "\\|")
        hint = r.hint.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{r.model}` | {r.outcome} | {detail} | {hint} | {baseline} | {adapter} "
                     f"| {r.restart_seconds} | {r.train_seconds} | {r.serve_seconds} |")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune sweep (tier d, LoRA only).")
    ap.add_argument("--target", required=True,
                    help="manifest target key (e.g. cpu, cuda-docker, cuda-k8s); "
                         "validated against the manifest after load")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--models", nargs="*", help="optional subset of model IDs to run")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--restart-timeout", type=int, default=3000,
                    help="health-wait cap (s). Default 3000 (50 min) covers a cold "
                         "`./scalarlm up nvidia --build` (vLLM compiles from source) "
                         "on a fresh GPU box; only a ceiling, so happy paths are "
                         "unaffected. Subsequent --force-recreate builds are cache hits.")
    ap.add_argument("--train-timeout", type=int, default=600)
    ap.add_argument("--serve-timeout", type=int, default=300)
    ap.add_argument("--generate-timeout", type=int, default=300,
                    help="per-call /v1/generate socket timeout (s). cray serves "
                         "generate synchronously, so this must exceed the first "
                         "post-restart inference latency (engine init/CUDA-graph "
                         "capture + worker warmup, ~146s on the GB10/DGX Spark). "
                         "Baseline + adapter calls retry within --serve-timeout.")
    ap.add_argument("--gpu-wait-timeout", type=int, default=7200,
                    help="k8s only: seconds to block while pods are Unschedulable "
                         "(GPUs busy) before giving up with RESTART_FAILED")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the offline LoRA no-op preflight; run every model "
                         "even if it is predicted to serve base output")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.target not in manifest["targets"]:
        ap.error(f"unknown target {args.target!r}; have {list(manifest['targets'])}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.sweep_run_id = uuid.uuid4().hex

    models = filter_models(manifest["models"], args.models)

    # Offline preflight: predict the silent LoRA no-op before paying for a
    # model's restart + train + serve, and skip those models. Compose-only —
    # k8s targets have no `compose_service`, so they run unfiltered (the
    # in-sweep ADAPTER_NO_OP discriminator is still ground truth there).
    results: list[Result] = []
    compose_service = manifest["targets"][args.target].get("compose_service")
    if not args.no_preflight and compose_service:
        from preflight import run_preflight
        # Preflight runs inside the target's already-built serving image. It is
        # gated on that image existing (run_preflight skips cleanly if not), since
        # `docker compose run` would otherwise attempt a doomed build of a missing
        # image — failing on the BASE_NAME build arg only `./scalarlm up` supplies.
        print(f"\n=== preflight ({len(models)} models, service={compose_service}) ===",
              flush=True)
        pf_results = run_preflight([m["id"] for m in models], compose_service)
        # Log EVERY result, not just the skips: a fail-open (build/introspection
        # error -> run the model) must be loud, otherwise a wholly-broken preflight
        # looks identical to "every model predicted OK".
        for mid in (m["id"] for m in models):
            r = pf_results.get(mid)
            if r is None:
                continue
            if r.error:
                tag = ("gated repo -> SKIP" if is_gated_repo_error(r.error)
                       else "fail-open (will run)")
                print(f"    [preflight] {mid}: ERROR -> {tag}: "
                      f"{r.error[:160]}", flush=True)
            else:
                print(f"    [preflight] {mid}: predicted_ok={r.predicted_ok} "
                      f"overlap={r.n_overlap}/{r.n_total}", flush=True)
        models, skipped = split_by_preflight(models, pf_results, args.target)
        for s in skipped:
            print(f"--> {s.model}: {s.outcome} ({s.hint})", flush=True)
        results.extend(skipped)

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
