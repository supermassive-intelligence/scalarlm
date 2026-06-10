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
