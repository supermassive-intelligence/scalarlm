"""
Capacity helpers grounded in what slurmctld actually has registered.

Earlier helpers read `load_all_nodes()` from the shared NFS discovery
directory. That directory accumulates stale JSON files from pods that
have been rescheduled, so it over-reports nodes; and each file holds
the host's raw `torch.cuda.device_count()`, which may be the full
physical GPU count even when only a subset is cgroup-allocated to the
pod. SLURM's view sidesteps both problems: `scontrol show nodes` lists
exactly the slurmd-registered nodes (= ScalarLM megatron pods) and
their `Gres=gpu:N` is the cap that `write_node_config` already applied.

When slurmctld isn't reachable (local dev before the controller boots,
running tests, early API-pod startup), we fall back to "this host" —
one node with whatever CUDA can see from the current container. That
matches the single-container docker-compose setup.
"""

import logging
import re
import subprocess
from typing import List

import torch

logger = logging.getLogger(__name__)

_GPU_RE = re.compile(r"Gres=gpu:(\d+)")
_NODENAME_RE = re.compile(r"^NodeName=(\S+)")


def get_slurm_nodes() -> List[dict]:
    """
    Return a list of {"name": str, "gpus": int} for each node slurmctld
    currently has registered. Empty list when slurmctld is unreachable.
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "nodes"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("scontrol unavailable: %s", e)
        return []

    if result.returncode != 0:
        logger.debug(
            "scontrol show nodes failed (rc=%s): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return []

    nodes: List[dict] = []
    current: dict | None = None
    for line in result.stdout.splitlines():
        m = _NODENAME_RE.match(line.strip())
        if m:
            if current is not None:
                nodes.append(current)
            current = {"name": m.group(1), "gpus": 0}
        if current is None:
            continue
        gm = _GPU_RE.search(line)
        if gm:
            current["gpus"] = int(gm.group(1))
    if current is not None:
        nodes.append(current)

    return nodes


def count_slurm_nodes() -> int:
    """
    Number of slurmd-registered nodes — one per scalarlm-megatron pod.
    Falls back to 1 (the current host) when slurmctld isn't responding.
    """
    nodes = get_slurm_nodes()
    if nodes:
        return len(nodes)
    return 1


def count_slurm_gpus() -> int:
    """
    Total GPUs available to scalarlm for training — the sum of `Gres=gpu:N`
    across slurmd-registered nodes. Falls back to `torch.cuda.device_count()`
    for local dev / early-startup cases where SLURM hasn't registered
    anything yet.
    """
    nodes = get_slurm_nodes()
    if nodes:
        return sum(n["gpus"] for n in nodes)

    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception as e:
        logger.debug("torch.cuda.device_count() failed: %s", e)
    return 0
