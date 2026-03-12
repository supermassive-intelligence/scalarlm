from cray_megatron.megatron.distribution.fsdp import SimpleFSDP
from cray_megatron.megatron.distribution.ddp import DDP
from cray_megatron.megatron.distribution.no_distribution import NoDistribution
from cray_infra.util.get_job_config import get_job_config

from gpu_aware_mpi import get_size, get_rank

import torch

import socket
import os
import json
import time
import subprocess

import logging

logger = logging.getLogger(__name__)


def load_distribution_strategy():
    device = get_device()

    strategy = {
        "device": device,
    }

    if get_size() > 1:
        distribution_strategy = get_job_config()["distribution_strategy"]

        if distribution_strategy == "ddp":
            logger.info("Using DDP distribution strategy.")
            strategy["strategy"] = DDP
        else:
            logger.warning(
                f"Unknown distribution strategy '{distribution_strategy}' "
                "specified. Defaulting to SimpleFSDP."
            )
            strategy["strategy"] = SimpleFSDP
    else:
        logger.info("Using NoDistribution distribution strategy.")
        strategy["strategy"] = NoDistribution

    return strategy


def get_device():
    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()
        selected_gpu = select_gpu()

        if gpu_count > 1:
            return torch.device(f"cuda:{selected_gpu}")

        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def apply_distribution_strategy(model_info):
    distribution_strategy = load_distribution_strategy()
    model_info["distribution_strategy"] = distribution_strategy
    return model_info


def select_gpu():
    rank = get_rank()
    gpu_count = torch.cuda.device_count()
    machine_id = get_machine_id()
    my_hostname = socket.gethostname()

    attempts = 10

    for attempt in range(attempts):
        try:
            hosts_on_this_machine = get_hosts_on_machine(machine_id)
            break
        except Exception as e:
            logger.error(f"Error getting hosts on machine (attempt {attempt + 1}/{attempts}): {e}")
            hosts_on_this_machine = []
            time.sleep(1)

    gpu_index = 0

    for i, host in enumerate(hosts_on_this_machine):
        if host == my_hostname:
            gpu_index = i % gpu_count
            break

    logger.info(
        f"Rank {rank} on host {my_hostname} out of {len(hosts_on_this_machine)} with "
        f"machine ID {machine_id} assigned GPU {gpu_index} out of {gpu_count} available GPUs."
    )
    return gpu_index


def get_machine_id():
    machine_id = None
    try:
        machine_id = get_board_serial()
    except Exception as e:
        logger.error(f"Error reading machine ID: {e}")
    return machine_id


def get_board_serial() -> str | None:
    result = subprocess.run(
        ["dmidecode", "-s", "baseboard-serial-number"],
        capture_output=True, text=True
    )
    serial = result.stdout.strip()
    return serial if serial else None


def get_hosts_on_machine(machine_id):
    node_path = "/app/cray/nfs/nodes"

    hostnames = []

    for filename in os.listdir(node_path):
        if filename.endswith(".json"):
            with open(os.path.join(node_path, filename), "r") as f:
                node_info = json.load(f)
                if node_info.get("machine_id") == machine_id:
                    hostnames.append(node_info["hostname"])

    return list(sorted(hostnames))
