import torch
from cray_megatron.megatron.distribution.fsdp import SimpleFSDP

from gpu_aware_mpi import get_size, get_rank

import socket
import os
import json
import logging

logger = logging.getLogger(__name__)


def load_distribution_strategy():
    device = get_device()

    strategy = {
        "device": device,
    }

    if get_size() > 1:
        strategy["strategy"] = SimpleFSDP

    return strategy


def get_device():
    if torch.cuda.is_available():

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

    hosts_on_this_machine = get_hosts_on_machine(machine_id)

    gpu_index = 0

    for i, host in enumerate(hosts_on_this_machine):
        if host == my_hostname:
            gpu_index = i % gpu_count
            break

    logger.info(
        f"Rank {rank} on host {my_hostname} with machine ID {machine_id} assigned GPU {gpu_index}"
    )
    return gpu_index


def get_machine_id():
    machine_id = None
    try:
        with open("/etc/machine-id", "r") as f:
            machine_id = f.read().strip()
    except Exception as e:
        logger.error(f"Error reading machine ID: {e}")
    return machine_id


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
