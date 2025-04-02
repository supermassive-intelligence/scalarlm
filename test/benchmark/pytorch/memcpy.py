import torch
import json
import time

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

# List of memcpy sizes, in bytes, should be multiples of the page size
# Go up to the tensor size used in Llama 3 (4096 * 128256 * 4) = 2_101_346_304
memcpy_sizes = [ i**2 for i in range(12, 64) if i**2 <= 2_101_346_304 ]


def main():
    benchmark_memcpy()


def benchmark_memcpy():
    logger.info("Running memcpy benchmark")
    results = run_memcpy_benchmark()

    save_results(results)


def run_memcpy_benchmark():

    warmup()

    results = {}

    for size in tqdm(memcpy_sizes):
        results[size] = run_memcpy(size)

    return results


def warmup():
    run_memcpy(4096)


def run_memcpy(size):
    a = torch.randn(size // 4) # size is in bytes, so divide by 4 to get number of floats
    b = torch.randn(size // 4)

    # copy for at least 1 second
    barrier()

    start = get_event()
    end = get_event()

    start_time = time.time()

    start.record()
    iterations = 0
    while time.time() - start_time < 1:
        b.copy_(a)
        iterations += 1
    end.record()

    barrier()
    time = start.elapsed_time(end) * 1e-3 / iterations

    return {
        "operational_intensity": 1 / 4,  # 1 FLOP per 4 bytes
        "flop/s": size / 4 / time,
        "bytes": size,
        "time": time,
        "iterations": iterations,
        "bandwidth": size / time,
        "GB/s": size / time / 1e9,
    }


class CPUEvent:
    def __init__(self):
        self.time = 0

    def record(self):
        self.time = time.time()

    def elapsed_time(self, other):
        return (other.time - self.time) * 1000


def get_event():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    else:
        return CPUEvent()


def barrier():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        pass


def save_results(results):
    # Save results to a json file
    path = "/app/cray/data/benchmark_memcpy.json"

    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
