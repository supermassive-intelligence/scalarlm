#!/usr/bin/env python3
"""
test_gpu_aware_mpi.py
=====================
Stress-tests every function exposed by the gpu_aware_mpi extension:
  send / recv
  allreduce
  allgather
  reduce_scatter
  alltoall
  barrier

Each test runs for --iterations rounds across several tensor sizes and
dtypes to catch races, size-mismatch bugs, and numerical errors.

Usage (via mpirun — handled automatically by docker-entrypoint.sh):
  mpirun -np 4 python3 test_gpu_aware_mpi.py \
      --cuda-device-ids 0,1,2,3 \
      --iterations 20
"""

import argparse
import sys
import time
from typing import List

import torch
import gpu_aware_mpi as mpi   # the extension under test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cuda-device-ids",
        type=str,
        default="",
        help="Comma-separated CUDA device IDs, one per rank (e.g. '0,1,2,3'). "
             "If empty, uses rank %% torch.cuda.device_count().",
    )
    p.add_argument(
        "--iterations", type=int, default=10,
        help="Number of iterations per test.",
    )
    return p.parse_args()


def rank_device(rank: int, device_ids: List[int]) -> torch.device:
    # The entrypoint sets CUDA_VISIBLE_DEVICES to the requested physical IDs,
    # so CUDA re-indexes them as 0..N-1 inside this process.  Use rank % N
    # to pin each rank to a distinct visible device regardless of the raw IDs.
    n = torch.cuda.device_count()
    if n > 0:
        return torch.device(f"cuda:{rank % n}")
    return torch.device("cpu")


def log(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def allpass(rank: int, name: str, ok: bool):
    status = "PASS" if ok else "FAIL"
    log(rank, f"{name}: {status}")
    return ok


def make_tensor(
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    fill: float,
) -> torch.Tensor:
    return torch.full((size,), fill, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Test: send / recv  (ring pattern: rank → rank+1)
# ---------------------------------------------------------------------------

def test_send_recv(
    rank: int,
    world: int,
    device: torch.device,
    iterations: int,
    dtype: torch.dtype,
    size: int,
) -> bool:
    name = f"send_recv  dtype={dtype} size={size}"
    ok = True

    src = (rank - 1) % world
    dst = (rank + 1) % world

    for it in range(iterations):
        send_val = float(rank * 1000 + it)
        send_buf = make_tensor(size, dtype, device, send_val)
        recv_buf = torch.zeros(size, dtype=dtype, device=device)

        # Even ranks send first; odd ranks recv first — avoids deadlock.
        if rank % 2 == 0:
            mpi.send(send_buf, dst)
            mpi.recv(recv_buf, src)
        else:
            mpi.recv(recv_buf, src)
            mpi.send(send_buf, dst)

        expected_val = float(src * 1000 + it)
        expected = make_tensor(size, dtype, device, expected_val)

        if not torch.allclose(recv_buf.float(), expected.float(), atol=1e-3):
            log(rank, f"{name} iter={it}: MISMATCH "
                f"got {recv_buf[0].item():.1f} expected {expected_val:.1f}")
            ok = False

    mpi.barrier()
    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Test: allreduce  (sum of ranks 0..world-1 = world*(world-1)/2)
# ---------------------------------------------------------------------------

def test_allreduce(
    rank: int,
    world: int,
    device: torch.device,
    iterations: int,
    dtype: torch.dtype,
    size: int,
) -> bool:
    name = f"allreduce  dtype={dtype} size={size}"
    ok = True
    expected_sum = float(world * (world - 1) // 2)

    for it in range(iterations):
        buf = make_tensor(size, dtype, device, float(rank))
        mpi.allreduce(buf)

        expected = make_tensor(size, dtype, device, expected_sum)
        if not torch.allclose(buf.float(), expected.float(), atol=1e-2):
            log(rank, f"{name} iter={it}: MISMATCH "
                f"got {buf[0].item()} expected {expected_sum}")
            ok = False

    mpi.barrier()
    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Test: allgather
# Each rank contributes [rank, rank, ...] (size elements).
# Result: [0,0,...,1,1,...,2,2,...] shape=(world*size,)
# ---------------------------------------------------------------------------

def test_allgather(
    rank: int,
    world: int,
    device: torch.device,
    iterations: int,
    dtype: torch.dtype,
    size: int,
) -> bool:
    name = f"allgather  dtype={dtype} size={size}"
    ok = True

    for it in range(iterations):
        shard = make_tensor(size, dtype, device, float(rank + it))
        gathered = torch.zeros(world * size, dtype=dtype, device=device)
        mpi.allgather(shard, gathered)

        for r in range(world):
            expected_val = float(r + it)
            chunk = gathered[r * size : (r + 1) * size]
            expected = make_tensor(size, dtype, device, expected_val)
            if not torch.allclose(chunk.float(), expected.float(), atol=1e-3):
                log(rank, f"{name} iter={it} chunk[{r}]: MISMATCH "
                    f"got {chunk[0].item():.1f} expected {expected_val:.1f}")
                ok = False

    mpi.barrier()
    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Test: reduce_scatter
# sendbuf[i*size:(i+1)*size] on all ranks holds value (rank + i).
# After reduce_scatter rank r holds sum over all ranks of (rank + r) for each element
# = sum_{rank=0}^{world-1}(rank + r) = world*(world-1)/2 + world*r
# ---------------------------------------------------------------------------

def test_reduce_scatter(
    rank: int,
    world: int,
    device: torch.device,
    iterations: int,
    dtype: torch.dtype,
    size: int,
) -> bool:
    name = f"reduce_scatter  dtype={dtype} size={size}"
    ok = True

    for it in range(iterations):
        # sendbuf: rank r contributes [r+0, r+0, ..., r+1, r+1, ..., r+world-1, ...]
        # i.e. chunk i = (rank + i + it)
        chunks = [
            make_tensor(size, dtype, device, float(rank + i + it))
            for i in range(world)
        ]
        sendbuf = torch.cat(chunks)
        recvbuf = torch.zeros(size, dtype=dtype, device=device)
        mpi.reduce_scatter(sendbuf, recvbuf)

        # Expected: sum over all ranks of their chunk[rank] = sum(r + rank + it)
        expected_val = float(world * (world - 1) // 2 + world * (rank + it))
        expected = make_tensor(size, dtype, device, expected_val)
        if not torch.allclose(recvbuf.float(), expected.float(), atol=world * 1e-2):
            log(rank, f"{name} iter={it}: MISMATCH "
                f"got {recvbuf[0].item():.1f} expected {expected_val:.1f}")
            ok = False

    mpi.barrier()
    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Test: alltoall
# sendbuf is split into world chunks; chunk i = rank * 100 + i.
# After alltoall rank r holds [0*100+r, 1*100+r, ..., (world-1)*100+r].
# ---------------------------------------------------------------------------

def test_alltoall(
    rank: int,
    world: int,
    device: torch.device,
    iterations: int,
    dtype: torch.dtype,
    size: int,
) -> bool:
    name = f"alltoall  dtype={dtype} size={size}"
    ok = True

    for it in range(iterations):
        chunks = [
            make_tensor(size, dtype, device, float(rank * 100 + i + it))
            for i in range(world)
        ]
        sendbuf = torch.cat(chunks)
        recvbuf = torch.zeros(world * size, dtype=dtype, device=device)
        mpi.alltoall(sendbuf, recvbuf)

        for i in range(world):
            expected_val = float(i * 100 + rank + it)
            chunk = recvbuf[i * size : (i + 1) * size]
            expected = make_tensor(size, dtype, device, expected_val)
            if not torch.allclose(chunk.float(), expected.float(), atol=1e-3):
                log(rank, f"{name} iter={it} chunk[{i}]: MISMATCH "
                    f"got {chunk[0].item():.1f} expected {expected_val:.1f}")
                ok = False

    mpi.barrier()
    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Test: barrier  (timing sanity — all ranks should complete within 5 s)
# ---------------------------------------------------------------------------

def test_barrier(
    rank: int,
    world: int,
    iterations: int,
) -> bool:
    name = "barrier"
    ok = True
    deadline = 5.0   # seconds

    for it in range(iterations):
        t0 = time.monotonic()
        mpi.barrier()
        elapsed = time.monotonic() - t0
        if elapsed > deadline:
            log(rank, f"{name} iter={it}: took {elapsed:.2f}s > {deadline}s deadline")
            ok = False

    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Rapid-fire stress test: allgather with alternating sizes
# This specifically exercises the double-buffer slot-reuse path 
# ---------------------------------------------------------------------------

def test_rapid_allgather_stress(
    rank: int,
    world: int,
    device: torch.device,
    iterations: int,
    dtype: torch.dtype,
) -> bool:
    name = f"rapid_allgather_stress  dtype={dtype}"
    ok = True

    # Alternate between a small and large size to maximise the chance that a
    # second allgather overwrites a slot before the first on_complete fires.
    sizes = [1024, 32 * 1024, 256, 128 * 1024]

    for it in range(iterations):
        size = sizes[it % len(sizes)]
        shard = make_tensor(size, dtype, device, float(rank + it))
        gathered = torch.zeros(world * size, dtype=dtype, device=device)
        mpi.allgather(shard, gathered)

        for r in range(world):
            expected_val = float(r + it)
            chunk = gathered[r * size : (r + 1) * size]
            if not torch.allclose(chunk.float(),
                                  torch.full_like(chunk, expected_val).float(),
                                  atol=1e-3):
                log(rank, f"{name} iter={it} size={size} chunk[{r}]: MISMATCH "
                    f"got {chunk[0].item():.1f} expected {expected_val:.1f}")
                ok = False

    mpi.barrier()
    return allpass(rank, name, ok)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SIZES  = [256, 4096, 65536]    # elements per rank shard


def main():
    args = parse_args()

    device_ids: List[int] = (
        [int(x) for x in args.cuda_device_ids.split(",") if x.strip()]
        if args.cuda_device_ids else []
    )

    rank  = mpi.get_rank()
    world = mpi.get_size()

    device = rank_device(rank, device_ids)
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f" gpu_aware_mpi test suite", flush=True)
        print(f" world={world}  iterations={args.iterations}", flush=True)
        print(f"{'='*60}\n", flush=True)

    log(rank, f"using device {device}")
    mpi.barrier()

    results = []

    # ---- barrier ----
    results.append(test_barrier(rank, world, args.iterations))

    # ---- per-dtype / per-size tests ----
    for dtype in DTYPES:
        for size in SIZES:
            # send/recv only makes sense with at least 2 ranks (always true)
            results.append(test_send_recv(rank, world, device, args.iterations, dtype, size))
            results.append(test_allreduce(rank, world, device, args.iterations, dtype, size))
            results.append(test_allgather(rank, world, device, args.iterations, dtype, size))
            results.append(test_reduce_scatter(rank, world, device, args.iterations, dtype, size))
            results.append(test_alltoall(rank, world, device, args.iterations, dtype, size))

    # ---- race-condition stress test ----
    for dtype in DTYPES:
        results.append(
            test_rapid_allgather_stress(rank, world, device, args.iterations * 3, dtype)
        )

    mpi.barrier()

    passed = sum(results)
    total  = len(results)

    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f" Results: {passed}/{total} tests passed", flush=True)
        if passed == total:
            print(" ALL TESTS PASSED ✓", flush=True)
        else:
            print(f" {total - passed} TEST(S) FAILED ✗", flush=True)
        print(f"{'='*60}\n", flush=True)

    mpi.finalize_mpi()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
