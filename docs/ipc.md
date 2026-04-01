# GPU P2P Transfer Learnings — Kubernetes Shared-Node Setup

## Hardware Environment

- **Node:** `blackwell-maxq-0` — single physical machine
- **GPUs:** 2× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (102 GB each, PCIe, not SXM/NVLink)
- **HCA:** ConnectX-6 (MT4123), single QSFP56 port, HDR 200Gb/s, IB mode, `phys_state: Disabled` (no cable)
- **Pods:** `scalarlm-megatron-0` and `scalarlm-megatron-1` — one GPU each, scheduled on the same node
- **Key pod spec:** `hostIPC: true`, `hostPID: true`, `SYS_PTRACE` capability, shared PVC at `/app/cray/jobs`

---

## What Was Tried and What Happened

### 1. CUDA IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)
**Result: Fails across different physical GPUs.**
CUDA IPC handles encode the physical device. Opening a handle from GPU-0 in a process bound to GPU-1 raises `cudaErrorInvalidValue` unless NVLink P2P is available. There is no workaround without NVLink or a shared physical GPU (time-slicing/MPS). This approach is only valid when both processes share the same physical device.

### 2. `/dev/shm` with `cudaHostRegister` — Manual mmap
**Result: ✓ 57 GB/s — the winning approach.**
Because `hostIPC: true` puts both pods in the same host IPC namespace, `/dev/shm` is literally the same RAM-backed tmpfs. The data path is:

```
GPU-0 ──PCIe DMA──▶ pinned /dev/shm pages ──PCIe DMA──▶ GPU-1
```

Key implementation details:
- `mmap` the same `/dev/shm` file in both pods
- Call `cudaHostRegister()` on the mmap'd region to make it a CUDA DMA target
- Use `torch.frombuffer()` to create a zero-copy CPU tensor over the region
- Use `tensor.copy_()` to trigger async DMA; sync with `torch.cuda.synchronize()`
- Use a shared PVC (`/app/cray/jobs`) as a side-channel for flag files and the IPC handle

**57 GB/s is at the PCIe 5.0 ceiling for this hardware.** RDMA loopback over the same PCIe bus would not exceed this.

### 3. MPI `Send/Recv` with UCX — First Attempt
**Result: 2.5 GB/s (TCP path).**
UCX determines locality by hostname comparison. `scalarlm-megatron-0 ≠ scalarlm-megatron-1` → UCX treats them as remote nodes → all data routes over TCP despite being on the same physical machine. Setting `UCX_TLS=posix,cma,sm,self` (without `tcp`) caused `MPI_Init` to segfault because UCC's collective init also requires a routable transport for address exchange.

**Fix for init:** Add `tcp` to `UCX_TLS` and set `OMPI_MCA_coll_ucc_enable=0` (UCC segfaults trying to use IB for collective init when IB port is down).

### 4. MPI `Send/Recv` with ob1 + vader BTL (CMA)
**Result: 3.86 GB/s — correct transport, wrong copy count.**
`OMPI_MCA_pml=ob1` + `OMPI_MCA_btl=vader,self,tcp` with `btl_vader_single_copy_mechanism=cma` correctly routes data through `process_vm_readv` (CMA) instead of TCP. However, `Send/Recv` fundamentally requires 3 copies: GPU→cpu_0, CMA cpu_0→cpu_1, cpu_1→GPU. The middle CMA copy is unavoidable with `Send/Recv`. Note: `btl=sm` was renamed to `btl=vader` in OpenMPI 3.0.

### 5. `MPI_Win_allocate_shared`
**Result: `MPI_ERR_RMA_SHARED` — hostname check blocks it.**
`MPI_Win_allocate_shared` is spec'd as intra-node only, and "node" means same hostname. Different pod names → MPI refuses regardless of physical co-location or `hostIPC: true`. There is no MPI abstraction that escapes this hostname check.

### 6. MPI-coordinated `/dev/shm` (Final Architecture)
**Result: ✓ 57 GB/s — matches manual version.**
The correct architecture separates concerns: MPI handles all synchronization (barriers, timing coordination, result gathering via `comm.gather()`), while the actual data transfer uses the manual mmap path. This is not a workaround — it is the correct design for this topology. CUDA events provide nanosecond-precision timing independent of MPI overhead.

---

## InfiniBand Status

The HCA port is in `InfiniBand` mode (confirmed via sysfs `link_layer`) but `phys_state: 3 (Disabled)` — no physical signal. The SM (`opensm.service`) is running as a systemd service and started correctly, but has nothing to manage. MFT tools (`mlxconfig`, `mlxlink`, `mst`) are not installed; the Mellanox download URLs have moved from `mellanox.com` to NVIDIA's CDN and specific paths have changed.

To enable RDMA loopback, a **QSFP56 passive loopback plug** is required (~$15-20 from FS.com or Amazon). This would transition the port from `Disabled → Polling → Active` and enable `ibv_rc_pingpong` loopback. However, RDMA bandwidth would not exceed the current 57 GB/s since both transfer paths are bottlenecked by the same PCIe bus.

**`rdma_rxe` (Soft-RoCE)** is an alternative — a kernel module (available since kernel 4.8) that implements RoCE v2 over any Ethernet interface with no hardware required. Useful for developing RDMA code but bandwidth is limited by the pod network interface.

---

## Key Principles Established

**PCIe ceiling:** For two PCIe GPUs on the same node (non-NVLink), ~57 GB/s is the practical bandwidth ceiling. All transfer paths — `/dev/shm`, RDMA loopback, CMA — traverse the same PCIe fabric.

**`hostIPC: true` is the enabler:** This single pod spec flag makes `/dev/shm` a shared physical resource between pods on the same node, enabling the zero-intermediate-copy data path.

**MPI hostname locality:** UCX and `MPI_Win_allocate_shared` both use hostname equality to determine if ranks are co-located. Different pod names defeat all MPI shared-memory optimizations regardless of physical placement. The only reliable workaround is to bypass MPI's data plane entirely and use mmap directly.

**Copy count is the key metric:** `/dev/shm` = 2 copies (D2H + H2D). `Send/Recv` + CMA = 3 copies (D2H + CMA + H2D). `Win_allocate_shared` would be 2 copies but is blocked by hostname check. Manual mmap is the only achievable 2-copy path in this topology.

---

## Recommended Architecture for This Setup

```
Control plane : MPI (Barrier, gather, timing)
Data plane    : mmap /dev/shm + cudaHostRegister + tensor.copy_()
Scheduling    : Slurm sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:1
MPI flags     : OMPI_MCA_pml=ob1, btl=vader,self,tcp
                OMPI_MCA_coll_ucc_enable=0, OMPI_MCA_coll_hcoll_enable=0
```
