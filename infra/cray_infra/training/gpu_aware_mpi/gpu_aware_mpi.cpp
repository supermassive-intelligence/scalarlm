#include "common.h"
#include "shm_channel.h"
#include "shm_transport.h"
#include "mpi_transport.h"

#include <iostream>
#include <vector>

// ===================================================================
// Lifecycle
// ===================================================================

static bool g_shm_discovered = false;

static void ensure_shm_discovered() {
    if (!g_shm_discovered) {
        shm_discover_peers();
        g_shm_discovered = true;
    }
}

void barrier() {
    ensure_mpi_initialized();
    MPI_Barrier(MPI_COMM_WORLD);
}

int get_rank() {
    ensure_mpi_initialized();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int get_size() {
    ensure_mpi_initialized();
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void finalize_mpi() {
    if (mpi_initialized_flag()) {
        shm_cleanup();
        MPI_Finalize();
        mpi_initialized_flag() = false;
        g_shm_discovered = false;
    }
}

// ===================================================================
// Send / Recv — dispatch to shm or standard MPI
// ===================================================================

void mpi_send(torch::Tensor& tensor, int dest) {
    ensure_mpi_initialized();
    ensure_shm_discovered();

    if (shm_is_peer(dest)) {
        shm_send(tensor, dest);
    } else {
        mpi_send_standard(tensor, dest);
    }
}

void mpi_recv(torch::Tensor& tensor, int source) {
    ensure_mpi_initialized();
    ensure_shm_discovered();

    if (shm_is_peer(source)) {
        shm_recv(tensor, source);
    } else {
        mpi_recv_standard(tensor, source);
    }
}

// ===================================================================
// Async send / recv — dispatch to shm or standard MPI
// ===================================================================

MpiRequest mpi_isend(torch::Tensor& tensor, int dest) {
    ensure_mpi_initialized();
    ensure_shm_discovered();

    if (shm_is_peer(dest))
        return shm_isend(tensor, dest);
    else
        return mpi_isend_standard(tensor, dest);
}

MpiRequest mpi_irecv(torch::Tensor& tensor, int source) {
    ensure_mpi_initialized();
    ensure_shm_discovered();

    if (shm_is_peer(source))
        return shm_irecv(tensor, source);
    else
        return mpi_irecv_standard(tensor, source);
}

// ===================================================================
// Collectives
// ===================================================================

void mpi_allreduce(torch::Tensor& tensor) {
    ensure_mpi_initialized();

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    sync_cuda_if_needed(tensor);

    MPI_Datatype datatype = get_mpi_datatype(tensor);
    int err = MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), tensor.numel(),
                            datatype, MPI_SUM, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed with error code: " + std::to_string(err));
    }

    sync_cuda_if_needed(tensor);
}

void mpi_allgather(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();
    ensure_shm_discovered();

    if (!sendbuf.is_contiguous()) sendbuf = sendbuf.contiguous();
    if (!recvbuf.is_contiguous()) recvbuf = recvbuf.contiguous();

    int rank = get_rank();
    int world_size = get_size();
    int count = sendbuf.numel();

    sync_cuda_if_needed(sendbuf);

    // Copy own data into the correct slot
    recvbuf.slice(0, rank * count, (rank + 1) * count).copy_(sendbuf);

    auto [datatype, typesize] = get_typesize(sendbuf.scalar_type());

    // Issue all sends (shm peers complete immediately, others are async MPI)
    std::vector<MpiRequest> send_reqs;
    send_reqs.reserve(world_size - 1);
    for (int i = 0; i < world_size; ++i) {
        if (i == rank) continue;
        send_reqs.push_back(mpi_isend(sendbuf, i));
    }

    // Issue all receives into the appropriate recvbuf slots
    std::vector<MpiRequest> recv_reqs;
    recv_reqs.reserve(world_size - 1);

    // We need tensor views for each recv slot; keep them alive until waits complete
    std::vector<torch::Tensor> recv_slices;
    recv_slices.reserve(world_size - 1);

    for (int i = 0; i < world_size; ++i) {
        if (i == rank) continue;
        auto slice = recvbuf.slice(0, i * count, (i + 1) * count);
        recv_slices.push_back(slice);
        recv_reqs.push_back(mpi_irecv(recv_slices.back(), i));
    }

    // Wait for all receives, then all sends
    mpi_waitall(recv_reqs);
    mpi_waitall(send_reqs);

    sync_cuda_if_needed(recvbuf);
}

void mpi_reduce_scatter(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();

    if (!sendbuf.is_contiguous()) sendbuf = sendbuf.contiguous();
    if (!recvbuf.is_contiguous()) recvbuf = recvbuf.contiguous();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    sync_cuda_if_needed(sendbuf);

    int count = recvbuf.numel();
    std::vector<int> recvcounts(world_size, count);
    auto [mpi_dtype, typesize] = get_typesize(sendbuf.scalar_type());

    int err = MPI_Reduce_scatter(sendbuf.data_ptr(), recvbuf.data_ptr(),
                                 recvcounts.data(), mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Reduce_scatter failed: " + std::to_string(err));

    sync_cuda_if_needed(recvbuf);
}

void mpi_alltoall(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();

    if (!sendbuf.is_contiguous()) sendbuf = sendbuf.contiguous();
    if (!recvbuf.is_contiguous()) recvbuf = recvbuf.contiguous();

    if (recvbuf.numel() != sendbuf.numel())
        throw std::runtime_error("mpi_alltoall: sendbuf and recvbuf must have the same numel");

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    sync_cuda_if_needed(sendbuf);

    int count = sendbuf.numel() / world_size;
    auto [mpi_dtype, typesize] = get_typesize(sendbuf.scalar_type());

    int err = MPI_Alltoall(sendbuf.data_ptr(), count, mpi_dtype,
                           recvbuf.data_ptr(), count, mpi_dtype, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Alltoall failed: " + std::to_string(err));

    sync_cuda_if_needed(recvbuf);
}

// ===================================================================
// pybind11
// ===================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("send",           &mpi_send,           "MPI Send (GPU-aware, shm-accelerated)");
    m.def("recv",           &mpi_recv,           "MPI Recv (GPU-aware, shm-accelerated)");
    m.def("allreduce",      &mpi_allreduce,      "MPI Allreduce (GPU-aware)");
    m.def("allgather",      &mpi_allgather,      "MPI Allgather (GPU-aware)");
    m.def("reduce_scatter", &mpi_reduce_scatter, "MPI Reduce_scatter (GPU-aware)");
    m.def("alltoall",       &mpi_alltoall,       "MPI Alltoall (GPU-aware)");
    m.def("barrier",        &barrier,            "MPI Barrier");
    m.def("get_rank",       &get_rank,           "Get MPI rank");
    m.def("get_size",       &get_size,           "Get MPI world size");
    m.def("finalize_mpi",   &finalize_mpi,       "Finalize MPI");
}
