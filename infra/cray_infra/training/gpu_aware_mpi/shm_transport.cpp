#include "shm_transport.h"
#include "shm_channel.h"
#include "common.h"

#include <cstring>

static constexpr int SHM_CTRL_TAG = 42;

// ---------------------------------------------------------------------------
// Blocking send/recv — used directly when ordering is already guaranteed
// ---------------------------------------------------------------------------

void shm_send(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(dest, nbytes);
    sync_cuda_if_needed(tensor);

    auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.send_data(), {tensor.numel()}, cpu_opts);
    shm_tensor.copy_(tensor);
    if (tensor.is_cuda()) sync_cuda_if_needed(tensor);

    ch.send_header()->nbytes = nbytes;

    char sig = 1;
    int err = MPI_Send(&sig, 1, MPI_CHAR, dest, SHM_CTRL_TAG, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_send: MPI_Send control signal failed: " +
                                 std::to_string(err));
}

void shm_recv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    char sig;
    MPI_Status status;
    int err = MPI_Recv(&sig, 1, MPI_CHAR, source, SHM_CTRL_TAG, MPI_COMM_WORLD, &status);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_recv: MPI_Recv control signal failed: " +
                                 std::to_string(err));

    uint64_t sent_bytes = ch.recv_header()->nbytes;
    if (sent_bytes != nbytes)
        throw std::runtime_error("shm_recv: expected " + std::to_string(nbytes) +
                                 " bytes but sender wrote " + std::to_string(sent_bytes));

    auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.recv_data(), {tensor.numel()}, cpu_opts);
    tensor.copy_(shm_tensor);
    sync_cuda_if_needed(tensor);
}

// ---------------------------------------------------------------------------
// Truly async send — copies data into shm eagerly, then posts a non-blocking
// control signal.  The send-side has nothing left to do at wait time.
// ---------------------------------------------------------------------------

MpiRequest shm_isend(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(dest, nbytes);
    sync_cuda_if_needed(tensor);

    // Eagerly copy into shm — this is cheap (local memcpy / DMA).
    auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.send_data(), {tensor.numel()}, cpu_opts);
    shm_tensor.copy_(tensor);
    if (tensor.is_cuda()) sync_cuda_if_needed(tensor);

    ch.send_header()->nbytes = nbytes;

    // Post the 1-byte control signal non-blocking.
    // shm_sig inside the returned MpiRequest provides stable storage for the
    // signal byte — MPI is not allowed to read it until MPI_Wait completes.
    MpiRequest req;
    req.shm_sig = 1;
    int err = MPI_Isend(&req.shm_sig, 1, MPI_CHAR, dest, SHM_CTRL_TAG,
                        MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_isend: MPI_Isend control signal failed: " +
                                 std::to_string(err));
    return req;  // on_complete is empty; nothing to do on the send side at wait time
}

// ---------------------------------------------------------------------------
// Truly async recv — posts a non-blocking recv for the control signal.
// The actual shm→tensor copy is deferred to the on_complete callback, which
// mpi_wait / mpi_waitall invoke after MPI_Wait confirms the signal arrived.
// ---------------------------------------------------------------------------

MpiRequest shm_irecv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    MpiRequest req;

    // Capture everything the callback needs by value/pointer.
    // - tensor: torch::Tensor is a reference-counted handle; safe to copy.
    // - &ch:    channels persist until shm_cleanup(), which is after all waits.
    // - nbytes: plain value.
    req.on_complete = [tensor, &ch, nbytes]() mutable {
        uint64_t sent_bytes = ch.recv_header()->nbytes;
        if (sent_bytes != nbytes)
            throw std::runtime_error("shm_irecv: expected " + std::to_string(nbytes) +
                                     " bytes but sender wrote " + std::to_string(sent_bytes));

        auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
        torch::Tensor shm_tensor = torch::from_blob(ch.recv_data(), {tensor.numel()}, cpu_opts);
        tensor.copy_(shm_tensor);
        sync_cuda_if_needed(tensor);
    };

    int err = MPI_Irecv(&req.shm_sig, 1, MPI_CHAR, source, SHM_CTRL_TAG,
                        MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_irecv: MPI_Irecv control signal failed: " +
                                 std::to_string(err));
    return req;
}
