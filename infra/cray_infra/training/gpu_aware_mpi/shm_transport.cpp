#include "shm_transport.h"
#include "shm_channel.h"
#include "common.h"

#include <cstring>

// MPI tag used for the 1-byte control signal (distinct from data-plane tag 0).
static constexpr int SHM_CTRL_TAG = 42;

void shm_send(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(dest, nbytes);

    // 1. GPU → shm DMA  (or CPU memcpy if tensor is on CPU)
    sync_cuda_if_needed(tensor);

    // Build a CPU tensor that views the shm data region, then copy into it.
    auto cpu_opts = torch::TensorOptions()
        .dtype(tensor.dtype())
        .device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.data(), {tensor.numel()}, cpu_opts);
    shm_tensor.copy_(tensor.is_cuda() ? tensor : tensor);

    if (tensor.is_cuda()) {
        sync_cuda_if_needed(tensor);  // wait for DMA to land in shm
    }

    // 2. Write byte count into header
    ch.header()->nbytes = nbytes;

    // 3. Signal the receiver via a 1-byte MPI message
    char sig = 1;
    int err = MPI_Send(&sig, 1, MPI_CHAR, dest, SHM_CTRL_TAG, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        throw std::runtime_error("shm_send: MPI_Send control signal failed: " +
                                 std::to_string(err));
    }
}

void shm_recv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    // 1. Wait for the sender's control signal
    char sig;
    MPI_Status status;
    int err = MPI_Recv(&sig, 1, MPI_CHAR, source, SHM_CTRL_TAG, MPI_COMM_WORLD, &status);
    if (err != MPI_SUCCESS) {
        throw std::runtime_error("shm_recv: MPI_Recv control signal failed: " +
                                 std::to_string(err));
    }

    // 2. Validate byte count
    uint64_t sent_bytes = ch.header()->nbytes;
    if (sent_bytes != nbytes) {
        throw std::runtime_error("shm_recv: expected " + std::to_string(nbytes) +
                                 " bytes but sender wrote " + std::to_string(sent_bytes));
    }

    // 3. shm → GPU DMA  (or CPU memcpy)
    auto cpu_opts = torch::TensorOptions()
        .dtype(tensor.dtype())
        .device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.data(), {tensor.numel()}, cpu_opts);
    tensor.copy_(shm_tensor);

    sync_cuda_if_needed(tensor);  // ensure DMA completes before returning
}
