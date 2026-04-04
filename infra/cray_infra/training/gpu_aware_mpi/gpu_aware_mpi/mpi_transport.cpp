#include "mpi_transport.h"
#include "common.h"

void mpi_send_standard(torch::Tensor& tensor, int dest) {
    ensure_mpi_initialized();

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    sync_cuda_if_needed(tensor);

    MPI_Datatype datatype = get_mpi_datatype(tensor);
    int err = MPI_Send(tensor.data_ptr(), tensor.numel(), datatype, dest, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Send failed with error code: " + std::to_string(err));
    }
}

void mpi_recv_standard(torch::Tensor& tensor, int source) {
    ensure_mpi_initialized();

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    MPI_Datatype datatype = get_mpi_datatype(tensor);
    MPI_Status status;

    int err = MPI_Recv(tensor.data_ptr(), tensor.numel(), datatype, source, 0,
                       MPI_COMM_WORLD, &status);
    if (err != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Recv failed with error code: " + std::to_string(err));
    }

    sync_cuda_if_needed(tensor);

    int recv_count;
    MPI_Get_count(&status, datatype, &recv_count);
    if (recv_count != tensor.numel()) {
        throw std::runtime_error("MPI_Recv: expected " + std::to_string(tensor.numel()) +
                                 " elements but got " + std::to_string(recv_count));
    }

    if (status.MPI_SOURCE != source) {
        throw std::runtime_error("MPI_Recv: expected source " + std::to_string(source) +
                                 " but got " + std::to_string(status.MPI_SOURCE));
    }
}

MpiRequest mpi_isend_standard(torch::Tensor& tensor, int dest) {
    ensure_mpi_initialized();

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    sync_cuda_if_needed(tensor);

    MpiRequest req;
    MPI_Datatype datatype = get_mpi_datatype(tensor);
    int err = MPI_Isend(tensor.data_ptr(), tensor.numel(), datatype, dest, 0,
                        MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Isend failed: " + std::to_string(err));
    return req;
}

MpiRequest mpi_irecv_standard(torch::Tensor& tensor, int source) {
    ensure_mpi_initialized();

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    MpiRequest req;
    MPI_Datatype datatype = get_mpi_datatype(tensor);
    int err = MPI_Irecv(tensor.data_ptr(), tensor.numel(), datatype, source, 0,
                        MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Irecv failed: " + std::to_string(err));
    return req;
}
