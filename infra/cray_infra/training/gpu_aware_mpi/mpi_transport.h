#pragma once

#include "common.h"
#include <torch/extension.h>

// Standard MPI_Send path (3 copies when GPU tensors go through CMA).
void mpi_send_standard(torch::Tensor& tensor, int dest);

// Standard MPI_Recv path.
void mpi_recv_standard(torch::Tensor& tensor, int source);

// Non-blocking standard MPI send/recv.
MpiRequest mpi_isend_standard(torch::Tensor& tensor, int dest);
MpiRequest mpi_irecv_standard(torch::Tensor& tensor, int dest);
