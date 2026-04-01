#pragma once

#include <torch/extension.h>

// Standard MPI_Send path (3 copies when GPU tensors go through CMA).
void mpi_send_standard(torch::Tensor& tensor, int dest);

// Standard MPI_Recv path.
void mpi_recv_standard(torch::Tensor& tensor, int source);
