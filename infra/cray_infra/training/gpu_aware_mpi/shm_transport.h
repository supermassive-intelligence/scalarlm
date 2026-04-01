#pragma once

#include "common.h"
#include <torch/extension.h>

// Send a tensor to `dest` via shared-memory DMA.
// The actual data travels through the mmap'd /dev/shm buffer;
// a 1-byte MPI message is used as the "data ready" signal.
void shm_send(torch::Tensor& tensor, int dest);

// Receive a tensor from `source` via shared-memory DMA.
// Blocks on a 1-byte MPI control message, then copies from shm to tensor.
void shm_recv(torch::Tensor& tensor, int source);

// Non-blocking variants for use in collectives.
// The shm path completes synchronously (memcpy is local), so the returned
// MpiRequest is already marked completed.
MpiRequest shm_isend(torch::Tensor& tensor, int dest);
MpiRequest shm_irecv(torch::Tensor& tensor, int source);
