#pragma once

#include "common.h"
#include <torch/extension.h>

// Send a tensor to `dest` via shared-memory DMA.
// The actual data travels through the mmap'd /dev/shm buffer;
// a 1-byte MPI message carrying the slot index is used as the "data ready" signal.
// Double-buffering ensures a second send cannot overwrite an unconsumed slot.
void shm_send(torch::Tensor& tensor, int dest);

// Receive a tensor from `source` via shared-memory DMA.
// Blocks on a 1-byte MPI control message, decodes the slot index,
// spin-waits on the generation counter for acquire ordering, then copies.
void shm_recv(torch::Tensor& tensor, int source);

// Non-blocking send: copies data into a shm slot eagerly, then posts a
// non-blocking MPI signal.  Returns immediately; nothing left to do at wait time.
MpiRequest shm_isend(torch::Tensor& tensor, int dest);

// Non-blocking recv: posts a non-blocking MPI_Irecv for the signal.
// The shm→tensor copy and slot-release happen in the on_complete callback
// invoked by mpi_waitall after MPI_Wait confirms the signal arrived.
MpiRequest shm_irecv(torch::Tensor& tensor, int source);
