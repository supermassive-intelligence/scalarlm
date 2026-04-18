#pragma once

#include "common.h"
#include <torch/extension.h>

// Send a tensor to `dest` via shared-memory DMA.
// Protocol: double-buffered slots in shm_channel with per-slot generation
// counters.  The sender picks a slot, memcpys the payload in, release-stores
// the generation into the slot header, then MPI_Sends a 16-byte ShmSignal
// carrying {gen, slot_idx} to the receiver.  See shm_channel.h for layout.
void shm_send(torch::Tensor& tensor, int dest);

// Receive a tensor from `source` via shared-memory DMA.
// Blocks on MPI_Recv of the ShmSignal, acquire-waits for slot.seq to match,
// copies out, and fetch-adds the reader_done counter so the sender can
// reuse the slot.
void shm_recv(torch::Tensor& tensor, int source);

// Non-blocking variants.  shm_isend eagerly copies into the slot + posts
// MPI_Isend for the signal; mpi_wait has nothing extra to do.  shm_irecv
// posts MPI_Irecv for the signal; on_complete spin-waits on seq, copies
// the payload out, and bumps reader_done.
MpiRequest shm_isend(torch::Tensor& tensor, int dest);
MpiRequest shm_irecv(torch::Tensor& tensor, int source);
