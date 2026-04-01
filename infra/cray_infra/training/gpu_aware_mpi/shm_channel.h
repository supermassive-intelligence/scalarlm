#pragma once

#include "common.h"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// SHM channel layout
// ---------------------------------------------------------------------------
// Each channel is a file in /dev/shm shared between two ranks.
// Layout:  [ShmHeader][data buffer of capacity bytes]
//
// The header contains the byte count written by the sender and a flag
// that is toggled to signal completion.  The flag protocol is:
//   sender:  writes data, writes nbytes, then MPI_Send(1 byte) to peer
//   receiver: MPI_Recv(1 byte), reads nbytes, copies data out
// So the flag is actually carried out-of-band via MPI to avoid spin-waits.
// ---------------------------------------------------------------------------

struct ShmHeader {
    uint64_t nbytes;       // number of payload bytes written by sender
    uint64_t capacity;     // total data buffer capacity (after this header)
};

struct ShmChannel {
    int       peer_rank;   // the remote rank this channel connects to
    void*     mapped;      // mmap base (includes header)
    size_t    mapped_size; // total mapped size (header + data capacity)
    bool      registered;  // whether cudaHostRegister succeeded

    ShmHeader* header() const {
        return reinterpret_cast<ShmHeader*>(mapped);
    }
    void* data() const {
        return static_cast<char*>(mapped) + sizeof(ShmHeader);
    }
    size_t capacity() const {
        return mapped_size - sizeof(ShmHeader);
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Probe /dev/shm to discover which ranks share the same shm namespace.
// Must be called after MPI_Init and collectively by all ranks.
void shm_discover_peers();

// Returns true if `peer_rank` is reachable via /dev/shm.
bool shm_is_peer(int peer_rank);

// Get (or lazily create) a channel to `peer_rank`.
// Channels are created on first use and persist until shm_cleanup().
ShmChannel& shm_get_channel(int peer_rank, size_t min_capacity);

// Tear down all channels (unregister, munmap, unlink).
void shm_cleanup();
