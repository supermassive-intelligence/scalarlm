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

// A duplex channel: two independent header+data regions inside one shm file.
// Half 0 (offset 0)                        : lo-rank → hi-rank
// Half 1 (offset sizeof(ShmHeader)+half_cap): hi-rank → lo-rank
struct ShmChannel {
    int       peer_rank;     // the remote rank this channel connects to
    void*     mapped;        // mmap base
    size_t    mapped_size;   // total mapped size (2 * (header + half_capacity))
    bool      registered;    // whether cudaHostRegister succeeded
    size_t    half_capacity; // data capacity of each direction
    bool      is_lower;      // true when local rank < peer_rank

    // Byte offset to the start of a given half (header included).
    size_t half_offset(int half) const {
        return half * (sizeof(ShmHeader) + half_capacity);
    }

    // --- send direction (local → peer) ---
    ShmHeader* send_header() const {
        int half = is_lower ? 0 : 1;
        return reinterpret_cast<ShmHeader*>(
            static_cast<char*>(mapped) + half_offset(half));
    }
    void* send_data() const {
        return reinterpret_cast<char*>(send_header()) + sizeof(ShmHeader);
    }

    // --- recv direction (peer → local) ---
    ShmHeader* recv_header() const {
        int half = is_lower ? 1 : 0;
        return reinterpret_cast<ShmHeader*>(
            static_cast<char*>(mapped) + half_offset(half));
    }
    void* recv_data() const {
        return reinterpret_cast<char*>(recv_header()) + sizeof(ShmHeader);
    }

    size_t capacity() const { return half_capacity; }
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
