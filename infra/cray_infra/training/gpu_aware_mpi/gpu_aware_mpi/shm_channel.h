#pragma once

#include "common.h"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <cstddef>
#include <atomic>

// ---------------------------------------------------------------------------
// SHM channel layout
// ---------------------------------------------------------------------------
// Each channel is a file in /dev/shm shared between two ranks.
//
//
// Layout (double-buffered, one slot per direction):
//   [ShmSlot 0][ShmSlot 1]   (two independent header+data regions per direction)

//
//   Double-buffering (ping-pong slots):
//      Each directed channel holds SHM_NUM_SLOTS=2 independent data slots.
//      The sender alternates write_slot 0↔1 on each send.
//      The slot index is embedded in the 1-byte MPI signal so the receiver
//      knows exactly which slot to read from.
//      The sender blocks before reusing a slot until the receiver has
//      confirmed it is done (via reader_done atomic), so a fast sender cannot
//      lap a slow receiver.
//
//   Generation counter + acquire/release ordering:
//      Each slot header contains an atomic seq counter.  The sender
//      increments seq *after* the memcpy (release store).  The receiver
//      spin-waits until slot.seq matches the expected generation before
//      reading data.  This closes the window where the MPI signal races
//      ahead of the store becoming visible on the receiver's core.
// ---------------------------------------------------------------------------

// Number of ping-pong slots per directed half-channel.
static constexpr int SHM_NUM_SLOTS = 2;

struct ShmSlotHeader {
    // Incremented by the sender (release) after every memcpy into this slot.
    // Read by the receiver (acquire) before copying out.
    // Padded to one cache line to avoid false sharing with the data buffer.
    std::atomic<uint64_t> seq;
    uint64_t              nbytes;      // payload bytes written this generation
    uint64_t              capacity;    // max bytes this slot can hold
    uint8_t               _pad[64 - sizeof(std::atomic<uint64_t>) - 2 * sizeof(uint64_t)];

    ShmSlotHeader() : seq(0), nbytes(0), capacity(0) {}
};
static_assert(sizeof(ShmSlotHeader) == 64, "ShmSlotHeader must be exactly one cache line");

// One slot = one header + one data buffer.
// The data buffer immediately follows the header in the mmap region.
// We do not embed the data array here because the capacity is runtime-determined.
// Use ShmSlotHeader* arithmetic to reach the data (see helpers in ShmChannel).

// Per-directed-channel control block that lives *outside* the mmap region
// (in process-local heap memory).  Tracks sender state that doesn't need to
// be shared.
struct ShmDirState {
    uint32_t write_slot{0};    // next slot index to write into (0 or 1)
    uint64_t next_gen{1};      // next generation number to stamp on a slot

    // Running count of slots the *receiver* has fully consumed.
    // The sender waits: next_gen - reader_done <= SHM_NUM_SLOTS.
    // This field IS shared (receiver writes it), so it lives in shm.
    // We keep a pointer to the shm copy here for convenience.
    std::atomic<uint64_t>* reader_done_shm{nullptr};
};

// A duplex channel: two directed half-channels inside one shm file.
// Half 0: lo-rank → hi-rank
// Half 1: hi-rank → lo-rank
// Each half has SHM_NUM_SLOTS slots.
struct ShmChannel {
    int    peer_rank;
    void*  mapped;           // mmap base
    size_t mapped_size;      // total mmap size
    bool   registered;       // cudaHostRegister succeeded
    size_t half_capacity;    // data capacity per slot (same for both slots in a half)
    bool   is_lower;         // true when local rank < peer_rank

    // Process-local sender state for the send direction (local → peer).
    // Only the local process writes here, so no sharing needed.
    ShmDirState send_state;

    // ---------------------------------------------------------------------------
    // Layout arithmetic
    // ---------------------------------------------------------------------------
    //
    // Each "half" of the channel contains SHM_NUM_SLOTS slots laid out as:
    //
    //   [ShmSlotHeader slot0][data0 of half_capacity bytes]
    //   [ShmSlotHeader slot1][data1 of half_capacity bytes]
    //   [atomic<uint64_t> reader_done]   ← shared, written by receiver
    //
    // slot_stride = sizeof(ShmSlotHeader) + half_capacity
    // half_size   = SHM_NUM_SLOTS * slot_stride + sizeof(atomic<uint64_t>)
    //
    // half 0 starts at offset 0, half 1 starts at offset half_size.

    size_t slot_stride() const {
        return sizeof(ShmSlotHeader) + half_capacity;
    }

    size_t half_size() const {
        return SHM_NUM_SLOTS * slot_stride() + sizeof(std::atomic<uint64_t>);
    }

    size_t half_offset(int half) const {
        return static_cast<size_t>(half) * half_size();
    }

    ShmSlotHeader* slot_header(int half, int slot) const {
        char* base = static_cast<char*>(mapped) + half_offset(half);
        return reinterpret_cast<ShmSlotHeader*>(base + static_cast<size_t>(slot) * slot_stride());
    }

    void* slot_data(int half, int slot) const {
        return reinterpret_cast<char*>(slot_header(half, slot)) + sizeof(ShmSlotHeader);
    }

    // The reader_done atomic sits after the last slot in the half.
    std::atomic<uint64_t>* reader_done_ptr(int half) const {
        char* base = static_cast<char*>(mapped) + half_offset(half);
        return reinterpret_cast<std::atomic<uint64_t>*>(
            base + SHM_NUM_SLOTS * slot_stride());
    }

    // Convenience: which half index is "send" and which is "recv" for this rank.
    int send_half() const { return is_lower ? 0 : 1; }
    int recv_half() const { return is_lower ? 1 : 0; }

    // Legacy-compatible accessors used by existing send/recv code.
    ShmSlotHeader* send_header(int slot) const { return slot_header(send_half(), slot); }
    void*          send_data  (int slot) const { return slot_data  (send_half(), slot); }
    ShmSlotHeader* recv_header(int slot) const { return slot_header(recv_half(), slot); }
    void*          recv_data  (int slot) const { return slot_data  (recv_half(), slot); }

    size_t capacity() const { return half_capacity; }
};

// ---------------------------------------------------------------------------
// Signal helpers
//
// ShmSignal is defined in common.h (included above) so MpiRequest can embed
// it as stable MPI buffer storage without a circular dependency.
// ---------------------------------------------------------------------------

static inline ShmSignal encode_signal(uint64_t gen, uint32_t slot_idx) {
    return ShmSignal{gen, slot_idx, 0};
}

static inline uint32_t decode_signal_slot(const ShmSignal& sig) {
    return sig.slot_idx;
}

static inline uint64_t decode_signal_gen(const ShmSignal& sig) {
    return sig.gen;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void shm_discover_peers();
bool shm_is_peer(int peer_rank);
ShmChannel& shm_get_channel(int peer_rank, size_t min_capacity);
void shm_cleanup();
