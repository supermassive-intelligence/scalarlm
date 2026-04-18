#pragma once

#include "common.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

// ---------------------------------------------------------------------------
// SHM channel layout
// ---------------------------------------------------------------------------
// Each channel is one file in /dev/shm shared between two ranks.  The file
// holds TWO directed half-channels (one per direction); each half is
// double-buffered into SHM_NUM_SLOTS ping-pong slots plus one atomic
// reader_done counter:
//
//      half N (N=0 is lo→hi, N=1 is hi→lo):
//          ┌─ slot 0 ─────────────────────────────────┐
//          │  ShmSlotHeader (cache-line padded)       │
//          │    atomic<uint64_t> seq                  │
//          │    uint64_t nbytes                       │
//          │    uint64_t capacity                     │
//          │  data buffer of half_capacity bytes       │
//          ├─ slot 1 ─────────────────────────────────┤
//          │  ShmSlotHeader                            │
//          │  data buffer of half_capacity bytes       │
//          ├──────────────────────────────────────────┤
//          │  atomic<uint64_t> reader_done             │
//          └──────────────────────────────────────────┘
//
// Protocol (sender):
//   1.  Spin-wait until (next_gen - reader_done_shm) <= SHM_NUM_SLOTS.
//       That is the backpressure — sender cannot lap receiver by more than
//       the slot count, so a slow receiver's data doesn't get clobbered.
//   2.  Pick slot = write_slot.  Memcpy payload into slot.data.  Stamp
//       slot.nbytes.  Release-store slot.seq = next_gen.  The release fence
//       pairs with the receiver's acquire load so the memcpy becomes
//       visible before the seq does on the peer's core.
//   3.  MPI_Send(ShmSignal{gen=next_gen, slot_idx=slot}).
//   4.  Advance write_slot (0↔1) and next_gen.
//
// Protocol (receiver):
//   1.  MPI_Recv(ShmSignal).
//   2.  Spin-load slot.seq with acquire ordering until it equals signal.gen.
//       In practice this is a short spin: MPI only delivers the signal
//       after both cores' caches have coherency on the payload.
//   3.  Memcpy slot.data → destination tensor.
//   4.  Fetch-add reader_done to unblock the sender's next reuse of a slot.
//
// The double buffering lets the sender pipeline one transfer ahead of the
// receiver without risking a write-before-read clobber.  SHM_NUM_SLOTS=2 is
// the smallest value that provides any pipelining; raising it further would
// only help if MPI's signal delivery were the bottleneck, which it isn't
// for intra-node shm transfers.
// ---------------------------------------------------------------------------

// Number of ping-pong slots per directed half-channel.
static constexpr int SHM_NUM_SLOTS = 2;

// Per-slot header.  Padded to one cache line so the seq atomic does not
// false-share with the data buffer that follows it in the mmap region.
struct ShmSlotHeader {
    std::atomic<uint64_t> seq;       // incremented by sender (release) on every write
    uint64_t              nbytes;    // payload bytes the sender wrote for slot.seq
    uint64_t              capacity;  // max bytes this slot can hold (set at create time)
    // Pad to exactly 64 bytes so the data buffer begins on a cache-line
    // boundary and seq owns its own line.
    uint8_t               _pad[64 - sizeof(std::atomic<uint64_t>) - 2 * sizeof(uint64_t)];

    ShmSlotHeader() : seq(0), nbytes(0), capacity(0) {}
};
static_assert(sizeof(ShmSlotHeader) == 64,
              "ShmSlotHeader must be exactly one cache line");

// Process-local sender state for the send direction (local → peer).
// Nothing in this struct needs to live in shm — the sender is the only
// writer.  The reader_done pointer, however, points INTO the shm region
// because the receiver writes to it.
struct ShmDirState {
    uint32_t                 write_slot{0};          // next slot to write (0 or 1)
    uint64_t                 next_gen{1};            // next generation to stamp on a slot
    std::atomic<uint64_t>*   reader_done_shm{nullptr};  // shared fetch-add'd by receiver
};

// A duplex channel: two directed half-channels inside one shm file.
// Half 0 (offset 0):         lo-rank → hi-rank
// Half 1 (offset half_size): hi-rank → lo-rank
struct ShmChannel {
    int       peer_rank;     // the remote rank this channel connects to
    void*     mapped;        // mmap base
    size_t    mapped_size;   // total mapped size
    bool      registered;    // whether cudaHostRegister succeeded
    size_t    half_capacity; // per-slot data capacity (same for both slots in a half)
    bool      is_lower;      // true when local rank < peer_rank

    // Process-local sender state for the send direction only (no receiver
    // state here — the receiver only reads shm + writes reader_done_shm).
    ShmDirState send_state;

    // -----------------------------------------------------------------------
    // Layout arithmetic
    // -----------------------------------------------------------------------
    //   slot_stride = sizeof(ShmSlotHeader) + half_capacity
    //   half_size   = SHM_NUM_SLOTS * slot_stride + sizeof(atomic<uint64_t>)
    //   reader_done_offset = SHM_NUM_SLOTS * slot_stride

    size_t slot_stride() const {
        return sizeof(ShmSlotHeader) + half_capacity;
    }
    size_t half_size() const {
        return SHM_NUM_SLOTS * slot_stride() + sizeof(std::atomic<uint64_t>);
    }
    size_t half_offset(int half) const {
        return static_cast<size_t>(half) * half_size();
    }

    // --- absolute positions ---
    ShmSlotHeader* slot_header(int half, int slot) const {
        auto* base = static_cast<char*>(mapped) + half_offset(half);
        return reinterpret_cast<ShmSlotHeader*>(base + slot * slot_stride());
    }
    void* slot_data(int half, int slot) const {
        return reinterpret_cast<char*>(slot_header(half, slot)) +
               sizeof(ShmSlotHeader);
    }
    std::atomic<uint64_t>* reader_done(int half) const {
        auto* base = static_cast<char*>(mapped) + half_offset(half);
        return reinterpret_cast<std::atomic<uint64_t>*>(
            base + SHM_NUM_SLOTS * slot_stride());
    }

    // --- direction-aware helpers ---
    int send_half() const { return is_lower ? 0 : 1; }
    int recv_half() const { return is_lower ? 1 : 0; }

    ShmSlotHeader* send_slot_header(int slot) const {
        return slot_header(send_half(), slot);
    }
    void* send_slot_data(int slot) const {
        return slot_data(send_half(), slot);
    }
    ShmSlotHeader* recv_slot_header(int slot) const {
        return slot_header(recv_half(), slot);
    }
    void* recv_slot_data(int slot) const {
        return slot_data(recv_half(), slot);
    }
    std::atomic<uint64_t>* recv_reader_done() const {
        // The counter the local RECEIVER writes into this channel's recv
        // direction lives in the recv half.
        return reader_done(recv_half());
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
// Channels are created on first use and persist until shm_cleanup().  If the
// existing channel's per-slot capacity is smaller than `min_capacity` the
// channel is torn down and recreated.
ShmChannel& shm_get_channel(int peer_rank, size_t min_capacity);

// Tear down all channels (unregister, munmap, unlink).
void shm_cleanup();
