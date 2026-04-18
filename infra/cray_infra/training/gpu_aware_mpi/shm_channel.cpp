#include "shm_channel.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <new>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#define cudaHostRegister       hipHostRegister
#define cudaHostUnregister     hipHostUnregister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaSuccess            hipSuccess
#endif

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
static std::unordered_set<int>              g_shm_peers;
static std::unordered_map<int, ShmChannel>  g_channels;
static bool                                 g_discovery_done = false;

static const char* SHM_PREFIX     = "scalarlm_";
static const char* PROBE_PREFIX   = "scalarlm_probe_";

// Default initial per-slot capacity (256 MB).  With SHM_NUM_SLOTS=2 the full
// mmap region for a directed half-channel is
//   2 * (sizeof(ShmSlotHeader) + 256 MB) + sizeof(atomic<uint64_t>)
// so about 512 MB of shm per direction, 1 GB per peer pair.
static constexpr size_t DEFAULT_CAPACITY = 256ULL * 1024 * 1024;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string probe_name(int rank) {
    return std::string(PROBE_PREFIX) + std::to_string(rank);
}

static std::string channel_name(int rank_a, int rank_b) {
    int lo = std::min(rank_a, rank_b);
    int hi = std::max(rank_a, rank_b);
    return std::string(SHM_PREFIX) + "ch_" + std::to_string(lo) + "_" + std::to_string(hi);
}

static void shm_unlink_safe(const std::string& name) {
    shm_unlink(name.c_str());  // ignore errors (may not exist)
}

// Total mmap bytes required for a channel with `per_slot_capacity` bytes per
// slot.  Two halves, each with SHM_NUM_SLOTS slots + one atomic reader_done.
static size_t channel_bytes_for_capacity(size_t per_slot_capacity) {
    size_t slot_stride = sizeof(ShmSlotHeader) + per_slot_capacity;
    size_t half_size   = SHM_NUM_SLOTS * slot_stride + sizeof(std::atomic<uint64_t>);
    return 2 * half_size;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

void shm_discover_peers() {
    if (g_discovery_done) return;

    ensure_mpi_initialized();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: each rank creates a probe file in /dev/shm
    std::string my_probe = probe_name(rank);
    shm_unlink_safe(my_probe);

    if (rank == 0) {
        std::cout << "[gpu_aware_mpi] shm discovery: probing " << size
                  << " ranks for /dev/shm visibility" << std::endl;
    }

    int fd = shm_open(my_probe.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        // /dev/shm not available — no peers
        if (rank == 0) {
            std::cout << "[gpu_aware_mpi] shm discovery: /dev/shm not available, "
                      << "all transfers will use MPI" << std::endl;
        }
        g_discovery_done = true;
        return;
    }
    ftruncate(fd, 1);
    close(fd);

    // Step 2: barrier so all probes are visible
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: check which other ranks' probes we can see
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        std::string peer_probe = probe_name(r);
        int pfd = shm_open(peer_probe.c_str(), O_RDONLY, 0);
        if (pfd >= 0) {
            g_shm_peers.insert(r);
            close(pfd);
        }
    }

    // Step 4: barrier then cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    shm_unlink_safe(my_probe);

    if (rank == 0) {
        if (g_shm_peers.empty()) {
            std::cout << "[gpu_aware_mpi] shm discovery: no shm peers found, "
                      << "all transfers will use MPI" << std::endl;
        } else {
            std::cout << "[gpu_aware_mpi] shm discovery: rank 0 can reach "
                      << g_shm_peers.size() << " peer(s) via /dev/shm:";
            for (int p : g_shm_peers) {
                std::cout << " " << p;
            }
            std::cout << std::endl;
        }
    }

    g_discovery_done = true;
}

bool shm_is_peer(int peer_rank) {
    return g_shm_peers.count(peer_rank) > 0;
}

// ---------------------------------------------------------------------------
// Channel creation / lookup
// ---------------------------------------------------------------------------

// Synchronize only the two ranks that share a channel.
// MPI_Barrier(MPI_COMM_WORLD) inside create_channel is wrong because channel
// creation is triggered lazily (on first use / on grow), so different rank
// pairs hit the barrier at different times — causing mismatched barriers and
// corrupted channel state.  A pairwise MPI_Sendrecv affects only the two
// peers involved and is safe to call from any point in the program.
static constexpr int PEER_SYNC_TAG = 43;  // distinct from SHM_CTRL_TAG = 42

static void peer_sync(int peer_rank) {
    char s = 0, r = 0;
    MPI_Sendrecv(&s, 1, MPI_CHAR, peer_rank, PEER_SYNC_TAG,
                 &r, 1, MPI_CHAR, peer_rank, PEER_SYNC_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Initialise the per-slot headers and reader_done counter for one half.
// Uses placement new because std::atomic is not trivially copyable — we
// mmap raw bytes, so the atomic has to be constructed in place.
static void init_half_in_place(void* half_base, size_t per_slot_capacity) {
    size_t slot_stride = sizeof(ShmSlotHeader) + per_slot_capacity;
    for (int s = 0; s < SHM_NUM_SLOTS; ++s) {
        auto* hdr = reinterpret_cast<ShmSlotHeader*>(
            static_cast<char*>(half_base) + s * slot_stride);
        // Zero first to wipe any leftover state from a recycled shm file,
        // then placement-new the header so its atomic<seq> is well-defined.
        std::memset(hdr, 0, sizeof(ShmSlotHeader));
        new (hdr) ShmSlotHeader();
        hdr->capacity = per_slot_capacity;
    }
    auto* rd = reinterpret_cast<std::atomic<uint64_t>*>(
        static_cast<char*>(half_base) + SHM_NUM_SLOTS * slot_stride);
    std::memset(rd, 0, sizeof(std::atomic<uint64_t>));
    new (rd) std::atomic<uint64_t>(0);
}

static ShmChannel create_channel(int peer_rank, size_t capacity) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string name  = channel_name(rank, peer_rank);
    size_t slot_stride = sizeof(ShmSlotHeader) + capacity;
    size_t half_size   = SHM_NUM_SLOTS * slot_stride + sizeof(std::atomic<uint64_t>);
    size_t total       = channel_bytes_for_capacity(capacity);

    int lo = std::min(rank, peer_rank);
    int fd;
    if (rank == lo) {
        shm_unlink_safe(name);
        fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0)
            throw std::runtime_error("shm_open create failed: " + std::string(strerror(errno)));
        if (ftruncate(fd, total) != 0) {
            close(fd);
            throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
        }
    }

    // Sync so the shm file exists before the higher rank opens it.
    peer_sync(peer_rank);

    if (rank != lo) {
        fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd < 0)
            throw std::runtime_error("shm_open open failed: " + std::string(strerror(errno)));
    }

    void* ptr = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED)
        throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));

    // Lower rank initialises both halves' headers + reader_done counters.
    if (rank == lo) {
        init_half_in_place(ptr, capacity);
        init_half_in_place(static_cast<char*>(ptr) + half_size, capacity);
    }

    // Sync so the higher rank doesn't read uninitialised headers.
    peer_sync(peer_rank);

    bool registered = false;
#if defined(USE_CUDA) || defined(USE_ROCM)
    if (cudaHostRegister(ptr, total, cudaHostRegisterDefault) == cudaSuccess)
        registered = true;
#endif

    if (rank == 0) {
        std::cout << "[gpu_aware_mpi] shm channel: " << name
                  << " created (" << SHM_NUM_SLOTS << " × "
                  << (capacity / (1024 * 1024)) << " MB per direction"
                  << ", cudaHostRegister=" << (registered ? "yes" : "no")
                  << ")" << std::endl;
    }

    ShmChannel ch;
    ch.peer_rank     = peer_rank;
    ch.mapped        = ptr;
    ch.mapped_size   = total;
    ch.registered    = registered;
    ch.half_capacity = capacity;
    ch.is_lower      = rank < peer_rank;
    ch.send_state    = {};
    // The local sender's backpressure counter lives in the send half — that
    // is, the RECEIVER (peer) writes to the reader_done counter of the half
    // the local rank writes into.  This is the pointer the sender spins on
    // to know when it can reuse a slot.
    ch.send_state.reader_done_shm = ch.reader_done(ch.send_half());

    return ch;
}

ShmChannel& shm_get_channel(int peer_rank, size_t min_capacity) {
    auto it = g_channels.find(peer_rank);
    if (it != g_channels.end() && it->second.capacity() >= min_capacity) {
        return it->second;
    }

    // Need to create or grow. If growing, tear down old first.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (it != g_channels.end()) {
        if (rank == 0) {
            std::cout << "[gpu_aware_mpi] shm channel: growing channel to peer "
                      << peer_rank << " from " << (it->second.capacity() / (1024 * 1024))
                      << " MB to " << (min_capacity / (1024 * 1024)) << " MB" << std::endl;
        }
        ShmChannel& old = it->second;
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (old.registered) cudaHostUnregister(old.mapped);
#endif
        munmap(old.mapped, old.mapped_size);
        if (rank < peer_rank) {
            shm_unlink_safe(channel_name(rank, peer_rank));
        }
        g_channels.erase(it);
    }

    size_t cap = std::max(min_capacity, DEFAULT_CAPACITY);
    g_channels[peer_rank] = create_channel(peer_rank, cap);
    return g_channels[peer_rank];
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void shm_cleanup() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0 && !g_channels.empty()) {
        std::cout << "[gpu_aware_mpi] shm cleanup: tearing down "
                  << g_channels.size() << " channel(s)" << std::endl;
    }

    for (auto& [peer, ch] : g_channels) {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (ch.registered) cudaHostUnregister(ch.mapped);
#endif
        munmap(ch.mapped, ch.mapped_size);
        if (rank < peer) {
            shm_unlink_safe(channel_name(rank, peer));
        }
    }
    g_channels.clear();
    g_shm_peers.clear();
    g_discovery_done = false;
}
