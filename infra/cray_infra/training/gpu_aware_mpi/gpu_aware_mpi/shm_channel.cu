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

static const char* SHM_PREFIX   = "scalarlm_";
static const char* PROBE_PREFIX = "scalarlm_probe_";

// Default initial data capacity per slot (256 MB).
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
    shm_unlink(name.c_str());
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

    std::string my_probe = probe_name(rank);
    shm_unlink_safe(my_probe);

    if (rank == 0) {
        std::cout << "[gpu_aware_mpi] shm discovery: probing " << size
                  << " ranks for /dev/shm visibility" << std::endl;
    }

    int fd = shm_open(my_probe.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        if (rank == 0) {
            std::cout << "[gpu_aware_mpi] shm discovery: /dev/shm not available, "
                      << "all transfers will use MPI" << std::endl;
        }
        g_discovery_done = true;
        return;
    }
    ftruncate(fd, 1);
    close(fd);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        std::string peer_probe = probe_name(r);
        int pfd = shm_open(peer_probe.c_str(), O_RDONLY, 0);
        if (pfd >= 0) {
            g_shm_peers.insert(r);
            close(pfd);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    shm_unlink_safe(my_probe);

    if (rank == 0) {
        if (g_shm_peers.empty()) {
            std::cout << "[gpu_aware_mpi] shm discovery: no shm peers found, "
                      << "all transfers will use MPI" << std::endl;
        } else {
            std::cout << "[gpu_aware_mpi] shm discovery: rank 0 can reach "
                      << g_shm_peers.size() << " peer(s) via /dev/shm:";
            for (int p : g_shm_peers) std::cout << " " << p;
            std::cout << std::endl;
        }
    }

    g_discovery_done = true;
}

bool shm_is_peer(int peer_rank) {
    return g_shm_peers.count(peer_rank) > 0;
}

// ---------------------------------------------------------------------------
// Peer-pair barrier
// ---------------------------------------------------------------------------

static constexpr int PEER_SYNC_TAG = 43;

static void peer_sync(int peer_rank) {
    char s = 0, r = 0;
    MPI_Sendrecv(&s, 1, MPI_CHAR, peer_rank, PEER_SYNC_TAG,
                 &r, 1, MPI_CHAR, peer_rank, PEER_SYNC_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// ---------------------------------------------------------------------------
// Channel creation
//
//   2 halves × [SHM_NUM_SLOTS × (ShmSlotHeader(64B) + data(capacity))
//                       + atomic<uint64_t>(8B) reader_done]
//
// The lower-rank process creates and initializes the shm file; the higher-rank
// process maps it read-write after a pairwise sync.
// ---------------------------------------------------------------------------

static ShmChannel create_channel(int peer_rank, size_t capacity) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string name = channel_name(rank, peer_rank);

    // Compute sizes using the same formulas as ShmChannel member functions.
    size_t slot_stride = sizeof(ShmSlotHeader) + capacity;
    size_t half_size   = SHM_NUM_SLOTS * slot_stride + sizeof(std::atomic<uint64_t>);
    size_t total       = 2 * half_size;

    int lo = std::min(rank, peer_rank);
    int fd;
    if (rank == lo) {
        shm_unlink_safe(name);
        fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0)
            throw std::runtime_error("shm_open create failed: " + std::string(strerror(errno)));
        if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
            close(fd);
            throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
        }
    }

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

    // Lower rank zero-initializes all slot headers and reader_done atomics
    // via placement new so atomics are in a valid state.
    if (rank == lo) {
        char* base = static_cast<char*>(ptr);
        for (int h = 0; h < 2; ++h) {
            char* half_base = base + h * half_size;
            for (int s = 0; s < SHM_NUM_SLOTS; ++s) {
                auto* hdr = reinterpret_cast<ShmSlotHeader*>(
                    half_base + static_cast<size_t>(s) * slot_stride);
                new (hdr) ShmSlotHeader();   // placement new: seq=0, nbytes=0
                hdr->capacity = capacity;
            }
            // Placement-new the reader_done atomic after the last slot.
            auto* rd = reinterpret_cast<std::atomic<uint64_t>*>(
                half_base + SHM_NUM_SLOTS * slot_stride);
            new (rd) std::atomic<uint64_t>(0);
        }
    }

    peer_sync(peer_rank);   // higher rank waits for initialization

    bool registered = false;
#if defined(USE_CUDA) || defined(USE_ROCM)
    int reg_flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
    auto reg_result = cudaHostRegister(ptr, total, reg_flags);
    if (reg_result == cudaSuccess) {
        registered = true;
    } else {
        std::cerr << "[gpu_aware_mpi] cudaHostRegister failed with code "
                  << reg_result << ", falling back to unregistered shm" << std::endl;
    }
#endif

    if (rank == 0) {
        std::cout << "[gpu_aware_mpi] shm channel: " << name
                  << " created (" << (capacity / (1024 * 1024))
                  << " MB x" << SHM_NUM_SLOTS << " slots x2 dirs"
                  << ", cudaHostRegister=" << (registered ? "yes" : "no")
                  << ")" << std::endl;
    }

    ShmChannel ch;
    ch.peer_rank    = peer_rank;
    ch.mapped       = ptr;
    ch.mapped_size  = total;
    ch.registered   = registered;
    ch.half_capacity = capacity;
    ch.is_lower     = (rank < peer_rank);

    // Wire up the sender's reader_done pointer to the shm atomic so the
    // receiver's increment is visible to the sender's slot-acquire spin.
    ch.send_state.reader_done_shm = ch.reader_done_ptr(ch.send_half());

    return ch;
}

// ---------------------------------------------------------------------------
// shm_get_channel  
// ---------------------------------------------------------------------------

ShmChannel& shm_get_channel(int peer_rank, size_t min_capacity) {
    auto it = g_channels.find(peer_rank);
    if (it != g_channels.end() && it->second.capacity() >= min_capacity) {
        return it->second;
    }

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
        if (rank < peer_rank)
            shm_unlink_safe(channel_name(rank, peer_rank));
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
        if (rank < peer)
            shm_unlink_safe(channel_name(rank, peer));
    }
    g_channels.clear();
    g_shm_peers.clear();
    g_discovery_done = false;
}
