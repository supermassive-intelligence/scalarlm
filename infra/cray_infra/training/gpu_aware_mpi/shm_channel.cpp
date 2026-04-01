#include "shm_channel.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <iostream>

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

// Default initial capacity for the data buffer (256 MB).
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

static ShmChannel create_channel(int peer_rank, size_t capacity) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string name = channel_name(rank, peer_rank);
    size_t total = sizeof(ShmHeader) + capacity;

    // The lower-ranked peer creates; both open.
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

    // Sync so the file exists before the higher rank opens it.
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != lo) {
        fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd < 0)
            throw std::runtime_error("shm_open open failed: " + std::string(strerror(errno)));
    }

    void* ptr = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED)
        throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));

    // Zero the header on creation (lower rank owns this)
    if (rank == lo) {
        memset(ptr, 0, sizeof(ShmHeader));
        auto* hdr = reinterpret_cast<ShmHeader*>(ptr);
        hdr->capacity = capacity;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Pin for CUDA DMA
    bool registered = false;
#if defined(USE_CUDA) || defined(USE_ROCM)
    if (cudaHostRegister(ptr, total, cudaHostRegisterDefault) == cudaSuccess) {
        registered = true;
    }
#endif

    if (rank == 0) {
        std::cout << "[gpu_aware_mpi] shm channel: " << name
                  << " created (" << (capacity / (1024 * 1024)) << " MB"
                  << ", cudaHostRegister=" << (registered ? "yes" : "no")
                  << ")" << std::endl;
    }

    return ShmChannel{peer_rank, ptr, total, registered};
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
