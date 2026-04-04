#include "shm_transport.h"
#include "shm_channel.h"
#include "common.h"

#include <cstring>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
static inline void cpu_relax() {
    _mm_pause();
}
#elif defined(__aarch64__) || defined(__arm__)
static inline void cpu_relax() {
    __asm__ __volatile__("yield");
}
#else
static inline void cpu_relax() {
}
#endif

static constexpr int SHM_CTRL_TAG = 42;

// ---------------------------------------------------------------------------
// Helper: memcpy that handles GPU tensors correctly
// ---------------------------------------------------------------------------

static inline void memcpy_to_shm(void* dst, const torch::Tensor& tensor, size_t nbytes) {
#if defined(USE_CUDA)
    if (tensor.is_cuda()) {
        cudaMemcpy(dst, tensor.data_ptr(), nbytes, cudaMemcpyDeviceToHost);
        return;
    }
#endif
    std::memcpy(dst, tensor.data_ptr(), nbytes);
}

static inline void memcpy_from_shm(torch::Tensor& tensor, const void* src, size_t nbytes) {
#if defined(USE_CUDA)
    if (tensor.is_cuda()) {
        cudaMemcpy(tensor.data_ptr(), src, nbytes, cudaMemcpyHostToDevice);
        return;
    }
#endif
    std::memcpy(tensor.data_ptr(), src, nbytes);
}

// ---------------------------------------------------------------------------
// Internal: acquire a send slot
// ---------------------------------------------------------------------------

static uint32_t sender_acquire_slot(ShmChannel& ch) {
    ShmDirState& st = ch.send_state;
    uint32_t idx    = st.write_slot % SHM_NUM_SLOTS;

    if (st.next_gen > static_cast<uint64_t>(SHM_NUM_SLOTS)) {
        uint64_t required = st.next_gen - static_cast<uint64_t>(SHM_NUM_SLOTS);
        while (st.reader_done_shm->load(std::memory_order_acquire) < required) {
            cpu_relax();
        }
    }

    return idx;
}

// ---------------------------------------------------------------------------
// Blocking send
// ---------------------------------------------------------------------------

void shm_send(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch  = shm_get_channel(dest, nbytes);
    uint32_t    idx = sender_acquire_slot(ch);

    memcpy_to_shm(ch.send_data(idx), tensor, nbytes);
    ch.send_header(idx)->nbytes = nbytes;

    uint64_t gen = ch.send_state.next_gen++;
    ch.send_header(idx)->seq.store(gen, std::memory_order_release);

    ch.send_state.write_slot = (idx + 1) % SHM_NUM_SLOTS;

    ShmSignal sig = encode_signal(gen, idx);
    int err = MPI_Send(&sig, sizeof(ShmSignal), MPI_BYTE, dest, SHM_CTRL_TAG, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_send: MPI_Send control signal failed: " +
                                 std::to_string(err));
}

// ---------------------------------------------------------------------------
// Blocking recv
// ---------------------------------------------------------------------------

void shm_recv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    ShmSignal  sig{};
    MPI_Status status;
    int err = MPI_Recv(&sig, sizeof(ShmSignal), MPI_BYTE, source, SHM_CTRL_TAG,
                       MPI_COMM_WORLD, &status);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_recv: MPI_Recv control signal failed: " +
                                 std::to_string(err));

    uint32_t idx          = decode_signal_slot(sig);
    uint64_t expected_gen = decode_signal_gen(sig);

    ShmSlotHeader* hdr = ch.recv_header(idx);
    while (hdr->seq.load(std::memory_order_acquire) != expected_gen) {
        cpu_relax();
    }

    uint64_t sent_bytes = hdr->nbytes;
    if (sent_bytes != nbytes) {
        ch.reader_done_ptr(ch.recv_half())->fetch_add(1, std::memory_order_release);
        throw std::runtime_error("shm_recv: expected " + std::to_string(nbytes) +
                                 " bytes but sender wrote " + std::to_string(sent_bytes));
    }

    memcpy_from_shm(tensor, ch.recv_data(idx), sent_bytes);

    ch.reader_done_ptr(ch.recv_half())->fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Non-blocking send
// ---------------------------------------------------------------------------

MpiRequest shm_isend(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch  = shm_get_channel(dest, nbytes);
    uint32_t    idx = sender_acquire_slot(ch);

    memcpy_to_shm(ch.send_data(idx), tensor, nbytes);
    ch.send_header(idx)->nbytes = nbytes;

    uint64_t gen = ch.send_state.next_gen++;
    ch.send_header(idx)->seq.store(gen, std::memory_order_release);

    ch.send_state.write_slot = (idx + 1) % SHM_NUM_SLOTS;

    MpiRequest req;
    req.shm_sig = encode_signal(gen, idx);

    int err = MPI_Isend(&req.shm_sig, sizeof(ShmSignal), MPI_BYTE, dest, SHM_CTRL_TAG,
                        MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_isend: MPI_Isend control signal failed: " +
                                 std::to_string(err));
    return req;
}

// ---------------------------------------------------------------------------
// Non-blocking recv
// ---------------------------------------------------------------------------

MpiRequest shm_irecv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    MpiRequest req;

    ShmSignal* sig_buf = new ShmSignal{};

    int err = MPI_Irecv(sig_buf, sizeof(ShmSignal), MPI_BYTE, source, SHM_CTRL_TAG,
                        MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS) {
        delete sig_buf;
        throw std::runtime_error("shm_irecv: MPI_Irecv control signal failed: " +
                                 std::to_string(err));
    }

    req.on_complete = [sig_buf, &ch, tensor, nbytes]() mutable {
        uint32_t idx          = decode_signal_slot(*sig_buf);
        uint64_t expected_gen = decode_signal_gen(*sig_buf);
        delete sig_buf;

        ShmSlotHeader* hdr = ch.recv_header(idx);

        while (hdr->seq.load(std::memory_order_acquire) != expected_gen) {
            cpu_relax();
        }

        uint64_t sent_bytes = hdr->nbytes;
        if (sent_bytes != nbytes) {
            ch.reader_done_ptr(ch.recv_half())->fetch_add(1, std::memory_order_release);
            throw std::runtime_error(
                "shm_irecv: expected " + std::to_string(nbytes) +
                " bytes but sender wrote " + std::to_string(sent_bytes));
        }

        memcpy_from_shm(tensor, ch.recv_data(idx), sent_bytes);

        ch.reader_done_ptr(ch.recv_half())->fetch_add(1, std::memory_order_release);
    };

    return req;
}
