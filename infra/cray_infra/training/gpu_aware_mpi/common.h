#pragma once

#include <mpi.h>
#include <torch/extension.h>
#include <torch/cuda.h>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// MPI lifecycle
// ---------------------------------------------------------------------------
inline bool& mpi_initialized_flag() {
    static bool flag = false;
    return flag;
}

inline void ensure_mpi_initialized() {
    if (!mpi_initialized_flag()) {
        MPI_Init(nullptr, nullptr);
        mpi_initialized_flag() = true;
    }
}

// ---------------------------------------------------------------------------
// CUDA sync helper
// ---------------------------------------------------------------------------
inline void sync_cuda_if_needed(const torch::Tensor& tensor) {
    if (tensor.is_cuda()) {
        torch::cuda::synchronize(tensor.device().index());
    }
}

// ---------------------------------------------------------------------------
// MPI datatype mapping
// ---------------------------------------------------------------------------
inline std::tuple<MPI_Datatype, size_t> get_typesize(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:  return {MPI_FLOAT,          sizeof(float)};
        case torch::kFloat64:  return {MPI_DOUBLE,         sizeof(double)};
        case torch::kFloat16:  return {MPI_SHORT,          sizeof(int16_t)};
        case torch::kBFloat16: return {MPI_SHORT,          sizeof(int16_t)};
        case torch::kInt32:    return {MPI_INT,            sizeof(int32_t)};
        case torch::kInt64:    return {MPI_LONG_LONG,      sizeof(int64_t)};
        case torch::kUInt8:    return {MPI_UNSIGNED_CHAR,  sizeof(uint8_t)};
        case torch::kInt8:     return {MPI_CHAR,           sizeof(int8_t)};
        default:
            throw std::runtime_error("Unsupported torch::ScalarType for MPI communication");
    }
}

inline MPI_Datatype get_mpi_datatype(const torch::Tensor& tensor) {
    return std::get<0>(get_typesize(tensor.scalar_type()));
}

// ---------------------------------------------------------------------------
// ShmSignal
//
// 16-byte payload sent via MPI_Isend/MPI_Irecv as the "data ready" ping on
// the shm fast path.  Carries the generation counter and the slot index so
// the receiver can spin-wait on the exact slot/generation it expects.  The
// generation counter lets the sender pipeline (double-buffered slots in
// shm_channel.h) without ambiguity about which write the signal refers to,
// and closes the memory-visibility window where an MPI signal could be
// observed before the memcpy it guards becomes visible on the peer's core.
//
// Fixed 16 bytes so the on-wire representation matches a primitive type
// group the MPI implementation will always treat as eager.
// ---------------------------------------------------------------------------
struct ShmSignal {
    uint64_t gen;        // generation the receiver must spin-wait for (slot.seq == gen)
    uint32_t slot_idx;   // which of the channel's slots the sender wrote
    uint32_t _pad{0};    // pad to 16 bytes; explicit so layout is stable
};
static_assert(sizeof(ShmSignal) == 16, "ShmSignal must be 16 bytes on the wire");

// ---------------------------------------------------------------------------
// Async request handle
//
// Both the standard MPI path and the shm path return an MpiRequest.
//
// Standard MPI path:
//   mpi_req holds a real MPI_Request; on_complete is empty.
//
// SHM path:
//   shm_isend: copies data into shm, then posts MPI_Isend for the ShmSignal.
//              mpi_req holds that Isend handle.  on_complete is empty — the
//              sender has nothing to do at wait time.
//   shm_irecv: posts MPI_Irecv for the ShmSignal.  on_complete holds a
//              closure that (a) spin-waits for slot.seq == signal.gen, then
//              (b) memcpys the payload out of shm, then (c) bumps the
//              receiver-side reader_done counter to unblock the sender's
//              backpressure on the next reuse of this slot.
//
// shm_sig provides stable storage for the ShmSignal buffer required by
// MPI_Isend/MPI_Irecv (the buffer must outlive the MPI operation).  Because
// an in-flight MPI_Request must not be copied behind MPI's back, MpiRequest
// is move-only — the deleted copy operators make accidental copies a
// compile-time error rather than a runtime double-wait.
// ---------------------------------------------------------------------------
struct MpiRequest {
    MPI_Request           mpi_req    = MPI_REQUEST_NULL;
    bool                  completed  = false;
    ShmSignal             shm_sig{};       // stable MPI buffer for the shm signal
    std::function<void()> on_complete;      // called once by mpi_wait after MPI_Wait

    MpiRequest() = default;
    MpiRequest(MpiRequest&&) noexcept = default;
    MpiRequest& operator=(MpiRequest&&) noexcept = default;
    MpiRequest(const MpiRequest&) = delete;
    MpiRequest& operator=(const MpiRequest&) = delete;
};

inline void mpi_wait(MpiRequest& req) {
    if (req.completed) return;
    MPI_Status st;
    int err = MPI_Wait(&req.mpi_req, &st);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Wait failed: " + std::to_string(err));
    if (req.on_complete) req.on_complete();
    req.completed = true;
}

inline void mpi_waitall(std::vector<MpiRequest>& reqs) {
    // Collect handles for all still-pending requests, preserving index mapping
    // so we can run each request's callback after MPI_Waitall completes.
    std::vector<MPI_Request> pending;
    std::vector<size_t>      indices;
    for (size_t i = 0; i < reqs.size(); ++i) {
        if (!reqs[i].completed) {
            pending.push_back(reqs[i].mpi_req);
            indices.push_back(i);
        }
    }
    if (pending.empty()) return;

    std::vector<MPI_Status> statuses(pending.size());
    int err = MPI_Waitall(static_cast<int>(pending.size()), pending.data(), statuses.data());
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Waitall failed: " + std::to_string(err));

    // All MPI ops have landed; run completion callbacks in posting order.
    for (size_t idx : indices) {
        if (reqs[idx].on_complete) reqs[idx].on_complete();
        reqs[idx].completed = true;
    }
}
