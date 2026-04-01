#pragma once

#include <mpi.h>
#include <torch/extension.h>
#include <torch/cuda.h>
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
        case torch::kFloat32:  return {MPI_FLOAT,         sizeof(float)};
        case torch::kFloat64:  return {MPI_DOUBLE,        sizeof(double)};
        case torch::kFloat16:  return {MPI_SHORT,         sizeof(int16_t)};
        case torch::kBFloat16: return {MPI_SHORT,         sizeof(int16_t)};
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
// Async request handle (wraps MPI_Request or a completed shm op)
// ---------------------------------------------------------------------------
struct MpiRequest {
    MPI_Request mpi_req = MPI_REQUEST_NULL;
    bool        completed = false;  // true if the op already finished (e.g. shm path)
};

inline void mpi_wait(MpiRequest& req) {
    if (req.completed) return;
    MPI_Status st;
    int err = MPI_Wait(&req.mpi_req, &st);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Wait failed: " + std::to_string(err));
    req.completed = true;
}

inline void mpi_waitall(std::vector<MpiRequest>& reqs) {
    // Collect the non-completed requests and wait on them in one call
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
    for (size_t idx : indices) reqs[idx].completed = true;
}
