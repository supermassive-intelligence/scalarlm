#pragma once

#include <mpi.h>
#include <torch/extension.h>
#include <torch/cuda.h>
#include <stdexcept>
#include <tuple>
#include <string>

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
