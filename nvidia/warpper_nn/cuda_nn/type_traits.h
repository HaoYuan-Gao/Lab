#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cudnn_frontend.h>

#include <type_traits>
#include <string>

namespace cuda_nn {

template <typename T>
struct CudaTypeTraits;

// TF32 is controlled globally by NVIDIA_TF32_OVERRIDE.
// Example:
//   export NVIDIA_TF32_OVERRIDE=0  // disable TF32 globally
//   export NVIDIA_TF32_OVERRIDE=1  // allow TF32 globally

template <>
struct CudaTypeTraits<float> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_32F;
    static constexpr cudaDataType_t scale_type = CUDA_R_32F;
    static constexpr cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F; // CUBLAS_COMPUTE_32F_FAST_TF32
    static constexpr cudnn_frontend::DataType_t frontend_data_type = cudnn_frontend::DataType_t::FLOAT;
    static constexpr cudnn_frontend::DataType_t frontend_compute_type = cudnn_frontend::DataType_t::FLOAT;
    static const char* name() { return "float"; }
};

template <>
struct CudaTypeTraits<double> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_64F;
    static constexpr cudaDataType_t scale_type = CUDA_R_64F;
    static constexpr cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_64F;
    static constexpr cudnn_frontend::DataType_t frontend_data_type = cudnn_frontend::DataType_t::DOUBLE;
    static constexpr cudnn_frontend::DataType_t frontend_compute_type = cudnn_frontend::DataType_t::DOUBLE;
    static const char* name() { return "double"; }
};

template <>
struct CudaTypeTraits<__half> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16F;
    static constexpr cudaDataType_t scale_type = CUDA_R_32F;
    static constexpr cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;
    static constexpr cudnn_frontend::DataType_t frontend_data_type = cudnn_frontend::DataType_t::HALF;
    static constexpr cudnn_frontend::DataType_t frontend_compute_type = cudnn_frontend::DataType_t::FLOAT;
    static const char* name() { return "half"; }
};

template <>
struct CudaTypeTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16BF;
    static constexpr cudaDataType_t scale_type = CUDA_R_32F;
    static constexpr cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;
    static constexpr cudnn_frontend::DataType_t frontend_data_type = cudnn_frontend::DataType_t::BFLOAT16;
    static constexpr cudnn_frontend::DataType_t frontend_compute_type = cudnn_frontend::DataType_t::FLOAT;
    static const char* name() { return "bf16"; }
};

template <typename T>
inline std::string dtype_name() {
    return CudaTypeTraits<T>::name();
}

}  // namespace cuda_nn
