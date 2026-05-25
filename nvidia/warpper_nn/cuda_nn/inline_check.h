#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cudnn.h>

#include <sstream>
#include <stdexcept>
#include <string>

#define GTENSOR_RED    "\033[1;31m"
#define GTENSOR_RESET  "\033[0m"

#define CHECK_CUDA(expr)                                                     \
    do {                                                                     \
        cudaError_t status__ = (expr);                                       \
        if (status__ != cudaSuccess) {                                       \
            std::ostringstream oss__;                                        \
            oss__                                                            \
                << GTENSOR_RED                                               \
                << "CUDA error\n"                                            \
                << "  location  : " << __FILE__ << ":" << __LINE__ << "\n"   \
                << "  expression: " << #expr << "\n"                         \
                << "  status    : " << cudaGetErrorString(status__)          \
                << GTENSOR_RESET;                                            \
            throw std::runtime_error(oss__.str());                           \
        }                                                                    \
    } while (0)

#define CHECK_CUBLASLT(expr)                                                 \
    do {                                                                     \
        cublasStatus_t status__ = (expr);                                    \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                             \
            std::ostringstream oss__;                                        \
            oss__                                                            \
                << GTENSOR_RED                                               \
                << "cuBLASLt error\n"                                        \
                << "  location  : " << __FILE__ << ":" << __LINE__ << "\n"   \
                << "  expression: " << #expr << "\n"                         \
                << "  status    : " << cublasGetStatusString(status__)      \
                << GTENSOR_RESET;                                            \
            throw std::runtime_error(oss__.str());                           \
        }                                                                    \
    } while (0)

#define CHECK_CUDNN(expr)                                                    \
    do {                                                                     \
        cudnnStatus_t status__ = (expr);                                     \
        if (status__ != CUDNN_STATUS_SUCCESS) {                              \
            std::ostringstream oss__;                                        \
            oss__                                                            \
                << GTENSOR_RED                                               \
                << "cuDNN error\n"                                           \
                << "  location  : " << __FILE__ << ":" << __LINE__ << "\n"   \
                << "  expression: " << #expr << "\n"                         \
                << "  status    : " << cudnnGetErrorString(status__)        \
                << GTENSOR_RESET;                                            \
            throw std::runtime_error(oss__.str());                           \
        }                                                                    \
    } while (0)
