#pragma once

#include "cuda_nn/type_traits.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cuda_nn {

enum class GemmEpilogue {
    None,
    Bias,
    ReluBias,
    GeluBias
};

struct CublasLtGemmConfig {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;

    GemmEpilogue epilogue = GemmEpilogue::None;

    // Used during algorithm selection.
    // The selected plan records the actual workspace size returned by cuBLASLt.
    size_t workspace_limit_bytes = 32ull * 1024ull * 1024ull;
};

// cublaslt 指定为行主序时, ReLU, dReLu, GELU, dGELU and Bias epilogue 不被支持
// 为更好的使用 cublaslt 本类只支持 Col Major 模式
// torch tensor 默认是行主序，使用时，请手动做 track ：D = A @ B + C ===> D^T = B^T @ A^T + C^T
template <typename T>
class CublasLtGemm {
public:
    explicit CublasLtGemm(const CublasLtGemmConfig& cfg,
                          int device_id = 0,
                          cudaStream_t stream = nullptr);

    CublasLtGemm(int64_t m,
                 int64_t n,
                 int64_t k,
                 int device_id = 0,
                 cudaStream_t stream = nullptr,
                 GemmEpilogue epilogue = GemmEpilogue::None);

    void set_stream(cudaStream_t stream);
    size_t workspace_size() const;

    // Column-major GEMM:
    //   D[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
    void run(const T* A,
             const T* B,
             const T* C,
             T* D,
             const void* alpha,
             const void* beta);

    /**
    * @brief 执行 GEMM 并加 bias：D = alpha * op(A) * op(B) + beta * C + bias
    *
    * bias 列向量，长度 N（广播到每行）
    */
    void run(const T* A,
             const T* B,
             const T* C,
             T* D,
             const T* bias,
             const void* alpha,
             const void* beta);

private:
    using Traits = CudaTypeTraits<T>;

    struct PlanEntry;

    CublasLtGemmConfig cfg_;
    int device_id_ = -1;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<PlanEntry> entry_;

    static CublasLtGemmConfig make_config(int64_t m,
                                          int64_t n,
                                          int64_t k,
                                          GemmEpilogue epilogue);

    static std::shared_ptr<PlanEntry> get_or_create_plan(int device_id, const CublasLtGemmConfig& cfg);
    static void build_entry(PlanEntry& e, const CublasLtGemmConfig& cfg);
    static void select_algo(PlanEntry& e, const CublasLtGemmConfig& cfg);

    void set_bias_pointer(const T* bias);
    void matmul(const T* A,
                const T* B,
                const T* C,
                T* D,
                const void* alpha,
                const void* beta,
                const T* bias = nullptr);
};

// Row-major adapter:
//   D_row[m,n] = alpha * A_row[m,k] * B_row[k,n] + beta * C_row[m,n]
// Internally:
//   D_col^T[n,m] = alpha * B_col^T[n,k] * A_col^T[k,m] + beta * C_col^T[n,m]
template <typename T>
class CublasLtRowMajorGemm {
public:
    CublasLtRowMajorGemm(int64_t m,
                         int64_t n,
                         int64_t k,
                         int device_id = 0,
                         cudaStream_t stream = nullptr,
                         GemmEpilogue epilogue = GemmEpilogue::None);

    void set_stream(cudaStream_t stream);
    size_t workspace_size() const;

    void run(const T* A,
             const T* B,
             const T* C,
             T* D,
             const void* alpha,
             const void* beta);

    // Row-major bias length is n, added to each output row.
    void run(const T* A,
             const T* B,
             const T* C,
             T* D,
             const T* bias,
             const void* alpha,
             const void* beta);

private:
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    CublasLtGemm<T> impl_;
};

extern template class CublasLtGemm<float>;
extern template class CublasLtGemm<double>;
extern template class CublasLtGemm<__half>;
extern template class CublasLtGemm<__nv_bfloat16>;

extern template class CublasLtRowMajorGemm<float>;
extern template class CublasLtRowMajorGemm<double>;
extern template class CublasLtRowMajorGemm<__half>;
extern template class CublasLtRowMajorGemm<__nv_bfloat16>;

}  // namespace cuda_nn
