#include "cuda_nn/cublaslt_gemm.h"

#include "cuda_nn/device_guard.h"
#include "cuda_nn/inline_check.h"
#include "cuda_nn/gpu_handle_pool.h"
#include "cuda_nn/gpu_workspace_pool.h"

#include <functional>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuda_nn {
namespace {

template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

bool epilogue_has_bias(GemmEpilogue epilogue) {
    return epilogue == GemmEpilogue::Bias ||
           epilogue == GemmEpilogue::ReluBias ||
           epilogue == GemmEpilogue::GeluBias;
}

cublasLtEpilogue_t to_cublas_epilogue(GemmEpilogue epilogue) {
    switch (epilogue) {
        case GemmEpilogue::None:
            return CUBLASLT_EPILOGUE_DEFAULT;
        case GemmEpilogue::Bias:
            return CUBLASLT_EPILOGUE_BIAS;
        case GemmEpilogue::ReluBias:
            return CUBLASLT_EPILOGUE_RELU_BIAS;
        case GemmEpilogue::GeluBias:
            return CUBLASLT_EPILOGUE_GELU_BIAS;
    }

    return CUBLASLT_EPILOGUE_DEFAULT;
}

void check_gemm_config(const CublasLtGemmConfig& cfg) {
    if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
        throw std::runtime_error("GEMM m/n/k must be positive.");
    }
}

std::size_t gemm_cache_key_hash(const CublasLtGemmConfig& cfg,
                                const std::string& dtype,
                                int device_id) {
    std::size_t seed = 0;
    hash_combine(seed, dtype);
    hash_combine(seed, device_id);
    hash_combine(seed, cfg.m);
    hash_combine(seed, cfg.n);
    hash_combine(seed, cfg.k);
    hash_combine(seed, static_cast<int>(cfg.epilogue));
    return seed;
}

}  // namespace

template <typename T>
struct CublasLtGemm<T>::PlanEntry {
    ~PlanEntry() {
        if (layout_d) {
            cublasLtMatrixLayoutDestroy(layout_d);
        }
        if (layout_c) {
            cublasLtMatrixLayoutDestroy(layout_c);
        }
        if (layout_b) {
            cublasLtMatrixLayoutDestroy(layout_b);
        }
        if (layout_a) {
            cublasLtMatrixLayoutDestroy(layout_a);
        }
        if (matmul_desc) {
            cublasLtMatmulDescDestroy(matmul_desc);
        }
    }

    // Borrowed from CublasLtHandlePool. Do not destroy here.
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_c = nullptr;
    cublasLtMatrixLayout_t layout_d = nullptr;
    cublasLtMatmulHeuristicResult_t best_algo{};

    // Borrowed from DeviceWorkspacePool at runtime. Do not free here.
    size_t workspace_size = 0;
};

template <typename T>
CublasLtGemm<T>::CublasLtGemm(const CublasLtGemmConfig& cfg,
                              int device_id,
                              cudaStream_t stream)
    : cfg_(cfg),
      device_id_(device_id),
      stream_(stream) {
    check_gemm_config(cfg_);
    entry_ = get_or_create_plan(device_id_, cfg_);
}

template <typename T>
CublasLtGemm<T>::CublasLtGemm(int64_t m,
                              int64_t n,
                              int64_t k,
                              int device_id,
                              cudaStream_t stream,
                              GemmEpilogue epilogue)
    : CublasLtGemm(make_config(m, n, k, epilogue), device_id, stream) {}

template <typename T>
void CublasLtGemm<T>::set_stream(cudaStream_t stream) {
    stream_ = stream;
}

template <typename T>
size_t CublasLtGemm<T>::workspace_size() const {
    return entry_->workspace_size;
}

template <typename T>
void CublasLtGemm<T>::run(const T* A,
                          const T* B,
                          const T* C,
                          T* D,
                          const void* alpha,
                          const void* beta) {
    if (cfg_.epilogue != GemmEpilogue::None) {
        throw std::runtime_error(
            "This GEMM was built with bias epilogue.");
    }

    matmul(A, B, C, D, alpha, beta);
}

template <typename T>
void CublasLtGemm<T>::run(const T* A,
                          const T* B,
                          const T* C,
                          T* D,
                          const T* bias,
                          const void* alpha,
                          const void* beta) {
    if (!epilogue_has_bias(cfg_.epilogue)) {
        throw std::runtime_error("This GEMM was built without bias epilogue.");
    }

    matmul(A, B, C, D, alpha, beta, bias);
}

template <typename T>
CublasLtGemmConfig CublasLtGemm<T>::make_config(int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                GemmEpilogue epilogue) {
    CublasLtGemmConfig cfg;
    cfg.m = m;
    cfg.n = n;
    cfg.k = k;
    cfg.epilogue = epilogue;
    return cfg;
}

template <typename T>
std::shared_ptr<typename CublasLtGemm<T>::PlanEntry>
CublasLtGemm<T>::get_or_create_plan(int device_id, const CublasLtGemmConfig& cfg) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::size_t, std::shared_ptr<PlanEntry>> cache;

    std::size_t key = gemm_cache_key_hash(cfg, dtype_name<T>(), device_id);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    auto entry = std::make_shared<PlanEntry>();

    // Build must use a concrete cuBLASLt handle. The Lease keeps the handle
    // exclusive during plan construction, then releases it automatically.
    auto lease = CublasLtHandlePool::instance().acquire(device_id);
    entry->handle = lease.get();

    build_entry(*entry, cfg);

    // RE-CHECK: The KEY must be unique.
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        cache.emplace(key, entry);
    }

    return entry;
}

template <typename T>
void CublasLtGemm<T>::build_entry(PlanEntry& e, const CublasLtGemmConfig& cfg) {
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(
        &e.matmul_desc,
        Traits::cublas_compute_type,
        Traits::scale_type));

    cublasOperation_t op_n = CUBLAS_OP_N;

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        e.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &op_n,
        sizeof(op_n)));

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        e.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &op_n,
        sizeof(op_n)));

    // Fixed column-major layouts:
    // A[m,k], B[k,n], C[m,n], D[m,n]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &e.layout_a,
        Traits::cuda_type,
        cfg.m,
        cfg.k,
        cfg.m));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &e.layout_b,
        Traits::cuda_type,
        cfg.k,
        cfg.n,
        cfg.k));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &e.layout_c,
        Traits::cuda_type,
        cfg.m,
        cfg.n,
        cfg.m));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &e.layout_d,
        Traits::cuda_type,
        cfg.m,
        cfg.n,
        cfg.m));

    cublasLtOrder_t order = CUBLASLT_ORDER_COL;

    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
        e.layout_a,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
        e.layout_b,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
        e.layout_c,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
        e.layout_d,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)));

    cublasLtEpilogue_t epilogue = to_cublas_epilogue(cfg.epilogue);
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        e.matmul_desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue)
    ));

    select_algo(e, cfg);
}

template <typename T>
void CublasLtGemm<T>::select_algo(PlanEntry& e, const CublasLtGemmConfig& cfg) {
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));

    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cfg.workspace_limit_bytes,
        sizeof(cfg.workspace_limit_bytes)));

    constexpr int kMaxHeuristics = 32;
    std::vector<cublasLtMatmulHeuristicResult_t> heurs(kMaxHeuristics);
    int returned = 0;

    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
        e.handle,
        e.matmul_desc,
        e.layout_a,
        e.layout_b,
        e.layout_c,
        e.layout_d,
        preference,
        kMaxHeuristics,
        heurs.data(),
        &returned));

    CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));

    for (int i = 0; i < returned; ++i) {
        if (heurs[i].state == CUBLAS_STATUS_SUCCESS && 
            heurs[i].workspaceSize <= cfg.workspace_limit_bytes) {
            e.best_algo = heurs[i];
            e.workspace_size = heurs[i].workspaceSize;
            return;
        }
    }

    throw std::runtime_error("cuBLASLt failed to find a suitable GEMM algorithm.");
}

template <typename T>
void CublasLtGemm<T>::set_bias_pointer(const T* bias) {
    const void* bias_ptr = static_cast<const void*>(bias);

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        entry_->matmul_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_ptr,
        sizeof(bias_ptr)));
}

template <typename T>
void CublasLtGemm<T>::matmul(const T* A,
                             const T* B,
                             const T* C,
                             T* D,
                             const void* alpha,
                             const void* beta,
                             const T* bias) {
    // The plan was selected with entry_->handle, so execute with the same handle.
    // acquire_specific() guarantees no other thread is using this handle/plan at the same time.
    auto lease = CublasLtHandlePool::instance().acquire_specific(device_id_, entry_->handle);
    cublasLtHandle_t handle = lease.get();

    DeviceGuard device_guard(device_id_);

    if (bias != nullptr) {
        set_bias_pointer(bias);
    }

    auto workspace_lease = DeviceWorkspacePool::instance().acquire(device_id_, entry_->workspace_size);
    void* workspace = workspace_lease.ptr();

    CHECK_CUBLASLT(cublasLtMatmul(
        handle,
        entry_->matmul_desc,
        alpha,
        A,
        entry_->layout_a,
        B,
        entry_->layout_b,
        beta,
        C,
        entry_->layout_c,
        D,
        entry_->layout_d,
        &entry_->best_algo.algo,
        workspace,
        entry_->workspace_size,
        stream_));
}

template <typename T>
CublasLtRowMajorGemm<T>::CublasLtRowMajorGemm(int64_t m,
                                              int64_t n,
                                              int64_t k,
                                              int device_id,
                                              cudaStream_t stream,
                                              GemmEpilogue epilogue)
    : m_(m),
      n_(n),
      k_(k),
      impl_(n, m, k, device_id, stream, epilogue) {}

template <typename T>
void CublasLtRowMajorGemm<T>::set_stream(cudaStream_t stream) {
    impl_.set_stream(stream);
}

template <typename T>
size_t CublasLtRowMajorGemm<T>::workspace_size() const {
    return impl_.workspace_size();
}

template <typename T>
void CublasLtRowMajorGemm<T>::run(const T* A,
                                  const T* B,
                                  const T* C,
                                  T* D,
                                  const void* alpha,
                                  const void* beta) {
    (void)m_;
    (void)n_;
    (void)k_;

    impl_.run(B, A, C, D, alpha, beta);
}

template <typename T>
void CublasLtRowMajorGemm<T>::run(const T* A,
                                  const T* B,
                                  const T* C,
                                  T* D,
                                  const T* bias,
                                  const void* alpha,
                                  const void* beta) {
    impl_.run(B, A, C, D, bias, alpha, beta);
}

template class CublasLtGemm<float>;
template class CublasLtGemm<double>;
template class CublasLtGemm<__half>;
template class CublasLtGemm<__nv_bfloat16>;

template class CublasLtRowMajorGemm<float>;
template class CublasLtRowMajorGemm<double>;
template class CublasLtRowMajorGemm<__half>;
template class CublasLtRowMajorGemm<__nv_bfloat16>;

}  // namespace cuda_nn
