#pragma once

#include "cuda_nn/type_traits.h"
#include "cuda_nn/cudnn_conv_defs.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuda_nn {

template <typename T, ConvKind Kind>
class CudnnLegacyConvBase {
public:
    CudnnLegacyConvBase(int device_id,
                        const ConvConfig& cfg,
                        cudaStream_t stream = nullptr);
    ~CudnnLegacyConvBase();

    void set_stream(cudaStream_t stream);
    const std::vector<int64_t>& y_shape() const;
    size_t workspace_size() const;

    // y = conv(x, w)
    void run(T* x, T* w, T* y);

    // y = conv(x, w) + bias, optionally followed by activation according to cfg.
    void run(T* x, T* w, T* bias, T* y);

private:
    using Traits = CudaTypeTraits<T>;

    struct Entry;

    int device_id_ = -1;
    ConvConfig cfg_;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<Entry> entry_;

    static std::shared_ptr<Entry> get_or_create_entry(int device_id,
                                                      const ConvConfig& cfg);
    static void build_entry(Entry& e, cudnnHandle_t handle, const ConvConfig& cfg);
    static std::vector<int64_t> make_bias_shape(const std::vector<int64_t>& y_shape);

    void execute(T* x, T* w, T* bias, T* y);
};

template <typename T>
using CudnnLegacyConv = CudnnLegacyConvBase<T, ConvKind::Forward>;

template <typename T>
using CudnnLegacyConvTranspose = CudnnLegacyConvBase<T, ConvKind::Transpose>;

extern template class CudnnLegacyConvBase<float, ConvKind::Forward>;
extern template class CudnnLegacyConvBase<double, ConvKind::Forward>;
extern template class CudnnLegacyConvBase<__half, ConvKind::Forward>;
extern template class CudnnLegacyConvBase<__nv_bfloat16, ConvKind::Forward>;

extern template class CudnnLegacyConvBase<float, ConvKind::Transpose>;
extern template class CudnnLegacyConvBase<double, ConvKind::Transpose>;
extern template class CudnnLegacyConvBase<__half, ConvKind::Transpose>;
extern template class CudnnLegacyConvBase<__nv_bfloat16, ConvKind::Transpose>;

}  // namespace cuda_nn
