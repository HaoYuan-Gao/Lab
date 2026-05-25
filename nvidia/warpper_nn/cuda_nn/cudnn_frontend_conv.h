#pragma once

#include "cuda_nn/type_traits.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuda_nn {

enum class Activation {
    None,
    Relu,
    Tanh,
    Sigmoid,
    Elu
};

enum class ConvKind {
    Forward,
    Transpose
};

enum class MemoryFormat {
    Contiguous,
    ChannelsLast
};

struct ConvConfig {
    // cfg.x and cfg.w are always logical dims in NCHW-style order.
    // MemoryFormat only controls tensor strides, not dim order.
    std::vector<int64_t> x;
    std::vector<int64_t> w;
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    std::vector<int64_t> output_padding;
    int64_t groups = 1;
    MemoryFormat memory_format = MemoryFormat::Contiguous;
    bool with_bias = false;
    Activation activation = Activation::None;

    int64_t spatial_ndim() const;
};

template <typename T, ConvKind Kind>
class CudnnFrontendConvBase {
public:
    CudnnFrontendConvBase(int device_id,
                          const ConvConfig& cfg,
                          cudaStream_t stream = nullptr);
    ~CudnnFrontendConvBase();

    void set_stream(cudaStream_t stream);
    const std::vector<int64_t>& y_shape() const;
    size_t workspace_size() const;
    
    // y = conv(x, w)
    void run(T* x, T* w, T* y);

    // y = conv(x, w) + bias
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
using CudnnFrontendConv = CudnnFrontendConvBase<T, ConvKind::Forward>;

template <typename T>
using CudnnFrontendConvTranspose = CudnnFrontendConvBase<T, ConvKind::Transpose>;

extern template class CudnnFrontendConvBase<float, ConvKind::Forward>;
extern template class CudnnFrontendConvBase<double, ConvKind::Forward>;
extern template class CudnnFrontendConvBase<__half, ConvKind::Forward>;
extern template class CudnnFrontendConvBase<__nv_bfloat16, ConvKind::Forward>;

extern template class CudnnFrontendConvBase<float, ConvKind::Transpose>;
extern template class CudnnFrontendConvBase<double, ConvKind::Transpose>;
extern template class CudnnFrontendConvBase<__half, ConvKind::Transpose>;
extern template class CudnnFrontendConvBase<__nv_bfloat16, ConvKind::Transpose>;

}  // namespace cuda_nn
