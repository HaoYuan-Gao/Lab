#include "cuda_nn/cudnn_legacy_conv.h"
#include "cuda_nn/gpu_handle_pool.h"
#include "cuda_nn/gpu_workspace_pool.h"
#include "cuda_nn/inline_check.h"
#include "cuda_nn/type_traits.h"
#include "cuda_nn/device_guard.h"

#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cuda_nn {
namespace {

void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

template <typename T>
void hash_combine_value(std::size_t& seed, const T& value) {
    hash_combine(seed, std::hash<T>{}(value));
}

void hash_combine_vector(std::size_t& seed, const std::vector<int64_t>& values) {
    hash_combine_value(seed, values.size());
    for (auto value : values) {
        hash_combine_value(seed, value);
    }
}

std::size_t conv_cache_key_hash(const ConvConfig& cfg,
                                const std::string& dtype,
                                ConvKind kind) {
    std::size_t seed = 0;
    hash_combine_value(seed, dtype);
    hash_combine_value(seed, static_cast<int>(kind));
    hash_combine_vector(seed, cfg.x);
    hash_combine_vector(seed, cfg.w);
    hash_combine_vector(seed, cfg.padding);
    hash_combine_vector(seed, cfg.stride);
    hash_combine_vector(seed, cfg.dilation);
    hash_combine_vector(seed, cfg.output_padding);
    hash_combine_value(seed, cfg.groups);
    hash_combine_value(seed, static_cast<int>(cfg.memory_format));
    hash_combine_value(seed, cfg.with_bias);
    hash_combine_value(seed, static_cast<int>(cfg.activation));
    return seed;
}

template <typename Check>
void check_spatial_vector(const std::vector<int64_t>& values,
                          int64_t spatial_ndim,
                          const char* name,
                          Check check) {
    if (static_cast<int64_t>(values.size()) != spatial_ndim) {
        throw std::runtime_error(std::string(name) + " size must equal spatial_ndim.");
    }

    for (auto value : values) {
        if (!check(value)) {
            throw std::runtime_error(std::string(name) + " values are not valid.");
        }
    }
}

std::vector<int64_t> get_strides(const std::vector<int64_t>& dims,
                                 MemoryFormat memory_format) {
    const int64_t ndim = static_cast<int64_t>(dims.size());
    if (ndim <= 0) {
        throw std::runtime_error("Tensor dims cannot be empty.");
    }
    std::vector<int64_t> strides(ndim);

    switch (memory_format) {
        case MemoryFormat::Contiguous: {
            strides[ndim - 1] = 1;
            for (int64_t i = ndim - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            break;
        }
        case MemoryFormat::ChannelsLast: {
            if (ndim < 3) {
                throw std::runtime_error("ChannelsLast requires ndim >= 3.");
            }
            strides[1] = 1;
            strides[ndim - 1] = dims[1];
            for (int64_t i = ndim - 2; i >= 2; --i) {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            strides[0] = dims[1];
            for (int64_t i = 2; i < ndim; ++i) {
                strides[0] *= dims[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported memory format.");
    }

    return strides;
}

std::vector<int> to_int_vector(const std::vector<int64_t>& values, const char* name) {
    std::vector<int> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0 || values[i] > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(std::string(name) +
                                     " value is outside the range supported by legacy cuDNN API.");
        }
        result[i] = static_cast<int>(values[i]);
    }
    return result;
}

void validate_conv_config(const ConvConfig& cfg, ConvKind kind) {
    const int64_t spatial_ndim = cfg.spatial_ndim();

    if (cfg.w.size() != cfg.x.size()) {
        throw std::runtime_error("ConvConfig.x and ConvConfig.w must have the same ndim.");
    }
    if (cfg.groups <= 0) {
        throw std::runtime_error("ConvConfig.groups must be positive.");
    }
    if (cfg.x[0] <= 0 || cfg.x[1] <= 0 || cfg.w[0] <= 0 || cfg.w[1] <= 0) {
        throw std::runtime_error("ConvConfig.x and ConvConfig.w batch/channel dimensions must be positive.");
    }

    for (int64_t i = 0; i < spatial_ndim; ++i) {
        if (cfg.x[2 + i] <= 0 || cfg.w[2 + i] <= 0) {
            throw std::runtime_error("ConvConfig.x and ConvConfig.w spatial dimensions must be positive.");
        }
    }

    check_spatial_vector(cfg.padding, spatial_ndim, "padding", [] (int64_t value) { return value >= 0; });
    check_spatial_vector(cfg.stride, spatial_ndim, "stride", [] (int64_t value) { return value > 0; });
    check_spatial_vector(cfg.dilation, spatial_ndim, "dilation", [] (int64_t value) { return value > 0; });

    switch (kind) {
        case ConvKind::Forward:
            if (!cfg.output_padding.empty()) {
                throw std::runtime_error("output_padding is only valid for transpose convolution.");
            }
            if (cfg.x[1] != cfg.w[1] * cfg.groups) {
                throw std::runtime_error("Forward conv requires x[1] == w[1] * groups.");
            }
            if (cfg.w[0] % cfg.groups != 0) {
                throw std::runtime_error("Forward conv requires w[0] divisible by groups.");
            }
            if (cfg.w[0] <= 0) {
                throw std::runtime_error("Forward conv requires positive output channels w[0].");
            }
            break;
        case ConvKind::Transpose:
            if (!cfg.output_padding.empty()) {
                check_spatial_vector(cfg.output_padding, spatial_ndim, "output_padding", [] (int64_t value) { return value >= 0; });
                for (int64_t i = 0; i < spatial_ndim; ++i) {
                    if (cfg.output_padding[i] >= cfg.stride[i]) {
                        throw std::runtime_error("output_padding values must be smaller than stride values.");
                    }
                }
            }
            if (cfg.x[1] != cfg.w[0]) {
                throw std::runtime_error("Transpose conv requires x[1] == w[0].");
            }
            if (cfg.w[0] % cfg.groups != 0) {
                throw std::runtime_error("Transpose conv requires w[0] divisible by groups.");
            }
            if (cfg.w[1] * cfg.groups <= 0) {
                throw std::runtime_error("Transpose conv requires positive output channels w[1] * groups.");
            }
            break;
        default:
            throw std::runtime_error("Unsupported conv kind.");
    }
}

std::vector<int64_t> infer_conv_output_shape(const ConvConfig& cfg, ConvKind kind) {
    const int64_t spatial_ndim = cfg.spatial_ndim();
    std::vector<int64_t> y(2 + spatial_ndim);
    y[0] = cfg.x[0];
    y[1] = kind == ConvKind::Forward ? cfg.w[0] : cfg.w[1] * cfg.groups;

    for (int64_t i = 0; i < spatial_ndim; ++i) {
        const int64_t input = cfg.x[2 + i];
        const int64_t kernel = cfg.w[2 + i];
        const int64_t pad = cfg.padding[i];
        const int64_t stride = cfg.stride[i];
        const int64_t dilation = cfg.dilation[i];
        const int64_t out_pad = cfg.output_padding.empty() ? 0 : cfg.output_padding[i];

        switch (kind) {
            case ConvKind::Forward:
                y[2 + i] = (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
                break;
            case ConvKind::Transpose:
                y[2 + i] = (input - 1) * stride - 2 * pad + dilation * (kernel - 1) + out_pad + 1;
                break;
            default:
                throw std::runtime_error("Unsupported conv kind.");
        }

        if (y[2 + i] <= 0) {
            throw std::runtime_error("Inferred convolution output shape is invalid.");
        }
    }

    return y;
}

cudnnTensorFormat_t to_cudnn_tensor_format(MemoryFormat memory_format) {
    switch (memory_format) {
        case MemoryFormat::Contiguous:
            return CUDNN_TENSOR_NCHW;
        case MemoryFormat::ChannelsLast:
            return CUDNN_TENSOR_NHWC;
        default:
            throw std::runtime_error("Unsupported memory format.");
    }
}

cudnnActivationMode_t to_cudnn_activation_mode(Activation activation) {
    switch (activation) {
        case Activation::Relu:
            return CUDNN_ACTIVATION_RELU;
        case Activation::Tanh:
            return CUDNN_ACTIVATION_TANH;
        case Activation::Sigmoid:
            return CUDNN_ACTIVATION_SIGMOID;
        case Activation::Elu:
            return CUDNN_ACTIVATION_ELU;
        default:
            throw std::runtime_error("Unsupported activation for legacy cuDNN activation op.");
    }
}

template <typename T>
cudnnMathType_t legacy_math_type() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    } else {
        return CUDNN_DEFAULT_MATH;
    }

    // like cublasLtGemm: float using NVIDIA_TF32_OVERRIDE to enable TF32
    // if constexpr (std::is_same_v<T, float>) {
    //     return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    // }
}

void destroy_tensor_descriptor(cudnnTensorDescriptor_t desc) {
    if (desc != nullptr) {
        cudnnDestroyTensorDescriptor(desc);
    }
}

void destroy_filter_descriptor(cudnnFilterDescriptor_t desc) {
    if (desc != nullptr) {
        cudnnDestroyFilterDescriptor(desc);
    }
}

void destroy_convolution_descriptor(cudnnConvolutionDescriptor_t desc) {
    if (desc != nullptr) {
        cudnnDestroyConvolutionDescriptor(desc);
    }
}

void destroy_activation_descriptor(cudnnActivationDescriptor_t desc) {
    if (desc != nullptr) {
        cudnnDestroyActivationDescriptor(desc);
    }
}

}  // namespace

template <typename T, ConvKind Kind>
struct CudnnLegacyConvBase<T, Kind>::Entry {
    using TensorDescriptorPtr =
        std::unique_ptr<std::remove_pointer_t<cudnnTensorDescriptor_t>,
                        decltype(&destroy_tensor_descriptor)>;
    using FilterDescriptorPtr =
        std::unique_ptr<std::remove_pointer_t<cudnnFilterDescriptor_t>,
                        decltype(&destroy_filter_descriptor)>;
    using ConvolutionDescriptorPtr =
        std::unique_ptr<std::remove_pointer_t<cudnnConvolutionDescriptor_t>,
                        decltype(&destroy_convolution_descriptor)>;
    using ActivationDescriptorPtr =
        std::unique_ptr<std::remove_pointer_t<cudnnActivationDescriptor_t>,
                        decltype(&destroy_activation_descriptor)>;

    TensorDescriptorPtr x_desc{nullptr, destroy_tensor_descriptor};
    FilterDescriptorPtr w_desc{nullptr, destroy_filter_descriptor};
    TensorDescriptorPtr bias_desc{nullptr, destroy_tensor_descriptor};
    TensorDescriptorPtr y_desc{nullptr, destroy_tensor_descriptor};
    ConvolutionDescriptorPtr conv_desc{nullptr, destroy_convolution_descriptor};
    ActivationDescriptorPtr activation_desc{nullptr, destroy_activation_descriptor};

    std::vector<int64_t> y_shape;
    size_t workspace_size = 0;
    // Borrowed from CudnnHandlePool. Do not destroy here.
    cudnnHandle_t handle = nullptr;

    cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
};

template <typename T, ConvKind Kind>
CudnnLegacyConvBase<T, Kind>::CudnnLegacyConvBase(int device_id,
                                                  const ConvConfig& cfg,
                                                  cudaStream_t stream)
    : device_id_(device_id), cfg_(cfg), stream_(stream) {
    validate_conv_config(cfg_, Kind);
    entry_ = get_or_create_entry(device_id_, cfg_);
}

template <typename T, ConvKind Kind>
CudnnLegacyConvBase<T, Kind>::~CudnnLegacyConvBase() = default;

template <typename T, ConvKind Kind>
void CudnnLegacyConvBase<T, Kind>::set_stream(cudaStream_t stream) {
    stream_ = stream;
}

template <typename T, ConvKind Kind>
const std::vector<int64_t>& CudnnLegacyConvBase<T, Kind>::y_shape() const {
    return entry_->y_shape;
}

template <typename T, ConvKind Kind>
size_t CudnnLegacyConvBase<T, Kind>::workspace_size() const {
    return entry_->workspace_size;
}

template <typename T, ConvKind Kind>
void CudnnLegacyConvBase<T, Kind>::run(T* x, T* w, T* y) {
    execute(x, w, nullptr, y);
}

template <typename T, ConvKind Kind>
void CudnnLegacyConvBase<T, Kind>::run(T* x, T* w, T* bias, T* y) {
    execute(x, w, bias, y);
}

template <typename T, ConvKind Kind>
std::shared_ptr<typename CudnnLegacyConvBase<T, Kind>::Entry>
CudnnLegacyConvBase<T, Kind>::get_or_create_entry(int device_id,
                                                  const ConvConfig& cfg) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::size_t, std::shared_ptr<Entry>> cache;

    std::size_t key = conv_cache_key_hash(cfg, dtype_name<T>(), Kind);
    hash_combine_value(key, device_id);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    auto entry = std::make_shared<Entry>();

    // Algorithm selection and workspace-size query require a concrete cuDNN handle.
    // The handle itself is not stored in Entry; execution borrows a handle from the pool.
    auto lease = CudnnHandlePool::instance().acquire(device_id);
    entry->handle = lease.get();
    build_entry(*entry, entry->handle, cfg);

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

template <typename T, ConvKind Kind>
void CudnnLegacyConvBase<T, Kind>::build_entry(Entry& e,
                                               cudnnHandle_t handle,
                                               const ConvConfig& cfg) {
    using LegacyTraits = CudaTypeTraits<T>;

    e.y_shape = infer_conv_output_shape(cfg, Kind);
    const auto x_strides = get_strides(cfg.x, cfg.memory_format);
    const auto y_strides = get_strides(e.y_shape, cfg.memory_format);
    const auto bias_shape = make_bias_shape(e.y_shape);
    const auto bias_strides = get_strides(bias_shape, cfg.memory_format);

    cudnnTensorDescriptor_t raw_x_desc = nullptr;
    cudnnFilterDescriptor_t raw_w_desc = nullptr;
    cudnnTensorDescriptor_t raw_bias_desc = nullptr;
    cudnnTensorDescriptor_t raw_y_desc = nullptr;
    cudnnConvolutionDescriptor_t raw_conv_desc = nullptr;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&raw_x_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&raw_w_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&raw_bias_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&raw_y_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&raw_conv_desc));

    e.x_desc.reset(raw_x_desc);
    e.w_desc.reset(raw_w_desc);
    e.bias_desc.reset(raw_bias_desc);
    e.y_desc.reset(raw_y_desc);
    e.conv_desc.reset(raw_conv_desc);

    const auto x_dims_i = to_int_vector(cfg.x, "x dims");
    const auto x_strides_i = to_int_vector(x_strides, "x strides");
    const auto w_dims_i = to_int_vector(cfg.w, "w dims");
    const auto y_dims_i = to_int_vector(e.y_shape, "y dims");
    const auto y_strides_i = to_int_vector(y_strides, "y strides");
    const auto bias_dims_i = to_int_vector(bias_shape, "bias dims");
    const auto bias_strides_i = to_int_vector(bias_strides, "bias strides");
    const auto pad_i = to_int_vector(cfg.padding, "padding");
    const auto stride_i = to_int_vector(cfg.stride, "stride");
    const auto dilation_i = to_int_vector(cfg.dilation, "dilation");

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(e.x_desc.get(),
                                           LegacyTraits::cudnn_data_type,
                                           static_cast<int>(x_dims_i.size()),
                                           x_dims_i.data(),
                                           x_strides_i.data()));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(e.w_desc.get(),
                                           LegacyTraits::cudnn_data_type,
                                           to_cudnn_tensor_format(cfg.memory_format),
                                           static_cast<int>(w_dims_i.size()),
                                           w_dims_i.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(e.y_desc.get(),
                                           LegacyTraits::cudnn_data_type,
                                           static_cast<int>(y_dims_i.size()),
                                           y_dims_i.data(),
                                           y_strides_i.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(e.bias_desc.get(),
                                           LegacyTraits::cudnn_data_type,
                                           static_cast<int>(bias_dims_i.size()),
                                           bias_dims_i.data(),
                                           bias_strides_i.data()));
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(e.conv_desc.get(),
                                                static_cast<int>(pad_i.size()),
                                                pad_i.data(),
                                                stride_i.data(),
                                                dilation_i.data(),
                                                CUDNN_CROSS_CORRELATION,
                                                LegacyTraits::cudnn_compute_type));
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(e.conv_desc.get(),
                                              static_cast<int>(cfg.groups)));
    CHECK_CUDNN(cudnnSetConvolutionMathType(e.conv_desc.get(), legacy_math_type<T>()));

    if (cfg.activation != Activation::None) {
        cudnnActivationDescriptor_t raw_activation_desc = nullptr;
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&raw_activation_desc));
        e.activation_desc.reset(raw_activation_desc);
        CHECK_CUDNN(cudnnSetActivationDescriptor(e.activation_desc.get(),
                                                 to_cudnn_activation_mode(cfg.activation),
                                                 CUDNN_PROPAGATE_NAN,
                                                 0.0));
    }

    int returned_algo_count = 0;
    if constexpr (Kind == ConvKind::Forward) {
        cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                                           e.x_desc.get(),
                                                           e.w_desc.get(),
                                                           e.conv_desc.get(),
                                                           e.y_desc.get(),
                                                           CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                                           &returned_algo_count,
                                                           perf_results));
        if (returned_algo_count <= 0 || perf_results[0].status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("No valid legacy cuDNN forward convolution algorithm found.");
        }
        e.fwd_algo = perf_results[0].algo;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                            e.x_desc.get(),
                                                            e.w_desc.get(),
                                                            e.conv_desc.get(),
                                                            e.y_desc.get(),
                                                            e.fwd_algo,
                                                            &e.workspace_size));
    } else {
        cudnnConvolutionBwdDataAlgoPerf_t perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,
                                                                e.w_desc.get(),
                                                                e.x_desc.get(),
                                                                e.conv_desc.get(),
                                                                e.y_desc.get(),
                                                                CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                                                                &returned_algo_count,
                                                                perf_results));
        if (returned_algo_count <= 0 || perf_results[0].status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("No valid legacy cuDNN backward-data convolution algorithm found.");
        }
        e.bwd_data_algo = perf_results[0].algo;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                                 e.w_desc.get(),
                                                                 e.x_desc.get(),
                                                                 e.conv_desc.get(),
                                                                 e.y_desc.get(),
                                                                 e.bwd_data_algo,
                                                                 &e.workspace_size));
    }
}

template <typename T, ConvKind Kind>
void CudnnLegacyConvBase<T, Kind>::execute(T* x, T* w, T* bias, T* y) {
    using LegacyTraits = CudaTypeTraits<T>;

    if (cfg_.with_bias && bias == nullptr) {
        throw std::runtime_error("Conv was built with bias, but run() received a null bias pointer.");
    }

    if (!cfg_.with_bias && bias != nullptr) {
        throw std::runtime_error("Conv was built without bias, but run() received a bias pointer.");
    }

    auto handle_lease = CudnnHandlePool::instance().acquire_specific(device_id_, entry_->handle);
    cudnnHandle_t handle = handle_lease.get();

    DeviceGuard device_guard(device_id_);
    CHECK_CUDNN(cudnnSetStream(handle, stream_));

    auto workspace_lease = DeviceWorkspacePool::instance().acquire(device_id_, entry_->workspace_size);
    void* workspace = workspace_lease.ptr();

    const typename LegacyTraits::ScaleType alpha = static_cast<typename LegacyTraits::ScaleType>(1);
    const typename LegacyTraits::ScaleType beta = static_cast<typename LegacyTraits::ScaleType>(0);
    const typename LegacyTraits::ScaleType add_alpha = static_cast<typename LegacyTraits::ScaleType>(1);
    const typename LegacyTraits::ScaleType add_beta = static_cast<typename LegacyTraits::ScaleType>(1);

    if constexpr (Kind == ConvKind::Forward) {
        CHECK_CUDNN(cudnnConvolutionForward(handle,
                                            &alpha,
                                            entry_->x_desc.get(),
                                            x,
                                            entry_->w_desc.get(),
                                            w,
                                            entry_->conv_desc.get(),
                                            entry_->fwd_algo,
                                            workspace,
                                            entry_->workspace_size,
                                            &beta,
                                            entry_->y_desc.get(),
                                            y));
    } else {
        CHECK_CUDNN(cudnnConvolutionBackwardData(handle,
                                                 &alpha,
                                                 entry_->w_desc.get(),
                                                 w,
                                                 entry_->x_desc.get(),
                                                 x,
                                                 entry_->conv_desc.get(),
                                                 entry_->bwd_data_algo,
                                                 workspace,
                                                 entry_->workspace_size,
                                                 &beta,
                                                 entry_->y_desc.get(),
                                                 y));
    }

    if (cfg_.with_bias) {
        CHECK_CUDNN(cudnnAddTensor(handle,
                                   &add_alpha,
                                   entry_->bias_desc.get(),
                                   bias,
                                   &add_beta,
                                   entry_->y_desc.get(),
                                   y));
    }

    if (cfg_.activation != Activation::None) {
        CHECK_CUDNN(cudnnActivationForward(handle,
                                           entry_->activation_desc.get(),
                                           &alpha,
                                           entry_->y_desc.get(),
                                           y,
                                           &beta,
                                           entry_->y_desc.get(),
                                           y));
    }
}

template <typename T, ConvKind Kind>
std::vector<int64_t> CudnnLegacyConvBase<T, Kind>::make_bias_shape(
    const std::vector<int64_t>& y_shape) {
    std::vector<int64_t> bias_shape(y_shape.size(), 1);
    bias_shape[1] = y_shape[1];
    return bias_shape;
}

template class CudnnLegacyConvBase<float, ConvKind::Forward>;
template class CudnnLegacyConvBase<double, ConvKind::Forward>;
template class CudnnLegacyConvBase<__half, ConvKind::Forward>;
template class CudnnLegacyConvBase<__nv_bfloat16, ConvKind::Forward>;

template class CudnnLegacyConvBase<float, ConvKind::Transpose>;
template class CudnnLegacyConvBase<double, ConvKind::Transpose>;
template class CudnnLegacyConvBase<__half, ConvKind::Transpose>;
template class CudnnLegacyConvBase<__nv_bfloat16, ConvKind::Transpose>;

}  // namespace cuda_nn
