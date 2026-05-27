#include "cuda_nn/cudnn_frontend_conv.h"
#include "cuda_nn/inline_check.h"
#include "cuda_nn/gpu_handle_pool.h"
#include "cuda_nn/gpu_workspace_pool.h"
#include "cuda_nn/device_guard.h"

#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace cuda_nn {
namespace {

void check_frontend_error(cudnn_frontend::error_t status, const char* expr, const char* file, int line) {
    if (status.is_bad()) {
        throw std::runtime_error(
            std::string("\033[31m") +
            std::string("[cuDNN Frontend Error]\n") +
            "Expression : " + expr + "\n" +
            "File       : " + file + ":" + std::to_string(line) + "\n" +
            "Message    : " + status.get_message() + 
            "\033[0m");
    }
}

// only support frontend convolution
#define CHECK_CUDNN_FE(expr) check_frontend_error((expr), #expr, __FILE__, __LINE__)

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

template<typename Check>
void check_spatial_vector(const std::vector<int64_t>& values, int64_t spatial_ndim, const char* name, Check check) {
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

cudnn_frontend::PointwiseMode_t to_frontend_pointwise_mode(Activation activation) {
    switch (activation) {
        case Activation::Relu:
            return cudnn_frontend::PointwiseMode_t::RELU_FWD;
        case Activation::Tanh:
            return cudnn_frontend::PointwiseMode_t::TANH_FWD;
        case Activation::Sigmoid:
            return cudnn_frontend::PointwiseMode_t::SIGMOID_FWD;
        case Activation::Elu:
            return cudnn_frontend::PointwiseMode_t::ELU_FWD;
        default:
            throw std::runtime_error("Unsupported activation for cuDNN frontend pointwise op.");
    }
}

}  // namespace

template <typename T, ConvKind Kind>
struct CudnnFrontendConvBase<T, Kind>::Entry {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> x_tensor;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> w_tensor;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> bias_tensor;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> y_tensor;
    std::vector<int64_t> y_shape;

    // Borrowed from CudnnWorkspacePool. Do not free here.
    size_t workspace_size = 0;
    // Borrowed from CudnnHandlePool. Do not destroy here.
    cudnnHandle_t handle = nullptr;
};

template <typename T, ConvKind Kind>
CudnnFrontendConvBase<T, Kind>::CudnnFrontendConvBase(int device_id,
                                                      const ConvConfig& cfg,
                                                      cudaStream_t stream)
    : device_id_(device_id), cfg_(cfg), stream_(stream) {
    validate_conv_config(cfg_, Kind);
    entry_ = get_or_create_entry(device_id_, cfg_);
}

template <typename T, ConvKind Kind>
CudnnFrontendConvBase<T, Kind>::~CudnnFrontendConvBase() = default;

template <typename T, ConvKind Kind>
void CudnnFrontendConvBase<T, Kind>::set_stream(cudaStream_t stream) {
    stream_ = stream;
}

template <typename T, ConvKind Kind>
const std::vector<int64_t>& CudnnFrontendConvBase<T, Kind>::y_shape() const {
    return entry_->y_shape;
}

template <typename T, ConvKind Kind>
size_t CudnnFrontendConvBase<T, Kind>::workspace_size() const {
    return entry_->workspace_size;
}

template <typename T, ConvKind Kind>
void CudnnFrontendConvBase<T, Kind>::run(T* x, T* w, T* y) {
    execute(x, w, nullptr, y);
}

template <typename T, ConvKind Kind>
void CudnnFrontendConvBase<T, Kind>::run(T* x, T* w, T* bias, T* y) {
    execute(x, w, bias, y);
}

template <typename T, ConvKind Kind>
std::shared_ptr<typename CudnnFrontendConvBase<T, Kind>::Entry>
CudnnFrontendConvBase<T, Kind>::get_or_create_entry(int device_id,
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

    // Build must use a concrete cuDNN handle. The Lease keeps the handle exclusive
    // during graph construction, then releases it automatically when leaving scope.
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
void CudnnFrontendConvBase<T, Kind>::build_entry(Entry& e,
                                                 cudnnHandle_t handle,
                                                 const ConvConfig& cfg) {
    e.y_shape = infer_conv_output_shape(cfg, Kind);
    const auto x_strides = get_strides(cfg.x, cfg.memory_format);
    const auto w_strides = get_strides(cfg.w, cfg.memory_format);
    const auto y_strides = get_strides(e.y_shape, cfg.memory_format);

    e.graph = std::make_shared<cudnn_frontend::graph::Graph>();
    e.graph->set_io_data_type(Traits::frontend_data_type)
        .set_compute_data_type(Traits::frontend_compute_type);

    e.x_tensor = e.graph->tensor(
        cudnn_frontend::graph::Tensor_attributes()
            .set_name("x")
            .set_dim(cfg.x)
            .set_stride(x_strides)
            .set_data_type(Traits::frontend_data_type));

    e.w_tensor = e.graph->tensor(
        cudnn_frontend::graph::Tensor_attributes()
            .set_name("w")
            .set_dim(cfg.w)
            .set_stride(w_strides)
            .set_data_type(Traits::frontend_data_type));

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> output;

    if constexpr (Kind == ConvKind::Forward) {
        auto conv_attr = cudnn_frontend::graph::Conv_fprop_attributes()
            .set_padding(cfg.padding)
            .set_stride(cfg.stride)
            .set_dilation(cfg.dilation)
            .set_compute_data_type(Traits::frontend_compute_type);
        output = e.graph->conv_fprop(e.x_tensor, e.w_tensor, conv_attr);
    } else {
        // Transpose convolution is represented as cuDNN data-gradient convolution:
        // dy(input), w(weight) -> dx(output).
        auto conv_attr = cudnn_frontend::graph::Conv_dgrad_attributes()
            .set_padding(cfg.padding)
            .set_stride(cfg.stride)
            .set_dilation(cfg.dilation)
            .set_compute_data_type(Traits::frontend_compute_type);
        output = e.graph->conv_dgrad(e.x_tensor, e.w_tensor, conv_attr);
    }
    output->set_dim(e.y_shape)
        .set_stride(y_strides)
        .set_data_type(Traits::frontend_data_type);

    if (cfg.with_bias) {
        // If the convolution result is consumed by a following pointwise op, it must be
        // a virtual intermediate tensor. Only the final graph tensor should be marked
        // as output. This is especially important for conv transpose, which is lowered
        // to Conv_dgrad in cuDNN frontend.
        output->set_is_virtual(true);

        const auto bias_shape = make_bias_shape(e.y_shape);
        const auto bias_strides = get_strides(bias_shape, cfg.memory_format);

        e.bias_tensor = e.graph->tensor(
            cudnn_frontend::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim(bias_shape)
                .set_stride(bias_strides)
                .set_data_type(Traits::frontend_data_type));

        auto bias_attr = cudnn_frontend::graph::Pointwise_attributes()
            .set_name("bias_add")
            .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
            .set_compute_data_type(Traits::frontend_compute_type);
        output = e.graph->pointwise(output, e.bias_tensor, bias_attr);
        output->set_dim(e.y_shape)
            .set_stride(y_strides)
            .set_data_type(Traits::frontend_data_type);
    }

    if (cfg.activation != Activation::None) {
        // Reference: the above description.
        output->set_is_virtual(true);

        auto act_attr = cudnn_frontend::graph::Pointwise_attributes()
            .set_name("activation")
            .set_mode(to_frontend_pointwise_mode(cfg.activation))
            .set_compute_data_type(Traits::frontend_compute_type);
        output = e.graph->pointwise(output, act_attr);
        output->set_dim(e.y_shape)
            .set_stride(y_strides)
            .set_data_type(Traits::frontend_data_type);
    }

    e.y_tensor = output;
    e.y_tensor->set_dim(e.y_shape)
        .set_stride(y_strides)
        .set_data_type(Traits::frontend_data_type)
        .set_output(true);

    CHECK_CUDNN_FE(e.graph->validate());
    CHECK_CUDNN_FE(e.graph->build_operation_graph(handle));
    CHECK_CUDNN_FE(e.graph->create_execution_plans({
        cudnn_frontend::HeurMode_t::A,
        cudnn_frontend::HeurMode_t::B,
        cudnn_frontend::HeurMode_t::FALLBACK
    }));
    CHECK_CUDNN_FE(e.graph->check_support());
    CHECK_CUDNN_FE(e.graph->build_plans());

    int64_t workspace_size = 0;
    CHECK_CUDNN_FE(e.graph->get_workspace_size(workspace_size));
    e.workspace_size = static_cast<size_t>(workspace_size);
}

template <typename T, ConvKind Kind>
void CudnnFrontendConvBase<T, Kind>::execute(T* x, T* w, T* bias, T* y) {
    // The graph was built with entry_->handle, so execute with the same handle.
    // acquire_specific() guarantees no other thread is using this handle at the same time.
    auto lease = CudnnHandlePool::instance().acquire_specific(device_id_, entry_->handle);
    cudnnHandle_t handle = lease.get();

    DeviceGuard device_guard(device_id_);
    CHECK_CUDNN(cudnnSetStream(handle, stream_));

    if (cfg_.with_bias && bias == nullptr) {
        throw std::runtime_error("Conv was built with bias, but run() received a null bias pointer.");
    }

    if (!cfg_.with_bias && bias != nullptr) {
        throw std::runtime_error("Conv was built without bias, but run() received a bias pointer.");
    }

    auto workspace_lease = DeviceWorkspacePool::instance().acquire(device_id_, entry_->workspace_size);
    void* workspace = workspace_lease.ptr();

    std::unordered_map<int64_t, void*> variant_pack = {
        {entry_->x_tensor->get_uid(), x},
        {entry_->w_tensor->get_uid(), w},
        {entry_->y_tensor->get_uid(), y},
    };

    if (cfg_.with_bias) {
        variant_pack[entry_->bias_tensor->get_uid()] = bias;
    }

    CHECK_CUDNN_FE(entry_->graph->execute(handle, variant_pack, workspace));
}

template <typename T, ConvKind Kind>
std::vector<int64_t> CudnnFrontendConvBase<T, Kind>::make_bias_shape(
    const std::vector<int64_t>& y_shape) {
    std::vector<int64_t> bias_shape(y_shape.size(), 1);
    bias_shape[1] = y_shape[1];
    return bias_shape;
}

template class CudnnFrontendConvBase<float, ConvKind::Forward>;
template class CudnnFrontendConvBase<double, ConvKind::Forward>;
template class CudnnFrontendConvBase<__half, ConvKind::Forward>;
template class CudnnFrontendConvBase<__nv_bfloat16, ConvKind::Forward>;

template class CudnnFrontendConvBase<float, ConvKind::Transpose>;
template class CudnnFrontendConvBase<double, ConvKind::Transpose>;
template class CudnnFrontendConvBase<__half, ConvKind::Transpose>;
template class CudnnFrontendConvBase<__nv_bfloat16, ConvKind::Transpose>;

}  // namespace cuda_nn
