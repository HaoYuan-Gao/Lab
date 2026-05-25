#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

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

    int64_t spatial_ndim() const {
        if (x.size() < 3) {
            throw std::runtime_error("ConvConfig.x must be [N, C, ...spatial].");
        }
        return static_cast<int64_t>(x.size()) - 2;
    }
};

} // namespace cuda_nn