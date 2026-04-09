/**
nvcc -O3 -std=c++17 conv2d_tansposed_frontend.cu -o ./bin/transposed_bias \
  -I$CUDA_HOME/include \
  -I$CUDNN_INCLUDE_DIR \
  -I./cudnn-frontend/include \
  -L$CUDA_HOME/lib64 \
  -L$CUDNN_LIBRARY_DIR \
  -lcudnn
 * 
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA(expr)                                                    \
    do {                                                                    \
        cudaError_t e = (expr);                                             \
        if (e != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(e)            \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CHECK_CUDNN(expr)                                                   \
    do {                                                                    \
        cudnnStatus_t s = (expr);                                           \
        if (s != CUDNN_STATUS_SUCCESS) {                                    \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(s)          \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CHECK_FE(expr)                                                      \
    do {                                                                    \
        auto fe_s = (expr);                                                 \
        if (!fe_s.is_good()) {                                              \
            std::cerr << "cuDNN frontend error: "                           \
                      << fe_s.get_message()                                 \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

static float benchmark(
    cudnnHandle_t handle,
    cudnn_frontend::graph::Graph& graph,
    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*>& tensor_ptrs,
    void* workspace,
    int warmup = 10,
    int iters = 100) {

    for (int i = 0; i < warmup; ++i) {
        CHECK_FE(graph.execute(handle, tensor_ptrs, workspace));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_FE(graph.execute(handle, tensor_ptrs, workspace));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(iters);
}

int main() {
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    std::cout << "cudnn version = " << cudnnGetVersion() << "\n";

    // "Input" to transposed-conv op (i.e. dy / activation input)
    constexpr int64_t N = 1;
    constexpr int64_t C_in = 64;
    constexpr int64_t H_in = 112;
    constexpr int64_t W_in = 112;

    // Filter shape in cuDNN convention: [K, C, R, S]
    // Here output channels = 3
    constexpr int64_t K = C_in;
    constexpr int64_t C_out = 3;
    constexpr int64_t R = 7;
    constexpr int64_t S = 7;

    constexpr int64_t pad_h = 3;
    constexpr int64_t pad_w = 3;
    constexpr int64_t stride_h = 2;
    constexpr int64_t stride_w = 2;
    constexpr int64_t dil_h = 1;
    constexpr int64_t dil_w = 1;

    // For conv_dgrad:
    // H_out = (H_in - 1) * stride - 2*pad + dilation*(R-1) + 1
    constexpr int64_t H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (R - 1) + 1; // 223
    constexpr int64_t W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (S - 1) + 1; // 223

    // NHWC packed stride, dims still written as [N, C, H, W]
    std::vector<int64_t> x_dim    = {N, C_in, H_in, W_in};
    std::vector<int64_t> x_stride = {H_in * W_in * C_in, 1, W_in * C_in, C_in};

    std::vector<int64_t> w_dim    = {K, C_out, R, S};
    std::vector<int64_t> w_stride = {R * S * C_out, 1, S * C_out, C_out};

    std::vector<int64_t> y_dim    = {N, C_out, H_out, W_out};
    std::vector<int64_t> y_stride = {H_out * W_out * C_out, 1, W_out * C_out, C_out};

    std::vector<int64_t> b_dim    = {1, C_out, 1, 1};
    std::vector<int64_t> b_stride = {C_out, 1, C_out, C_out};

    cudnn_frontend::graph::Graph graph;
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto X = graph.tensor(
        cudnn_frontend::graph::Tensor_attributes()
            .set_name("X")
            .set_dim(x_dim)
            .set_stride(x_stride));

    auto W = graph.tensor(
        cudnn_frontend::graph::Tensor_attributes()
            .set_name("W")
            .set_dim(w_dim)
            .set_stride(w_stride));
    
    auto Bias = graph.tensor(
        cudnn_frontend::graph::Tensor_attributes()
            .set_name("Bias")
            .set_dim(b_dim)
            .set_stride(b_stride));

    // conv transpose == conv_dgrad
    auto ConvT_Y = graph.conv_dgrad(
        X,
        W,
        cudnn_frontend::graph::Conv_dgrad_attributes()
            .set_padding({pad_h, pad_w})
            .set_stride({stride_h, stride_w})
            .set_dilation({dil_h, dil_w})
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));

    ConvT_Y->set_name("ConvT_Y")
        .set_dim(y_dim)
        .set_stride(y_stride)
        .set_output(true)
        .set_is_virtual(true);

    auto Y = graph.pointwise(
        ConvT_Y,
        Bias,
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));

    CHECK_FE(graph.validate());
    CHECK_FE(graph.build_operation_graph(handle));
    CHECK_FE(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}));
    CHECK_FE(graph.check_support(handle));
    CHECK_FE(graph.build_plans(handle));

    auto workspace_size = graph.get_workspace_size();

    void* x_dev  = nullptr;
    void* w_dev  = nullptr;
    void* b_dev = nullptr;
    void* y_dev  = nullptr;
    void* ws_dev = nullptr;

    CHECK_CUDA(cudaMalloc(&x_dev, static_cast<size_t>(N * C_in * H_in * W_in * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&w_dev, static_cast<size_t>(K * C_out * R * S * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&b_dev, static_cast<size_t>(C_out * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&y_dev, static_cast<size_t>(N * C_out * H_out * W_out * sizeof(float))));
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&ws_dev, workspace_size));
    }

    CHECK_CUDA(cudaMemset(x_dev, 0, static_cast<size_t>(N * C_in * H_in * W_in * sizeof(float))));
    CHECK_CUDA(cudaMemset(w_dev, 0, static_cast<size_t>(K * C_out * R * S * sizeof(float))));
    CHECK_CUDA(cudaMemset(b_dev, 0, static_cast<size_t>(C_out * sizeof(float))));
    CHECK_CUDA(cudaMemset(y_dev, 0, static_cast<size_t>(N * C_out * H_out * W_out * sizeof(float))));
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMemset(ws_dev, 0, workspace_size));
    }

    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensor_ptrs = {
        {X, x_dev},
        {W, w_dev},
        {Bias, b_dev},
        {Y, y_dev},
    };

    std::cout << "Input : [" << N << ", " << C_in << ", " << H_in << ", " << W_in << "]\n";
    std::cout << "Weight: [" << K << ", " << C_out << ", " << R << ", " << S << "]\n";
    std::cout << "Bias  : [" << C_out << "]\n";
    std::cout << "Output: [" << N << ", " << C_out << ", " << H_out << ", " << W_out << "]\n";
    std::cout << "Workspace bytes: " << workspace_size << "\n";

    CHECK_FE(graph.execute(handle, tensor_ptrs, ws_dev));
    CHECK_CUDA(cudaDeviceSynchronize());

    float avg_ms = benchmark(handle, graph, tensor_ptrs, ws_dev, 10, 100);
    std::cout << "Average time: " << avg_ms << " ms\n";

    if (ws_dev) CHECK_CUDA(cudaFree(ws_dev));
    if (y_dev)  CHECK_CUDA(cudaFree(y_dev));
    if (w_dev)  CHECK_CUDA(cudaFree(w_dev));
    if (x_dev)  CHECK_CUDA(cudaFree(x_dev));

    CHECK_CUDNN(cudnnDestroy(handle));
    return 0;
}