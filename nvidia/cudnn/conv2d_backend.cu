/**
nvcc -O3 -std=c++17 conv2d_backend_nhwc.cu -o ./bin/conv_backend \
  -I$CUDA_HOME/include \
  -I$CUDNN_INCLUDE_DIR \
  -L$CUDA_HOME/lib64 \
  -L$CUDNN_LIBRARY_DIR \
  -lcudnn -lcudnn_cnn -lcudnn_ops -lcudnn_graph
*/

#include <cudnn.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDNN(expr)                                                   \
    do {                                                                    \
        cudnnStatus_t s = (expr);                                           \
        if (s != CUDNN_STATUS_SUCCESS) {                                    \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(s)          \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CHECK_CUDA(expr)                                                    \
    do {                                                                    \
        cudaError_t e = (expr);                                             \
        if (e != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(e)            \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

static cudnnBackendDescriptor_t makeTensorDesc(
    int64_t uid,
    const std::vector<int64_t>& dim,
    const std::vector<int64_t>& stride,
    cudnnDataType_t dtype,
    bool is_virtual = false
) {
    cudnnBackendDescriptor_t t;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &t));

    int64_t alignment = 16;
    CHECK_CUDNN(cudnnBackendSetAttribute(
        t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64,
        static_cast<int64_t>(dim.size()), dim.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64,
        static_cast<int64_t>(stride.size()), stride.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));

    if (is_virtual) {
        int64_t one = 1;
        CHECK_CUDNN(cudnnBackendSetAttribute(
            t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &one));
    }

    CHECK_CUDNN(cudnnBackendFinalize(t));
    return t;
}

static cudnnBackendDescriptor_t makeConvDesc(
    const std::array<int64_t, 2>& pad,
    const std::array<int64_t, 2>& stride,
    const std::array<int64_t, 2>& dilation,
    cudnnDataType_t compute_type
) {
    cudnnBackendDescriptor_t conv;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv));

    int64_t spatial_dim = 2;
    auto mode = CUDNN_CROSS_CORRELATION;

    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial_dim));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute_type));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pad.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pad.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilation.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, stride.data()));

    CHECK_CUDNN(cudnnBackendFinalize(conv));
    return conv;
}

static cudnnBackendDescriptor_t makePointwiseDesc(
    cudnnPointwiseMode_t mode,
    cudnnDataType_t compute_type
) {
    cudnnBackendDescriptor_t pw;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pw));

    CHECK_CUDNN(cudnnBackendSetAttribute(
        pw, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &compute_type));

    CHECK_CUDNN(cudnnBackendFinalize(pw));
    return pw;
}

static cudnnBackendDescriptor_t makeConvFwdOp(
    cudnnBackendDescriptor_t x,
    cudnnBackendDescriptor_t w,
    cudnnBackendDescriptor_t y,
    cudnnBackendDescriptor_t conv
) {
    cudnnBackendDescriptor_t op;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));

    double alpha = 1.0, beta = 0.0;

    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
        CUDNN_TYPE_DOUBLE, 1, &alpha));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
        CUDNN_TYPE_DOUBLE, 1, &beta));

    CHECK_CUDNN(cudnnBackendFinalize(op));
    return op;
}

static cudnnBackendDescriptor_t makeAddOp(
    cudnnBackendDescriptor_t a,
    cudnnBackendDescriptor_t b,
    cudnnBackendDescriptor_t y,
    cudnnBackendDescriptor_t pw
) {
    cudnnBackendDescriptor_t op;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &op));

    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &a));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y));

    CHECK_CUDNN(cudnnBackendFinalize(op));
    return op;
}

static float benchmarkPlan(
    cudnnHandle_t handle,
    cudnnBackendDescriptor_t plan,
    cudnnBackendDescriptor_t variant_pack,
    int warmup = 10,
    int iters = 100
) {
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUDNN(cudnnBackendExecute(handle, plan, variant_pack));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDNN(cudnnBackendExecute(handle, plan, variant_pack));
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

    const int64_t N = 1;
    const int64_t C = 3;
    const int64_t H = 224;
    const int64_t W = 224;

    const int64_t K = 64;
    const int64_t R = 7;
    const int64_t S = 7;

    const int64_t P = 112;
    const int64_t Q = 112;

    // NHWC packed strides, dims still written as [N,C,H,W]
    std::vector<int64_t> x_dim = {N, C, H, W};
    std::vector<int64_t> x_str = {H * W * C, 1, W * C, C};

    std::vector<int64_t> w_dim = {K, C, R, S};
    std::vector<int64_t> w_str = {R * S * C, 1, S * C, C};

    std::vector<int64_t> y_dim = {N, K, P, Q};
    std::vector<int64_t> y_str = {P * Q * K, 1, Q * K, K};

    std::vector<int64_t> bias_dim = {1, K, 1, 1};
    std::vector<int64_t> bias_str = {K, 1, K, K};

    auto X     = makeTensorDesc(100, x_dim,    x_str,    CUDNN_DATA_FLOAT, false);
    auto Wt    = makeTensorDesc(101, w_dim,    w_str,    CUDNN_DATA_FLOAT, false);
    auto Bias  = makeTensorDesc(102, bias_dim, bias_str, CUDNN_DATA_FLOAT, false);
    auto ConvY = makeTensorDesc(103, y_dim,    y_str,    CUDNN_DATA_FLOAT, true);
    auto Y     = makeTensorDesc(104, y_dim,    y_str,    CUDNN_DATA_FLOAT, false);

    auto ConvDesc = makeConvDesc({3, 3}, {2, 2}, {1, 1}, CUDNN_DATA_FLOAT);
    auto AddDesc  = makePointwiseDesc(CUDNN_POINTWISE_ADD, CUDNN_DATA_FLOAT);

    auto ConvOp = makeConvFwdOp(X, Wt, ConvY, ConvDesc);
    auto AddOp  = makeAddOp(ConvY, Bias, Y, AddDesc);

    cudnnBackendDescriptor_t opGraph;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));

    std::array<cudnnBackendDescriptor_t, 2> ops = {ConvOp, AddOp};
    CHECK_CUDNN(cudnnBackendSetAttribute(
        opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        static_cast<int64_t>(ops.size()), ops.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
        CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendFinalize(opGraph));

    cudnnBackendDescriptor_t heur;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));

    auto heur_mode = CUDNN_HEUR_MODE_A;
    CHECK_CUDNN(cudnnBackendSetAttribute(
        heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        heur, CUDNN_ATTR_ENGINEHEUR_MODE,
        CUDNN_TYPE_HEUR_MODE, 1, &heur_mode));
    CHECK_CUDNN(cudnnBackendFinalize(heur));

    constexpr int MAX_ENGINE_CONFIGS = 8;
    std::array<cudnnBackendDescriptor_t, MAX_ENGINE_CONFIGS> engine_configs{};
    for (int i = 0; i < MAX_ENGINE_CONFIGS; ++i) {
        CHECK_CUDNN(cudnnBackendCreateDescriptor(
            CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engine_configs[i]));
    }

    int64_t returned_count = 0;
    CHECK_CUDNN(cudnnBackendGetAttribute(
        heur,
        CUDNN_ATTR_ENGINEHEUR_RESULTS,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        MAX_ENGINE_CONFIGS,
        &returned_count,
        engine_configs.data()));

    if (returned_count <= 0) {
        std::cerr << "No engine config found from heuristics.\n";
        for (auto& ec : engine_configs) {
            if (ec) cudnnBackendDestroyDescriptor(ec);
        }
        return EXIT_FAILURE;
    }

    auto engine_cfg = engine_configs[0];

    cudnnBackendDescriptor_t plan;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
        CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_cfg));
    CHECK_CUDNN(cudnnBackendFinalize(plan));

    int64_t ws_size = 0;
    CHECK_CUDNN(cudnnBackendGetAttribute(
        plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
        CUDNN_TYPE_INT64, 1, nullptr, &ws_size));

    void* x_dev = nullptr;
    void* w_dev = nullptr;
    void* b_dev = nullptr;
    void* y_dev = nullptr;
    void* ws_dev = nullptr;

    CHECK_CUDA(cudaMalloc(&x_dev, static_cast<size_t>(N * C * H * W * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&w_dev, static_cast<size_t>(K * C * R * S * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&b_dev, static_cast<size_t>(K * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&y_dev, static_cast<size_t>(N * K * P * Q * sizeof(float))));
    if (ws_size > 0) {
        CHECK_CUDA(cudaMalloc(&ws_dev, static_cast<size_t>(ws_size)));
    }

    CHECK_CUDA(cudaMemset(x_dev, 0, static_cast<size_t>(N * C * H * W * sizeof(float))));
    CHECK_CUDA(cudaMemset(w_dev, 0, static_cast<size_t>(K * C * R * S * sizeof(float))));
    CHECK_CUDA(cudaMemset(b_dev, 0, static_cast<size_t>(K * sizeof(float))));
    CHECK_CUDA(cudaMemset(y_dev, 0, static_cast<size_t>(N * K * P * Q * sizeof(float))));
    if (ws_size > 0) {
        CHECK_CUDA(cudaMemset(ws_dev, 0, static_cast<size_t>(ws_size)));
    }

    std::array<int64_t, 4> uids = {100, 101, 102, 104};
    std::array<void*, 4> ptrs = {x_dev, w_dev, b_dev, y_dev};

    cudnnBackendDescriptor_t variant_pack;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant_pack));

    CHECK_CUDNN(cudnnBackendSetAttribute(
        variant_pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        CUDNN_TYPE_VOID_PTR, static_cast<int64_t>(ptrs.size()), ptrs.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        variant_pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
        CUDNN_TYPE_INT64, static_cast<int64_t>(uids.size()), uids.data()));
    if (ws_size > 0) {
        CHECK_CUDNN(cudnnBackendSetAttribute(
            variant_pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
            CUDNN_TYPE_VOID_PTR, 1, &ws_dev));
    }
    CHECK_CUDNN(cudnnBackendFinalize(variant_pack));

    std::cout << "Input : [" << N << ", " << C << ", " << H << ", " << W << "]\n";
    std::cout << "Weight: [" << K << ", " << C << ", " << R << ", " << S << "]\n";
    std::cout << "Bias  : [" << K << "]\n";
    std::cout << "Output: [" << N << ", " << K << ", " << P << ", " << Q << "]\n";
    std::cout << "Workspace bytes: " << ws_size << "\n";

    CHECK_CUDNN(cudnnBackendExecute(handle, plan, variant_pack));
    CHECK_CUDA(cudaDeviceSynchronize());

    float avg_ms = benchmarkPlan(handle, plan, variant_pack, 10, 100);
    std::cout << "Average time: " << avg_ms << " ms\n";

    CHECK_CUDNN(cudnnBackendDestroyDescriptor(variant_pack));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(plan));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(heur));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(opGraph));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(AddOp));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(ConvOp));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(AddDesc));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(ConvDesc));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(Y));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(ConvY));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(Bias));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(Wt));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(X));

    for (auto& ec : engine_configs) {
        if (ec) CHECK_CUDNN(cudnnBackendDestroyDescriptor(ec));
    }

    if (ws_dev) CHECK_CUDA(cudaFree(ws_dev));
    if (y_dev)  CHECK_CUDA(cudaFree(y_dev));
    if (b_dev)  CHECK_CUDA(cudaFree(b_dev));
    if (w_dev)  CHECK_CUDA(cudaFree(w_dev));
    if (x_dev)  CHECK_CUDA(cudaFree(x_dev));

    CHECK_CUDNN(cudnnDestroy(handle));
    return 0;
}