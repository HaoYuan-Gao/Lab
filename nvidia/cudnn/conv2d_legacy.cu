/**  compile instruction
 *
 nvcc -O3 -std=c++17 conv2d_legacy.cu -o ./bin/conv_legacy \
  -I$CUDA_HOME/include \
  -I$CUDNN_INCLUDE_DIR \
  -L$CUDA_HOME/lib64 \
  -L$CUDNN_LIBRARY_DIR \
  -lcudnn
 */

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CHECK_CUDNN(call)                                                         \
    do {                                                                          \
        cudnnStatus_t status = (call);                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                     \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

struct Conv2DParams {
    int N, C, H, W;
    int K, R, S;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dil_h, dil_w;
    bool use_bias;
};

class CudnnConv2D {
public:
    explicit CudnnConv2D(const Conv2DParams& p) : p_(p) {
        CHECK_CUDNN(cudnnCreate(&handle_));

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc_));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc_));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&b_desc_));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc_));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            x_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            p_.N, p_.C, p_.H, p_.W));

        CHECK_CUDNN(cudnnSetFilter4dDescriptor(
            w_desc_,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            p_.K, p_.C, p_.R, p_.S));

        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            p_.pad_h, p_.pad_w,
            p_.stride_h, p_.stride_w,
            p_.dil_h, p_.dil_w,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT));

        CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc_, 1));

        CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
            conv_desc_, x_desc_, w_desc_,
            &N_out_, &K_out_, &H_out_, &W_out_));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            y_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            N_out_, K_out_, H_out_, W_out_));

        if (p_.use_bias) {
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(
                b_desc_,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1, p_.K, 1, 1));
        }

        // demo里固定一个 algo
        algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            handle_, x_desc_, w_desc_, conv_desc_, y_desc_, algo_, &workspace_bytes_));

        if (workspace_bytes_ > 0) {
            CHECK_CUDA(cudaMalloc(&workspace_, workspace_bytes_));
        }
    }

    ~CudnnConv2D() {
        if (workspace_) cudaFree(workspace_);

        if (x_desc_) cudnnDestroyTensorDescriptor(x_desc_);
        if (y_desc_) cudnnDestroyTensorDescriptor(y_desc_);
        if (b_desc_) cudnnDestroyTensorDescriptor(b_desc_);
        if (w_desc_) cudnnDestroyFilterDescriptor(w_desc_);
        if (conv_desc_) cudnnDestroyConvolutionDescriptor(conv_desc_);
        if (handle_) cudnnDestroy(handle_);
    }

    void forward(
        const float* x_dev,
        const float* w_dev,
        const float* b_dev,
        float* y_dev,
        float alpha = 1.0f,
        float beta = 0.0f
    ) {
        CHECK_CUDNN(cudnnConvolutionForward(
            handle_,
            &alpha,
            x_desc_, x_dev,
            w_desc_, w_dev,
            conv_desc_,
            algo_,
            workspace_, workspace_bytes_,
            &beta,
            y_desc_, y_dev));

        if (p_.use_bias && b_dev != nullptr) {
            const float one = 1.0f;
            CHECK_CUDNN(cudnnAddTensor(
                handle_,
                &one,
                b_desc_, b_dev,
                &one,
                y_desc_, y_dev));
        }
    }

    int N_out() const { return N_out_; }
    int K_out() const { return K_out_; }
    int H_out() const { return H_out_; }
    int W_out() const { return W_out_; }
    size_t workspace_bytes() const { return workspace_bytes_; }

private:
    Conv2DParams p_;

    cudnnHandle_t handle_{};
    cudnnTensorDescriptor_t x_desc_{};
    cudnnTensorDescriptor_t y_desc_{};
    cudnnTensorDescriptor_t b_desc_{};
    cudnnFilterDescriptor_t w_desc_{};
    cudnnConvolutionDescriptor_t conv_desc_{};

    cudnnConvolutionFwdAlgo_t algo_{};
    void* workspace_{nullptr};
    size_t workspace_bytes_{0};

    int N_out_{0};
    int K_out_{0};
    int H_out_{0};
    int W_out_{0};
};

static void fill_random(std::vector<float>& v, float low = -1.0f, float high = 1.0f) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& x : v) x = dist(rng);
}

static void print_first_n(const std::vector<float>& v, int n) {
    int m = std::min<int>(n, static_cast<int>(v.size()));
    for (int i = 0; i < m; ++i) {
        std::cout << v[i] << (i + 1 == m ? '\n' : ' ');
    }
}

static float benchmark_conv(
    CudnnConv2D& conv,
    const float* x_dev,
    const float* w_dev,
    const float* b_dev,
    float* y_dev,
    int warmup = 10,
    int iters = 100
) {
    for (int i = 0; i < warmup; ++i) {
        conv.forward(x_dev, w_dev, b_dev, y_dev, 1.0f, 0.0f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        conv.forward(x_dev, w_dev, b_dev, y_dev, 1.0f, 0.0f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float avg_ms = total_ms / static_cast<float>(iters);

    std::cout << "Benchmark result:\n";
    std::cout << "  warmup          : " << warmup << "\n";
    std::cout << "  iterations      : " << iters << "\n";
    std::cout << "  total time (ms) : " << total_ms << "\n";
    std::cout << "  avg time (ms)   : " << avg_ms << "\n";
    std::cout << "  iter/s          : " << 1000.0f / avg_ms << "\n";

    return avg_ms;
}

int main() {
    Conv2DParams p;
    p.N = 1;
    p.C = 3;
    p.H = 224;
    p.W = 224;

    p.K = 64;
    p.R = 7;
    p.S = 7;

    p.pad_h = 3;
    p.pad_w = 3;
    p.stride_h = 2;
    p.stride_w = 2;
    p.dil_h = 1;
    p.dil_w = 1;

    p.use_bias = true;

    CudnnConv2D conv(p);

    const size_t x_numel = static_cast<size_t>(p.N) * p.C * p.H * p.W;
    const size_t w_numel = static_cast<size_t>(p.K) * p.C * p.R * p.S;
    const size_t b_numel = static_cast<size_t>(p.K);
    const size_t y_numel = static_cast<size_t>(conv.N_out()) * conv.K_out() * conv.H_out() * conv.W_out();

    std::vector<float> x_host(x_numel);
    std::vector<float> w_host(w_numel);
    std::vector<float> b_host(b_numel);
    std::vector<float> y_host(y_numel);

    fill_random(x_host);
    fill_random(w_host);
    fill_random(b_host);

    float* x_dev = nullptr;
    float* w_dev = nullptr;
    float* b_dev = nullptr;
    float* y_dev = nullptr;

    CHECK_CUDA(cudaMalloc(&x_dev, x_numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_dev, w_numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_dev, b_numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y_dev, y_numel * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(x_dev, x_host.data(), x_numel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(w_dev, w_host.data(), w_numel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_dev, b_host.data(), b_numel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(y_dev, 0, y_numel * sizeof(float)));

    std::cout << "Input : [" << p.N << ", " << p.C << ", " << p.H << ", " << p.W << "]\n";
    std::cout << "Weight: [" << p.K << ", " << p.C << ", " << p.R << ", " << p.S << "]\n";
    std::cout << "Bias  : [" << p.K << "]\n";
    std::cout << "Output: [" << conv.N_out() << ", " << conv.K_out() << ", "
              << conv.H_out() << ", " << conv.W_out() << "]\n";
    std::cout << "Workspace bytes: " << conv.workspace_bytes() << "\n";

    benchmark_conv(conv, x_dev, w_dev, p.use_bias ? b_dev : nullptr, y_dev, 10, 100);

    CHECK_CUDA(cudaMemcpy(y_host.data(), y_dev, y_numel * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "First 10 output values:\n";
    print_first_n(y_host, 10);

    CHECK_CUDA(cudaFree(x_dev));
    CHECK_CUDA(cudaFree(w_dev));
    CHECK_CUDA(cudaFree(b_dev));
    CHECK_CUDA(cudaFree(y_dev));

    return 0;
}