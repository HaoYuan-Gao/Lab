// 编译：nvcc graph_node.cu -o graph_node
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(expr)                                      \
    do {                                                      \
        cudaError_t err = (expr);                             \
        if (err != cudaSuccess) {                             \
            throw std::runtime_error(cudaGetErrorString(err));\
        }                                                     \
    } while (0)

__global__ void scale_kernel(
    const float* x,
    float* tmp,
    int n,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        tmp[idx] = x[idx] * scale;
    }
}

__global__ void add_kernel(
    const float* tmp,
    const float* bias,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tmp[idx] + bias[idx];
    }
}

class SimpleGraph {
public:
    SimpleGraph(int n, float scale)
        : n_(n), scale_(scale) {
        threads_ = 256;
        blocks_ = (n + threads_ - 1) / threads_;
    }

    ~SimpleGraph() {
        if (exec_) {
            cudaGraphExecDestroy(exec_);
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
        }
    }

    void instantiate(
        const float* x,
        float* tmp,
        const float* bias,
        float* out
    ) {
        x_ = x;
        tmp_ = tmp;
        bias_ = bias;
        out_ = out;

        CUDA_CHECK(cudaGraphCreate(&graph_, 0));

        add_scale_node();
        add_add_node();

        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
    }

    void update_ptrs(
        const float* x,
        float* tmp,
        const float* bias,
        float* out
    ) {
        x_ = x;
        tmp_ = tmp;
        bias_ = bias;
        out_ = out;

        cudaKernelNodeParams scale_params{};
        scale_params.func = reinterpret_cast<void*>(scale_kernel);
        scale_params.gridDim = dim3(blocks_);
        scale_params.blockDim = dim3(threads_);
        scale_params.sharedMemBytes = 0;
        scale_params.kernelParams = scale_args_;
        scale_params.extra = nullptr;

        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
            exec_,
            scale_node_,
            &scale_params
        ));

        cudaKernelNodeParams add_params{};
        add_params.func = reinterpret_cast<void*>(add_kernel);
        add_params.gridDim = dim3(blocks_);
        add_params.blockDim = dim3(threads_);
        add_params.sharedMemBytes = 0;
        add_params.kernelParams = add_args_;
        add_params.extra = nullptr;

        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
            exec_,
            add_node_,
            &add_params
        ));
    }

    void launch(cudaStream_t stream) {
        CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    }

private:
    void add_scale_node() {
        scale_args_[0] = &x_;
        scale_args_[1] = &tmp_;
        scale_args_[2] = &n_;
        scale_args_[3] = &scale_;

        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(scale_kernel);
        params.gridDim = dim3(blocks_);
        params.blockDim = dim3(threads_);
        params.sharedMemBytes = 0;
        params.kernelParams = scale_args_;
        params.extra = nullptr;

        CUDA_CHECK(cudaGraphAddKernelNode(
            &scale_node_,
            graph_,
            nullptr,
            0,
            &params
        ));
    }

    void add_add_node() {
        add_args_[0] = &tmp_;
        add_args_[1] = &bias_;
        add_args_[2] = &out_;
        add_args_[3] = &n_;

        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(add_kernel);
        params.gridDim = dim3(blocks_);
        params.blockDim = dim3(threads_);
        params.sharedMemBytes = 0;
        params.kernelParams = add_args_;
        params.extra = nullptr;

        cudaGraphNode_t deps[] = {scale_node_};

        CUDA_CHECK(cudaGraphAddKernelNode(
            &add_node_,
            graph_,
            deps,
            1,
            &params
        ));
    }

private:
    int n_;
    int threads_;
    int blocks_;
    float scale_;

    const float* x_ = nullptr;
    float* tmp_ = nullptr;
    const float* bias_ = nullptr;
    float* out_ = nullptr;

    void* scale_args_[4]{};
    void* add_args_[4]{};

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaGraphNode_t scale_node_ = nullptr;
    cudaGraphNode_t add_node_ = nullptr;
};

int main() {
    constexpr int n = 8;
    constexpr float scale = 2.0f;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // -------------------------
    // host buffers
    // -------------------------

    std::vector<float> h_x1(n);
    std::vector<float> h_bias1(n);
    std::vector<float> h_out1(n);

    std::vector<float> h_x2(n);
    std::vector<float> h_bias2(n);
    std::vector<float> h_out2(n);

    for (int i = 0; i < n; ++i) {
        h_x1[i] = static_cast<float>(i);
        h_bias1[i] = 100.0f;

        h_x2[i] = static_cast<float>(i) * 10.0f;
        h_bias2[i] = 200.0f;
    }

    // -------------------------
    // device buffers
    // -------------------------

    float *x1, *tmp1, *bias1, *out1;
    float *x2, *tmp2, *bias2, *out2;

    CUDA_CHECK(cudaMalloc(&x1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out1, n * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&x2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out2, n * sizeof(float)));

    // -------------------------
    // H2D memcpy
    // -------------------------

    CUDA_CHECK(cudaMemcpyAsync(
        x1,
        h_x1.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaMemcpyAsync(
        bias1,
        h_bias1.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaMemcpyAsync(
        x2,
        h_x2.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaMemcpyAsync(
        bias2,
        h_bias2.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));

    // -------------------------
    // graph
    // -------------------------

    SimpleGraph graph(n, scale);

    // first launch
    graph.instantiate(x1, tmp1, bias1, out1);
    graph.launch(stream);

    // second launch with ptr update
    graph.update_ptrs(x2, tmp2, bias2, out2);
    graph.launch(stream);

    // -------------------------
    // D2H memcpy
    // -------------------------

    CUDA_CHECK(cudaMemcpyAsync(
        h_out1.data(),
        out1,
        n * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    ));

    CUDA_CHECK(cudaMemcpyAsync(
        h_out2.data(),
        out2,
        n * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // -------------------------
    // print
    // -------------------------

    std::cout << "==== out1 ====\n";

    for (int i = 0; i < n; ++i) {
        std::cout << h_out1[i] << " ";
    }

    std::cout << "\n";

    std::cout << "==== out2 ====\n";

    for (int i = 0; i < n; ++i) {
        std::cout << h_out2[i] << " ";
    }

    std::cout << "\n";

    // -------------------------
    // cleanup
    // -------------------------

    cudaFree(x1);
    cudaFree(tmp1);
    cudaFree(bias1);
    cudaFree(out1);

    cudaFree(x2);
    cudaFree(tmp2);
    cudaFree(bias2);
    cudaFree(out2);

    cudaStreamDestroy(stream);

    return 0;
}
