// 编译：nvcc -std=c++17 graph_conditional.cu -o graph_conditional
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

__global__ void decide_kernel(
    cudaGraphConditionalHandle handle,
    int run_branch
) {
    // run_branch != 0: execute conditional body
    // run_branch == 0: skip conditional body
    cudaGraphSetConditional(handle, run_branch != 0);
}

__global__ void add_kernel(
    const float* x,
    const float* bias,
    float* tmp,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        tmp[idx] = x[idx] + bias[idx];
    }
}

__global__ void final_kernel(
    const float* tmp,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = tmp[idx] * 10.0f;
    }
}

class ConditionalGraph {
public:
    ConditionalGraph(int n)
        : n_(n) {
        threads_ = 256;
        blocks_ = (n + threads_ - 1) / threads_;
    }

    ~ConditionalGraph() {
        if (exec_ != nullptr) {
            cudaGraphExecDestroy(exec_);
        }

        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
        }
    }

    void instantiate(
        const float* x,
        const float* bias,
        float* tmp,
        float* out,
        int run_branch
    ) {
        x_ = x;
        bias_ = bias;
        tmp_ = tmp;
        out_ = out;
        run_branch_ = run_branch;

        CUDA_CHECK(cudaGraphCreate(&graph_, 0));

        CUDA_CHECK(cudaGraphConditionalHandleCreate(
            &condition_handle_,
            graph_,
            0,
            cudaGraphCondAssignDefault
        ));

        add_decide_node();
        add_conditional_node();
        add_final_node();

        CUDA_CHECK(cudaGraphInstantiate(
            &exec_,
            graph_,
            nullptr,
            nullptr,
            0
        ));
    }

    void update_ptrs(
        const float* x,
        const float* bias,
        float* tmp,
        float* out
    ) {
        x_ = x;
        bias_ = bias;
        tmp_ = tmp;
        out_ = out;

        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
            exec_,
            add_node_,
            &add_params_
        ));

        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
            exec_,
            final_node_,
            &final_params_
        ));
    }

    void update_branch(int run_branch) {
        run_branch_ = run_branch;

        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
            exec_,
            decide_node_,
            &decide_params_
        ));
    }

    void launch(cudaStream_t stream) {
        CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    }

private:
    void add_decide_node() {
        decide_args_[0] = &condition_handle_;
        decide_args_[1] = &run_branch_;

        decide_params_.func = reinterpret_cast<void*>(decide_kernel);
        decide_params_.gridDim = dim3(1);
        decide_params_.blockDim = dim3(1);
        decide_params_.sharedMemBytes = 0;
        decide_params_.kernelParams = decide_args_;
        decide_params_.extra = nullptr;

        CUDA_CHECK(cudaGraphAddKernelNode(
            &decide_node_,
            graph_,
            nullptr,
            0,
            &decide_params_
        ));
    }

    void add_conditional_node() {
        cudaGraphNodeParams cond_params{};
        cond_params.type = cudaGraphNodeTypeConditional;
        cond_params.conditional.handle = condition_handle_;
        cond_params.conditional.type = cudaGraphCondTypeIf;
        cond_params.conditional.size = 1;

        cudaGraphNode_t deps[] = {decide_node_};

        CUDA_CHECK(cudaGraphAddNode(
            &conditional_node_,
            graph_,
            deps,
            1,
            &cond_params
        ));

        cudaGraph_t body_graph = cond_params.conditional.phGraph_out[0];

        add_args_[0] = &x_;
        add_args_[1] = &bias_;
        add_args_[2] = &tmp_;
        add_args_[3] = &n_;

        add_params_.func = reinterpret_cast<void*>(add_kernel);
        add_params_.gridDim = dim3(blocks_);
        add_params_.blockDim = dim3(threads_);
        add_params_.sharedMemBytes = 0;
        add_params_.kernelParams = add_args_;
        add_params_.extra = nullptr;

        CUDA_CHECK(cudaGraphAddKernelNode(
            &add_node_,
            body_graph,
            nullptr,
            0,
            &add_params_
        ));
    }

    void add_final_node() {
        final_args_[0] = &tmp_;
        final_args_[1] = &out_;
        final_args_[2] = &n_;

        final_params_.func = reinterpret_cast<void*>(final_kernel);
        final_params_.gridDim = dim3(blocks_);
        final_params_.blockDim = dim3(threads_);
        final_params_.sharedMemBytes = 0;
        final_params_.kernelParams = final_args_;
        final_params_.extra = nullptr;

        cudaGraphNode_t deps[] = {conditional_node_};

        CUDA_CHECK(cudaGraphAddKernelNode(
            &final_node_,
            graph_,
            deps,
            1,
            &final_params_
        ));
    }

private:
    int n_;
    int threads_;
    int blocks_;

    const float* x_ = nullptr;
    const float* bias_ = nullptr;
    float* tmp_ = nullptr;
    float* out_ = nullptr;

    int run_branch_ = 0;

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;

    cudaGraphConditionalHandle condition_handle_{};
    cudaGraphNode_t decide_node_ = nullptr;
    cudaGraphNode_t conditional_node_ = nullptr;
    cudaGraphNode_t add_node_ = nullptr;
    cudaGraphNode_t final_node_ = nullptr;

    void* decide_args_[2]{};
    void* add_args_[4]{};
    void* final_args_[3]{};

    cudaKernelNodeParams decide_params_{};
    cudaKernelNodeParams add_params_{};
    cudaKernelNodeParams final_params_{};
};

int main() {
    constexpr int n = 8;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<float> h_x1(n);
    std::vector<float> h_bias1(n);
    std::vector<float> h_tmp1(n, 1.0f);
    std::vector<float> h_out1(n);

    std::vector<float> h_x2(n);
    std::vector<float> h_bias2(n);
    std::vector<float> h_tmp2(n, 1.0f);
    std::vector<float> h_out2(n);

    for (int i = 0; i < n; ++i) {
        h_x1[i] = static_cast<float>(i);
        h_bias1[i] = 100.0f;

        h_x2[i] = static_cast<float>(i);
        h_bias2[i] = 200.0f;
    }

    float *x1, *bias1, *tmp1, *out1;
    float *x2, *bias2, *tmp2, *out2;

    CUDA_CHECK(cudaMalloc(&x1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out1, n * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&x2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out2, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(x1, h_x1.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(bias1, h_bias1.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tmp1, h_tmp1.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(x2, h_x2.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(bias2, h_bias2.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tmp2, h_tmp2.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));

    ConditionalGraph graph(n);

    // 第一次：run_branch = 1，会执行 add_kernel
    // tmp = x + bias
    // out = tmp * 10
    graph.instantiate(x1, bias1, tmp1, out1, 1);
    graph.launch(stream);

    // 第二次：更新 ptr，并且 run_branch = 0，会跳过 add_kernel
    // tmp 保持初始值 1
    // out = tmp * 10 = 10
    graph.update_ptrs(x2, bias2, tmp2, out2);
    graph.update_branch(0);
    graph.launch(stream);

    CUDA_CHECK(cudaMemcpyAsync(h_out1.data(), out1, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out2.data(), out2, n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "==== branch enabled ====\n";

    for (int i = 0; i < n; ++i) {
        std::cout << h_out1[i] << " ";
    }

    std::cout << "\n";

    std::cout << "==== branch disabled ====\n";

    for (int i = 0; i < n; ++i) {
        std::cout << h_out2[i] << " ";
    }

    std::cout << "\n";

    cudaFree(x1);
    cudaFree(bias1);
    cudaFree(tmp1);
    cudaFree(out1);

    cudaFree(x2);
    cudaFree(bias2);
    cudaFree(tmp2);
    cudaFree(out2);

    cudaStreamDestroy(stream);

    return 0;
}
