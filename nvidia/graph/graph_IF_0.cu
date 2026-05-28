// nvcc graph_IF_0.cu -O3 --use_fast_math -std=c++17 -lcudart -o test
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                      \
  do {                                                        \
    cudaError_t err = (expr);                                 \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error: %s\n", cudaGetErrorString(err));    \
      std::abort();                                           \
    }                                                         \
  } while (0)

enum class NoiseFunc : int {
  Exp = 0,
  Sech = 1,
  Sigmoid = 2,
};

enum class OperatorType : int {
  Linear = 0,
  Conv = 1,
  ConvT = 2,
};

__device__ __forceinline__ float ComputeInferStdDevice(
    float y,
    float a_minus_b,
    float b,
    float noise_scale,
    float k,
    float x0_half,
    NoiseFunc func) {
  float s;

  if (func == NoiseFunc::Exp) {
    float e = __expf(-k * y);
    s = fmaf(a_minus_b, e, b);
  } else if (func == NoiseFunc::Sech) {
    float x = k * y;
    float e = __expf(-fabsf(x));
    float e2 = e * e;
    float sech = (2.f * e) * __frcp_rn(1.f + e2);
    s = fmaf(a_minus_b, sech, b);
  } else {
    float e = __expf(k * (y - x0_half));
    float sig = __frcp_rn(1.f + e);
    s = fmaf(a_minus_b, sig, b);
  }

  return s * noise_scale;
}

template <typename T>
__global__ void InferStdAndBiasKernel(
    const T* __restrict__ y_std,
    float* __restrict__ infer_std,
    float* __restrict__ error_bias,
    int64_t Cout,
    float a_minus_b,
    float b,
    float noise_scale,
    float sigma_bias,
    float k,
    float x0_half,
    NoiseFunc func,
    uint64_t seed) {
  int64_t c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c >= Cout) {
    return;
  }

  float y = static_cast<float>(y_std[c]);

  infer_std[c] = ComputeInferStdDevice(
      y,
      a_minus_b,
      b,
      noise_scale,
      k,
      x0_half,
      func);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, c, 0, &state);

  error_bias[c] = curand_normal(&state) * sigma_bias;
}

template <typename T, bool EnablePost>
__global__ void ApplyLinearFusedKernel2D(
    const float* __restrict__ infer_std,
    const float* __restrict__ error_bias,
    const float* __restrict__ g_value,
    T* __restrict__ y,
    int64_t outer,
    int64_t Cout,
    int64_t g_count,
    uint64_t seed) {
  int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t row = blockIdx.y;

  if (c >= Cout || row >= outer) {
    return;
  }

  int64_t idx = row * Cout + c;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  float rnd = curand_normal(&state);
  float delta = fmaf(rnd, infer_std[c], error_bias[c]);

  if constexpr (EnablePost) {
    float g = g_value[g_count == 1 ? 0 : c];

    float v = static_cast<float>(y[idx]);
    v = v / g + delta;
    v = fminf(127.f, fmaxf(-128.f, v));
    v = nearbyintf(v);

    y[idx] = static_cast<T>(v);
  } else {
    y[idx] = static_cast<T>(delta);
  }
}

template <typename T, bool EnablePost>
__global__ void ApplyConvFusedKernel(
    const float* __restrict__ infer_std,
    const float* __restrict__ error_bias,
    const float* __restrict__ g_value,
    T* __restrict__ y,
    int64_t N,
    int64_t Cout,
    int64_t HW,
    int64_t g_count,
    uint64_t seed) {
  int64_t hw = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t c = blockIdx.y;
  int64_t n = blockIdx.z;

  if (hw >= HW) {
    return;
  }

  int64_t idx = (n * Cout + c) * HW + hw;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  float rnd = curand_normal(&state);
  float delta = fmaf(rnd, infer_std[c], error_bias[c]);

  if constexpr (EnablePost) {
    float g = g_value[g_count == 1 ? 0 : c];

    float v = static_cast<float>(y[idx]);
    v = v / g + delta;
    v = fminf(127.f, fmaxf(-128.f, v));
    v = nearbyintf(v);

    y[idx] = static_cast<T>(v);
  } else {
    y[idx] = static_cast<T>(delta);
  }
}

template <typename T, bool EnablePost>
class NoiseGraph {
 public:
  static size_t WorkspaceBytes(int64_t Cout) {
    return sizeof(float) * Cout * 2;
  }

  void Build(
      const T* y_std,
      const float* g_value,
      T* y,
      float* workspace,
      int64_t N,
      int64_t numel,
      int64_t Cout,
      int64_t HW,
      int64_t g_count,
      float a,
      float b,
      float epsilon,
      float x0,
      float noise_scale,
      float sigma_bias,
      NoiseFunc func,
      OperatorType op,
      uint64_t seed) {
    y_std_ = y_std;
    g_value_ = g_value;
    y_ = y;

    infer_std_ = workspace;
    error_bias_ = workspace + Cout;

    N_ = N;
    numel_ = numel;
    Cout_ = Cout;
    HW_ = HW;
    g_count_ = g_count;
    linear_outer_ = numel / Cout;

    a_minus_b_ = a - b;
    b_ = b;
    noise_scale_ = noise_scale;
    sigma_bias_ = sigma_bias;
    func_ = func;
    op_ = op;

    x0_half_ = x0 * 0.5f;

    if (func == NoiseFunc::Exp) {
      k_ = std::log(a_minus_b_ / epsilon) / x0;
    } else if (func == NoiseFunc::Sech) {
      k_ = std::acosh(2.f / epsilon - 1.f) / x0;
    } else {
      k_ = (2.f / x0) * std::log(a_minus_b_ / epsilon - 1.f);
    }

    bias_seed_ = seed;
    noise_seed_ = seed + 1;

    threads_ = 256;
    blocks_c_ = static_cast<unsigned int>(
        (Cout_ + threads_ - 1) / threads_);

    CUDA_CHECK(cudaGraphCreate(&graph_, 0));

    AddInferNode();
    AddApplyNode();

    CUDA_CHECK(cudaGraphInstantiate(
        &graph_exec_,
        graph_,
        nullptr,
        nullptr,
        0));
  }

  void Launch(cudaStream_t stream) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
  }

  void UpdateSeeds(uint64_t seed) {
    bias_seed_ = seed;
    noise_seed_ = seed + 1;

    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
        graph_exec_,
        infer_node_,
        &infer_params_));

    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
        graph_exec_,
        apply_node_,
        &apply_params_));
  }

  void Destroy() {
    if (graph_exec_ != nullptr) {
      CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
      graph_exec_ = nullptr;
    }

    if (graph_ != nullptr) {
      CUDA_CHECK(cudaGraphDestroy(graph_));
      graph_ = nullptr;
    }
  }

 private:
  void AddInferNode() {
    infer_args_[0] = &y_std_;
    infer_args_[1] = &infer_std_;
    infer_args_[2] = &error_bias_;
    infer_args_[3] = &Cout_;
    infer_args_[4] = &a_minus_b_;
    infer_args_[5] = &b_;
    infer_args_[6] = &noise_scale_;
    infer_args_[7] = &sigma_bias_;
    infer_args_[8] = &k_;
    infer_args_[9] = &x0_half_;
    infer_args_[10] = &func_;
    infer_args_[11] = &bias_seed_;

    infer_params_ = {};
    infer_params_.func =
        reinterpret_cast<void*>(InferStdAndBiasKernel<T>);
    infer_params_.gridDim = dim3(blocks_c_);
    infer_params_.blockDim = dim3(threads_);
    infer_params_.sharedMemBytes = 0;
    infer_params_.kernelParams = infer_args_;
    infer_params_.extra = nullptr;

    CUDA_CHECK(cudaGraphAddKernelNode(
        &infer_node_,
        graph_,
        nullptr,
        0,
        &infer_params_));
  }

  void AddApplyNode() {
    apply_params_ = {};

    if (op_ == OperatorType::Linear) {
      apply_args_[0] = &infer_std_;
      apply_args_[1] = &error_bias_;
      apply_args_[2] = &g_value_;
      apply_args_[3] = &y_;
      apply_args_[4] = &linear_outer_;
      apply_args_[5] = &Cout_;
      apply_args_[6] = &g_count_;
      apply_args_[7] = &noise_seed_;

      apply_params_.func =
          reinterpret_cast<void*>(
              ApplyLinearFusedKernel2D<T, EnablePost>);

      apply_params_.gridDim = dim3(
          static_cast<unsigned int>((Cout_ + threads_ - 1) / threads_),
          static_cast<unsigned int>(linear_outer_));

      apply_params_.blockDim = dim3(threads_);

    } else {
      apply_args_[0] = &infer_std_;
      apply_args_[1] = &error_bias_;
      apply_args_[2] = &g_value_;
      apply_args_[3] = &y_;
      apply_args_[4] = &N_;
      apply_args_[5] = &Cout_;
      apply_args_[6] = &HW_;
      apply_args_[7] = &g_count_;
      apply_args_[8] = &noise_seed_;

      apply_params_.func =
          reinterpret_cast<void*>(
              ApplyConvFusedKernel<T, EnablePost>);

      apply_params_.gridDim = dim3(
          static_cast<unsigned int>((HW_ + threads_ - 1) / threads_),
          static_cast<unsigned int>(Cout_),
          static_cast<unsigned int>(N_));

      apply_params_.blockDim = dim3(threads_);
    }

    apply_params_.sharedMemBytes = 0;
    apply_params_.kernelParams = apply_args_;
    apply_params_.extra = nullptr;

    cudaGraphNode_t deps[] = {infer_node_};

    CUDA_CHECK(cudaGraphAddKernelNode(
        &apply_node_,
        graph_,
        deps,
        1,
        &apply_params_));
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;

  cudaGraphNode_t infer_node_ = nullptr;
  cudaGraphNode_t apply_node_ = nullptr;

  cudaKernelNodeParams infer_params_{};
  cudaKernelNodeParams apply_params_{};

  void* infer_args_[12]{};
  void* apply_args_[9]{};

  const T* y_std_ = nullptr;
  const float* g_value_ = nullptr;
  T* y_ = nullptr;

  float* infer_std_ = nullptr;
  float* error_bias_ = nullptr;

  int64_t N_ = 1;
  int64_t numel_ = 0;
  int64_t Cout_ = 0;
  int64_t HW_ = 1;
  int64_t g_count_ = 1;
  int64_t linear_outer_ = 1;

  float a_minus_b_ = 0.f;
  float b_ = 0.f;
  float noise_scale_ = 1.f;
  float sigma_bias_ = 0.f;
  float k_ = 0.f;
  float x0_half_ = 0.f;

  NoiseFunc func_ = NoiseFunc::Sech;
  OperatorType op_ = OperatorType::Conv;

  uint64_t bias_seed_ = 0;
  uint64_t noise_seed_ = 1;

  unsigned int blocks_c_ = 0;
  unsigned int threads_ = 256;
};

template <typename T, bool EnablePost>
void run_case(const char* name, OperatorType op) {
  constexpr int64_t N = 32;
  constexpr int64_t C = 256;
  constexpr int64_t H = 64;
  constexpr int64_t W = 64;

  int64_t HW = (op == OperatorType::Linear) ? 1 : H * W;
  int64_t numel = (op == OperatorType::Linear)
      ? N * C
      : N * C * H * W;

  std::vector<T> h_y(numel);
  std::vector<T> h_y_std(C);
  std::vector<float> h_g(C);

  for (int64_t i = 0; i < numel; ++i) {
    h_y[i] = static_cast<T>((i % 255) - 128);
  }

  for (int64_t c = 0; c < C; ++c) {
    h_y_std[c] = static_cast<T>(0.1f + 0.001f * c);
    h_g[c] = 1.0f + 0.0001f * c;
  }

  T* d_y = nullptr;
  T* d_y_std = nullptr;
  float* d_g = nullptr;
  float* d_workspace = nullptr;

  CUDA_CHECK(cudaMalloc(&d_y, numel * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_y_std, C * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_g, C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_workspace,
      NoiseGraph<T, EnablePost>::WorkspaceBytes(C)));

  CUDA_CHECK(cudaMemcpy(
      d_y,
      h_y.data(),
      numel * sizeof(T),
      cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(
      d_y_std,
      h_y_std.data(),
      C * sizeof(T),
      cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(
      d_g,
      h_g.data(),
      C * sizeof(float),
      cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  NoiseGraph<T, EnablePost> graph;

  graph.Build(
      d_y_std,
      d_g,
      d_y,
      d_workspace,
      N,
      numel,
      C,
      HW,
      C,
      1.0f,
      0.7f,
      0.01f,
      1.0f,
      1.0f,
      0.02f,
      NoiseFunc::Sech,
      op,
      1234);

  for (int i = 0; i < 10; ++i) {
    graph.UpdateSeeds(1234 + i);
    graph.Launch(stream);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start;
  cudaEvent_t stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int iters = 100;

  CUDA_CHECK(cudaEventRecord(start, stream));

  for (int i = 0; i < iters; ++i) {
    graph.UpdateSeeds(9999 + i);
    graph.Launch(stream);
  }

  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  float avg_ms = total_ms / iters;
  double elems_per_sec =
      static_cast<double>(numel) / (avg_ms * 1e-3);

  printf("[%s | EnablePost=%d]\n", name, EnablePost ? 1 : 0);
  printf("numel: %lld\n", static_cast<long long>(numel));
  printf("avg time: %.6f ms\n", avg_ms);
  printf("throughput: %.3f Gelem/s\n\n", elems_per_sec / 1e9);

  graph.Destroy();

  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_y_std));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_workspace));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
  run_case<float, false>("linear fused graph", OperatorType::Linear);
  run_case<float, true>("linear fused graph", OperatorType::Linear);

  run_case<float, false>("conv fused graph", OperatorType::Conv);
  run_case<float, true>("conv fused graph", OperatorType::Conv);

  return 0;
}