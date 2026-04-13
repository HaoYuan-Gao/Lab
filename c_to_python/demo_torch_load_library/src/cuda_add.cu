// nvcc -O3 -Xcompiler -fPIC -shared ./src/cuda_add.cu -o ./lib/libcuda_add.so

#include <cuda_runtime.h>
#include <cstdint>

// ----------------------------------------
// CUDA kernel
// ----------------------------------------
__global__ void sample_add_kernel(
    const float* x,
    const float* y,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

// ----------------------------------------
// 工具函数：stream 句柄转换
// ----------------------------------------
static inline cudaStream_t as_cuda_stream(unsigned long long stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

// ----------------------------------------
// 参数检查
// ----------------------------------------
static inline int validate_args(
    const float* x,
    const float* y,
    float* out,
    int n
) {
    if (x == nullptr || y == nullptr || out == nullptr) {
        return static_cast<int>(cudaErrorInvalidDevicePointer);
    }
    if (n < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return 0;
}

// ----------------------------------------
// 异步版本：只发射 kernel，不等待执行完成
//
// 返回值：
//   0    成功
//   非 0 CUDA 错误码
// ----------------------------------------
extern "C" int sample_add_cuda(
    const float* x,
    const float* y,
    float* out,
    int n,
    int device,
    unsigned long long stream
) {
    int rc = validate_args(x, y, out, n);
    if (rc != 0) {
        return rc;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t cuda_stream = as_cuda_stream(stream);

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    if (blocks == 0) {
        return 0;
    }

    sample_add_kernel<<<blocks, threads, 0, cuda_stream>>>(x, y, out, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return 0;
}

// ----------------------------------------
// 同步版本：发射 kernel 后等待当前 stream 完成
//
// 适合调试，不适合正式性能路径
// ----------------------------------------
extern "C" int sample_add_cuda_sync(
    const float* x,
    const float* y,
    float* out,
    int n,
    int device,
    unsigned long long stream
) {
    int rc = sample_add_cuda(x, y, out, n, device, stream);
    if (rc != 0) {
        return rc;
    }

    cudaStream_t cuda_stream = as_cuda_stream(stream);
    cudaError_t err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return 0;
}

// ----------------------------------------
// 错误码转字符串
// Python 侧可通过 ctypes 调用
// ----------------------------------------
extern "C" const char* sample_cuda_get_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

// ----------------------------------------
// 返回 CUDA runtime version
// 调试环境时可能有用
// ----------------------------------------
extern "C" int sample_cuda_runtime_version() {
    int version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&version);
    if (err != cudaSuccess) {
        return -1;
    }
    return version;
}