#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

py::array_t<float> add(py::array_t<float> a_np, py::array_t<float> b_np) {
    py::buffer_info a_buf = a_np.request(), b_buf = b_np.request();

    if (a_buf.size != b_buf.size) {
        throw std::runtime_error("Mismatched sizes");
    }

    int n = a_buf.size;
    float *a_host = static_cast<float*>(a_buf.ptr);
    float *b_host = static_cast<float*>(b_buf.ptr);

    // 分配 GPU 内存
    float *a_device, *b_device, *out_device;
    cudaMalloc(&a_device,   n * sizeof(float));
    cudaMalloc(&b_device,   n * sizeof(float));
    cudaMalloc(&out_device, n * sizeof(float));

    // 拷贝输入数据到 GPU
    cudaMemcpy(a_device, a_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, n * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    add_kernel<<<grid, block>>>(a_device, b_device, out_device, n);

    // 拷贝输出数据到 Host
    auto result = py::array_t<float>(n);
    cudaMemcpy(result.mutable_data(), out_device, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(out_device);

    return result;
}

/**
 * @brief Construct a new pybind11 module object
 * 
 * @param name module name, such as "demo_pybind11"
 * @param m pybind11 module object
 */
PYBIND11_MODULE(demo_pybind11, m) {
    m.def("add", &add, "Add two arrays via GPU");
}