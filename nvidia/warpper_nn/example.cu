#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "cuda_nn/inline_check.h"
#include "cuda_nn/cublaslt_gemm.h"
#include "cuda_nn/cudnn_legacy_conv.h"
#include "cuda_nn/cudnn_frontend_conv.h"

using cuda_nn::GemmEpilogue;
using cuda_nn::Activation;
using cuda_nn::CublasLtGemm;
using cuda_nn::CublasLtRowMajorGemm;
using cuda_nn::CudnnFrontendConv;
using cuda_nn::CudnnFrontendConvTranspose;
using cuda_nn::ConvConfig;
using cuda_nn::MemoryFormat;

using cuda_nn::CudnnLegacyConv;
using cuda_nn::CudnnLegacyConvTranspose;

static void fill_random(std::vector<float>& v, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) {
        x = dist(gen);
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("size mismatch in max_abs_diff");
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

static size_t volume(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto dim : shape) {
        n *= static_cast<size_t>(dim);
    }
    return n;
}

static std::vector<int64_t> strides_for_format(const std::vector<int64_t>& dims,
                                               MemoryFormat memory_format) {
    std::vector<int64_t> strides(dims.size());
    if (memory_format == MemoryFormat::Contiguous) {
        strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
        }
        return strides;
    }

    strides[1] = 1;
    strides[dims.size() - 1] = dims[1];
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 2; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    strides[0] = dims[1];
    for (size_t i = 2; i < dims.size(); ++i) {
        strides[0] *= dims[i];
    }
    return strides;
}

static size_t tensor_offset(const std::vector<int64_t>& strides,
                            int64_t a,
                            int64_t b,
                            int64_t c,
                            int64_t d) {
    return static_cast<size_t>(a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3]);
}

static void check_result(
    const std::string& name,
    const std::vector<float>& got,
    const std::vector<float>& ref,
    float tol
) {
    float diff = max_abs_diff(got, ref);
    std::cout << name << " max_abs_diff = " << diff << "\n";
    if (diff > tol) {
        throw std::runtime_error(name + " correctness check failed");
    }
}

static void conv2d_ref_nchw(
    const ConvConfig& cfg,
    const std::vector<float>& x,
    const std::vector<float>& w,
    const std::vector<float>& bias,
    std::vector<float>& y
) {
    const int64_t N = cfg.x[0];
    const int64_t C = cfg.x[1];
    const int64_t H = cfg.x[2];
    const int64_t W = cfg.x[3];
    const int64_t K = cfg.w[0];
    const int64_t R = cfg.w[2];
    const int64_t S = cfg.w[3];
    const int64_t P = (H + 2 * cfg.padding[0] - cfg.dilation[0] * (R - 1) - 1) / cfg.stride[0] + 1;
    const int64_t Q = (W + 2 * cfg.padding[1] - cfg.dilation[1] * (S - 1) - 1) / cfg.stride[1] + 1;
    const auto x_strides = strides_for_format(cfg.x, cfg.memory_format);
    const auto w_strides = strides_for_format(cfg.w, cfg.memory_format);
    const auto y_strides = strides_for_format({N, K, P, Q}, cfg.memory_format);

    y.assign(static_cast<size_t>(N * K * P * Q), 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t p = 0; p < P; ++p) {
                for (int64_t q = 0; q < Q; ++q) {
                    float acc = cfg.with_bias ? bias[static_cast<size_t>(k)] : 0.0f;
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t r = 0; r < R; ++r) {
                            const int64_t ih = p * cfg.stride[0] - cfg.padding[0] + r * cfg.dilation[0];
                            if (ih < 0 || ih >= H) {
                                continue;
                            }
                            for (int64_t s = 0; s < S; ++s) {
                                const int64_t iw = q * cfg.stride[1] - cfg.padding[1] + s * cfg.dilation[1];
                                if (iw < 0 || iw >= W) {
                                    continue;
                                }
                                const float xv = x[tensor_offset(x_strides, n, c, ih, iw)];
                                const float wv = w[tensor_offset(w_strides, k, c, r, s)];
                                acc += xv * wv;
                            }
                        }
                    }
                    if (cfg.activation == Activation::Relu && acc < 0.0f) {
                        acc = 0.0f;
                    }
                    y[tensor_offset(y_strides, n, k, p, q)] = acc;
                }
            }
        }
    }
}

static void conv_transpose2d_ref_nchw(
    const ConvConfig& cfg,
    const std::vector<float>& x,
    const std::vector<float>& w,
    const std::vector<float>& bias,
    std::vector<float>& y
) {
    const int64_t N = cfg.x[0];
    const int64_t IC = cfg.x[1];
    const int64_t H = cfg.x[2];
    const int64_t W = cfg.x[3];
    const int64_t OC = cfg.w[1] * cfg.groups;
    const int64_t R = cfg.w[2];
    const int64_t S = cfg.w[3];
    const int64_t OH = (H - 1) * cfg.stride[0] - 2 * cfg.padding[0] +
                       cfg.dilation[0] * (R - 1) +
                       (cfg.output_padding.empty() ? 0 : cfg.output_padding[0]) + 1;
    const int64_t OW = (W - 1) * cfg.stride[1] - 2 * cfg.padding[1] +
                       cfg.dilation[1] * (S - 1) +
                       (cfg.output_padding.empty() ? 0 : cfg.output_padding[1]) + 1;
    const auto x_strides = strides_for_format(cfg.x, cfg.memory_format);
    const auto w_strides = strides_for_format(cfg.w, cfg.memory_format);
    const auto y_strides = strides_for_format({N, OC, OH, OW}, cfg.memory_format);

    y.assign(static_cast<size_t>(N * OC * OH * OW), 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t ih = 0; ih < H; ++ih) {
                for (int64_t iw = 0; iw < W; ++iw) {
                    const float xv = x[tensor_offset(x_strides, n, ic, ih, iw)];
                    for (int64_t oc = 0; oc < OC; ++oc) {
                        for (int64_t r = 0; r < R; ++r) {
                            const int64_t oh = ih * cfg.stride[0] - cfg.padding[0] + r * cfg.dilation[0];
                            if (oh < 0 || oh >= OH) {
                                continue;
                            }
                            for (int64_t s = 0; s < S; ++s) {
                                const int64_t ow = iw * cfg.stride[1] - cfg.padding[1] + s * cfg.dilation[1];
                                if (ow < 0 || ow >= OW) {
                                    continue;
                                }
                                const float wv = w[tensor_offset(w_strides, ic, oc, r, s)];
                                y[tensor_offset(y_strides, n, oc, oh, ow)] += xv * wv;
                            }
                        }
                    }
                }
            }
        }
    }

    if (cfg.with_bias) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc = 0; oc < OC; ++oc) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        y[tensor_offset(y_strides, n, oc, oh, ow)] += bias[static_cast<size_t>(oc)];
                    }
                }
            }
        }
    }
}

static float benchmark_ms(
    cudaStream_t stream,
    int warmup,
    int iters,
    const std::function<void()>& fn
) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / static_cast<float>(iters);
}

// Column-major CPU reference:
// D[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
static void gemm_ref_col_major(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    std::vector<float>& D,
    float alpha,
    float beta
) {
    D.assign(static_cast<size_t>(m * n), 0.0f);

    for (int64_t col = 0; col < n; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            float acc = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                float a = A[static_cast<size_t>(row + p * m)];
                float b = B[static_cast<size_t>(p + col * k)];
                acc += a * b;
            }

            D[static_cast<size_t>(row + col * m)] =
                alpha * acc + beta * C[static_cast<size_t>(row + col * m)];
        }
    }
}

// Row-major CPU reference:
// D[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
static void gemm_ref_row_major(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    std::vector<float>& D,
    float alpha,
    float beta
) {
    D.assign(static_cast<size_t>(m * n), 0.0f);

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                float a = A[static_cast<size_t>(row * k + p)];
                float b = B[static_cast<size_t>(p * n + col)];
                acc += a * b;
            }

            D[static_cast<size_t>(row * n + col)] =
                alpha * acc + beta * C[static_cast<size_t>(row * n + col)];
        }
    }
}

static void test_gemm(cudaStream_t stream, int device_id) {
    const int64_t M = 512;
    const int64_t N = 256;
    const int64_t K = 1024;

    std::vector<float> h_A(static_cast<size_t>(M * K));
    std::vector<float> h_B(static_cast<size_t>(K * N));
    std::vector<float> h_C(static_cast<size_t>(M * N));
    std::vector<float> h_D(static_cast<size_t>(M * N));
    std::vector<float> h_bias(static_cast<size_t>(M));
    std::vector<float> h_ref;

    fill_random(h_A, -0.5f, 0.5f);
    fill_random(h_B, -0.5f, 0.5f);
    fill_random(h_C, -0.5f, 0.5f);
    fill_random(h_bias, -0.5f, 0.5f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_D = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_D, h_D.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    float* d_bias = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    CublasLtGemm<float> gemm(M, N, K, device_id, stream, cuda_nn::GemmEpilogue::Bias);

    float alpha = 1.25f;
    float beta = 0.5f;

    gemm.run(d_A, d_B, d_C, d_D, d_bias, &alpha, &beta);

    CHECK_CUDA(cudaMemcpyAsync(h_D.data(), d_D, h_D.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    gemm_ref_col_major(M, N, K, h_A, h_B, h_C, h_ref, alpha, beta);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            h_ref[static_cast<size_t>(m + n * M)] += h_bias[static_cast<size_t>(m)];
        }
    }
    check_result("GEMM", h_D, h_ref, 2e-3f);

    float avg_ms = benchmark_ms(stream, 10, 100, [&] {
        gemm.run(d_A, d_B, d_C, d_D, d_bias, &alpha, &beta);
    });

    std::cout << "GEMM avg_ms = " << avg_ms << "\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_bias));
}

static void test_row_major_gemm(cudaStream_t stream, int device_id) {
    const int64_t M = 512;
    const int64_t N = 256;
    const int64_t K = 1024;

    std::vector<float> h_A(static_cast<size_t>(M * K));
    std::vector<float> h_B(static_cast<size_t>(K * N));
    std::vector<float> h_C(static_cast<size_t>(M * N));
    std::vector<float> h_D(static_cast<size_t>(M * N));
    std::vector<float> h_bias(static_cast<size_t>(N));
    std::vector<float> h_ref;

    fill_random(h_A, -0.5f, 0.5f);
    fill_random(h_B, -0.5f, 0.5f);
    fill_random(h_C, -0.5f, 0.5f);
    fill_random(h_bias, -0.5f, 0.5f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_D = nullptr;
    float* d_bias = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_D, h_D.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    CublasLtRowMajorGemm<float> gemm(M, N, K, device_id, stream, GemmEpilogue::Bias);

    float alpha = 1.25f;
    float beta = 0.5f;

    gemm.run(d_A, d_B, d_C, d_D, d_bias, &alpha, &beta);

    CHECK_CUDA(cudaMemcpyAsync(h_D.data(), d_D, h_D.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    gemm_ref_row_major(M, N, K, h_A, h_B, h_C, h_ref, alpha, beta);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            h_ref[static_cast<size_t>(m * N + n)] += h_bias[static_cast<size_t>(n)];
        }
    }
    check_result("RowMajor GEMM", h_D, h_ref, 2e-3f);

    float avg_ms = benchmark_ms(stream, 10, 100, [&] {
        gemm.run(d_A, d_B, d_C, d_D, d_bias, &alpha, &beta);
    });

    std::cout << "RowMajor GEMM avg_ms = " << avg_ms << "\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
}

static void test_conv_demo(cudaStream_t stream, int device_id) {
    ConvConfig cfg;
    cfg.x = {4, 32, 32, 32};
    cfg.w = {64, 32, 3, 3};
    cfg.padding = {1, 1};
    cfg.stride = {1, 1};
    cfg.dilation = {1, 1};
    cfg.groups = 1;
    cfg.memory_format = MemoryFormat::ChannelsLast;
    cfg.with_bias = true;
    cfg.activation = Activation::Relu;

    const std::vector<int64_t> y_shape = {4, 64, 32, 32};

    std::vector<float> h_x(volume(cfg.x));
    std::vector<float> h_w(volume(cfg.w));
    std::vector<float> h_bias(static_cast<size_t>(cfg.w[0]));
    std::vector<float> h_y(volume(y_shape));
    std::vector<float> h_ref;

    fill_random(h_x, -0.5f, 0.5f);
    fill_random(h_w, -0.5f, 0.5f);
    fill_random(h_bias, -0.1f, 0.1f);

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_bias = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    CudnnFrontendConv<float> conv(device_id, cfg, stream);
    conv.run(d_x, d_w, d_bias, d_y);

    CHECK_CUDA(cudaMemcpyAsync(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    conv2d_ref_nchw(cfg, h_x, h_w, h_bias, h_ref);
    check_result("Conv2D bias relu large", h_y, h_ref, 2e-3f);
    std::cout << "Conv2D bias relu large workspace_bytes = " << conv.workspace_size() << "\n";

    float avg_ms = benchmark_ms(stream, 10, 100, [&] {
        conv.run(d_x, d_w, d_bias, d_y);
    });

    std::cout << "Conv2D bias relu large avg_ms = " << avg_ms << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_y));
}

static void test_conv_demo_contiguous(cudaStream_t stream, int device_id) {
    ConvConfig cfg;
    cfg.x = {4, 32, 32, 32};
    cfg.w = {64, 32, 3, 3};
    cfg.padding = {1, 1};
    cfg.stride = {1, 1};
    cfg.dilation = {1, 1};
    cfg.groups = 1;
    cfg.memory_format = MemoryFormat::Contiguous;
    cfg.with_bias = true;
    cfg.activation = Activation::Relu;

    const std::vector<int64_t> y_shape = {4, 64, 32, 32};

    std::vector<float> h_x(volume(cfg.x));
    std::vector<float> h_w(volume(cfg.w));
    std::vector<float> h_bias(static_cast<size_t>(cfg.w[0]));
    std::vector<float> h_y(volume(y_shape));
    std::vector<float> h_ref;

    fill_random(h_x, -0.5f, 0.5f);
    fill_random(h_w, -0.5f, 0.5f);
    fill_random(h_bias, -0.1f, 0.1f);

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_bias = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    CudnnFrontendConv<float> conv(device_id, cfg, stream);
    conv.run(d_x, d_w, d_bias, d_y);

    CHECK_CUDA(cudaMemcpyAsync(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    conv2d_ref_nchw(cfg, h_x, h_w, h_bias, h_ref);
    check_result("Conv2D bias relu NCHW large", h_y, h_ref, 2e-3f);

    std::cout << "Conv2D bias relu NCHW large workspace_bytes = "
              << conv.workspace_size() << "\n";

    float avg_ms = benchmark_ms(stream, 10, 100, [&] {
        conv.run(d_x, d_w, d_bias, d_y);
    });

    std::cout << "Conv2D bias relu NCHW large avg_ms = "
              << avg_ms << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_y));
}

static void test_conv_transpose_demo(cudaStream_t stream, int device_id) {
    ConvConfig cfg;
    cfg.x = {4, 32, 32, 32};
    cfg.w = {32, 64, 3, 3};
    cfg.padding = {1, 1};
    cfg.stride = {1, 1};
    cfg.dilation = {1, 1};
    cfg.output_padding = {0, 0};
    cfg.groups = 1;
    cfg.memory_format = MemoryFormat::ChannelsLast;
    cfg.with_bias = true;
    cfg.activation = Activation::None;

    const std::vector<int64_t> y_shape = {4, 64, 32, 32};

    std::vector<float> h_x(volume(cfg.x));
    std::vector<float> h_w(volume(cfg.w));
    std::vector<float> h_bias(static_cast<size_t>(y_shape[1]));
    std::vector<float> h_y(volume(y_shape));
    std::vector<float> h_ref;

    fill_random(h_x, -0.5f, 0.5f);
    fill_random(h_w, -0.5f, 0.5f);
    fill_random(h_bias, -0.1f, 0.1f);

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_bias = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    CudnnFrontendConvTranspose<float> conv_t(device_id, cfg, stream);
    conv_t.run(d_x, d_w, d_bias, d_y);

    CHECK_CUDA(cudaMemcpyAsync(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    conv_transpose2d_ref_nchw(cfg, h_x, h_w, h_bias, h_ref);
    check_result("ConvTranspose2D+bias NHWC large", h_y, h_ref, 2e-3f);
    std::cout << "ConvTranspose2D+bias NHWC large workspace_bytes = " << conv_t.workspace_size() << "\n";

    float avg_ms = benchmark_ms(stream, 10, 100, [&] {
        conv_t.run(d_x, d_w, d_bias, d_y);
    });

    std::cout << "ConvTranspose2D+bias NHWC large avg_ms = " << avg_ms << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_y));
}

static void test_conv_transpose_demo_contiguous(cudaStream_t stream, int device_id) {
    ConvConfig cfg;
    cfg.x = {4, 32, 32, 32};
    cfg.w = {32, 64, 3, 3};
    cfg.padding = {1, 1};
    cfg.stride = {1, 1};
    cfg.dilation = {1, 1};
    cfg.output_padding = {0, 0};
    cfg.groups = 1;
    cfg.memory_format = MemoryFormat::Contiguous;
    cfg.with_bias = true;
    cfg.activation = Activation::None;

    const std::vector<int64_t> y_shape = {4, 64, 32, 32};

    std::vector<float> h_x(volume(cfg.x));
    std::vector<float> h_w(volume(cfg.w));
    std::vector<float> h_bias(volume(y_shape));
    std::vector<float> h_y(volume(y_shape));
    std::vector<float> h_ref;

    fill_random(h_x, -0.5f, 0.5f);
    fill_random(h_w, -0.5f, 0.5f);
    fill_random(h_bias, -0.1f, 0.1f);

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_bias = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    CudnnLegacyConvTranspose<float> conv_t(device_id, cfg, stream);
    conv_t.run(d_x, d_w, d_bias, d_y);

    CHECK_CUDA(cudaMemcpyAsync(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    conv_transpose2d_ref_nchw(cfg, h_x, h_w, h_bias, h_ref);
    check_result("ConvTranspose2D+bias NCHW large", h_y, h_ref, 2e-3f);

    std::cout << "ConvTranspose2D+bias NCHW large workspace_bytes = "
              << conv_t.workspace_size() << "\n";

    float avg_ms = benchmark_ms(stream, 10, 100, [&] {
        conv_t.run(d_x, d_w, d_bias, d_y);
    });

    std::cout << "ConvTranspose2D+bias NCHW large avg_ms = "
              << avg_ms << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_y));
}

int main() {
    try {
        int device_id = 0;
        CHECK_CUDA(cudaGetDevice(&device_id));

        cudaStream_t stream = nullptr;
        CHECK_CUDA(cudaStreamCreate(&stream));

        std::cout << "CUDA/cuBLASLt/cuDNN wrapper example\n";

        test_gemm(stream, device_id);
        test_row_major_gemm(stream, device_id);
        test_conv_demo(stream, device_id);
        test_conv_demo_contiguous(stream, device_id);
        test_conv_transpose_demo(stream, device_id);
        test_conv_transpose_demo_contiguous(stream, device_id);

        CHECK_CUDA(cudaStreamDestroy(stream));

        std::cout << "All tests passed.\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
