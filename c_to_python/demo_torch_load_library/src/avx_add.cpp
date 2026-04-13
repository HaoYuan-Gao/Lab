// g++ -O3 -mavx -fPIC -shared ./src/avx_add.cpp -o ./lib/libavx_add.so

#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

// AVX 向量加法
extern "C" void sample_add_avx(float* x, float* y, float* out, int n) {
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vout = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(out + i, vout);
    }

    for (; i < n; i++) {
        out[i] = x[i] + y[i];
    }
}

float* alloc_aligned(int n) {
    float* ptr = nullptr;
    if (posix_memalign((void**)&ptr, 32, n * sizeof(float)) != 0) {
        return nullptr;
    }
    return ptr;
}

int main() {
    const int n = 1 << 20; // 1M 数据

    // 分配内存（32字节对齐更好）
    float* x = alloc_aligned(n);
    float* y = alloc_aligned(n);
    float* out = alloc_aligned(n);

    if (!x || !y || !out) {
        std::cerr << "alloc failed\n";
        return -1;
    }

    // 初始化数据
    for (int i = 0; i < n; i++) {
        x[i] = i * 0.5f;
        y[i] = i * 2.0f;
    }

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    sample_add_avx(x, y, out, n);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    // 校验结果
    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = x[i] + y[i];
        if (std::fabs(out[i] - expected) > 1e-5) {
            std::cout << "Error at " << i << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Result correct ✅" << std::endl;
    }

    std::cout << "Time: " << duration << " ms" << std::endl;

    // 打印前几个结果看看
    for (int i = 0; i < 5; i++) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    // 释放内存
    free(x);
    free(y);
    free(out);

    return 0;
}