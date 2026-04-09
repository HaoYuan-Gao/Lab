// file: pack_demo.cu
#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <cuda_runtime.h>

// ----------------------- utils -----------------------
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Host: float bits -> ordered uint32 (same mapping as device)
static inline uint32_t host_float_to_ordered_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // map sign so unsigned comparison matches float comparison
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

static inline float host_ordered_uint_to_float(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ----------------------- device helpers -----------------------
__device__ __forceinline__ uint32_t float_to_ordered_uint(float f) {
    uint32_t u = __float_as_uint(f);

    // 目的是确保比较的时候，负数不会变成大的值
    return (u & 0x80000000u) ? ~u /*负数*/ : (u ^ 0x80000000u) /*正数*/;
}

// atomic update packed (ordered_key<<32 | idx)
__device__ __forceinline__
void atomic_min_pack_f32_idx(float* rms_and_idx, float val, uint32_t idx) {
    // pointer must be 8-byte aligned
    unsigned long long* addr = reinterpret_cast<unsigned long long*>(rms_and_idx);

    uint32_t new_key = float_to_ordered_uint(val);
    uint64_t new_pack = ( (uint64_t)new_key << 32 ) | (uint64_t)idx;

    // atomic read current value
    unsigned long long old_pack = atomicAdd(addr, 0ull);
    while (true) {
        uint32_t old_key = (uint32_t)(old_pack >> 32);
        // if old_key <= new_key then old <= new, so nothing to do
        if (!(new_key < old_key)) break;

        unsigned long long prev = atomicCAS(addr, old_pack, new_pack);
        if (prev == old_pack) break; // success
        old_pack = prev; // retry with updated old
    }
}

// ----------------------- kernel -----------------------
/*
 Each thread generates a candidate (val, idx) and tries to update global packed winner.
 We'll design values so that one thread has the minimum.
*/
__global__ void race_kernel(float* out_pack_as_floatptr, int threads_per_block) {
    // each thread's idx (unique across grid)
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // design value: e.g., val = base - tid * 0.01f, so larger tid => smaller val
    float base = 1000.0f;
    float val = base - (float)tid * 0.01f;

    // to create some duplicates and out-of-order, mod by some number
    // but it's fine as is.
    atomic_min_pack_f32_idx(out_pack_as_floatptr, val, tid);
}

// ----------------------- main -----------------------
int main() {
    printf("atomic_min_pack_f32_idx demo\n");

    // allocate one 64-bit pack on device (we will treat as float[2] pointer in device function)
    uint64_t host_init_pack;
    // initial key = ordered(FLT_MAX), idx = 0xFFFFFFFF
    uint32_t init_key = host_float_to_ordered_uint(FLT_MAX);
    uint32_t init_idx = 0xFFFFFFFFu;
    host_init_pack = ( (uint64_t)init_key << 32 ) | (uint64_t)init_idx;

    unsigned long long* d_pack = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pack), sizeof(uint64_t)));

    // copy initial pack to device
    CUDA_CHECK(cudaMemcpy(d_pack, &host_init_pack, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // launch many threads to race
    int blocks = 128;
    int threads = 256;
    int total_threads = blocks * threads;
    printf("Launching %d blocks x %d threads = %d threads\n", blocks, threads, total_threads);

    // Launch kernel
    race_kernel<<<blocks, threads>>>(reinterpret_cast<float*>(d_pack), threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back pack
    uint64_t host_result_pack = 0;
    CUDA_CHECK(cudaMemcpy(&host_result_pack, d_pack, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // unpack
    uint32_t res_key = (uint32_t)(host_result_pack >> 32);
    uint32_t res_idx = (uint32_t)(host_result_pack & 0xffffffffu);
    float res_val = host_ordered_uint_to_float(res_key);

    printf("Result packed = 0x%016llx\n", (unsigned long long)host_result_pack);
    printf("Recovered min value = %.9f, idx = %u\n", res_val, res_idx);

    // expected: the largest tid (total_threads-1) produces the smallest val (base - tid*0.01)
    uint32_t expected_idx = total_threads - 1;
    float expected_val = 1000.0f - (float)expected_idx * 0.01f;
    printf("Expected min value = %.9f, expected idx = %u\n", expected_val, expected_idx);

    // cleanup
    CUDA_CHECK(cudaFree(d_pack));
    return 0;
}
