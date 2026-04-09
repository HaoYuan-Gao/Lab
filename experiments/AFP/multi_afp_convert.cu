// quant_convert.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#ifndef MAX_PAIRS
#define MAX_PAIRS 8
#endif

struct QuantParams {
    int M;
    int S;
    int N;
    int mask_bits;
    int group_up;
    int mantissa_min;
    int mantissa_max;
    int limit; // 2^(M + (S * (N - 1))
    float scale_factor; // limit - 1
    float inv_sf;
};

// clamp helper
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x)); 
}
__device__ __forceinline__ int clampi(int x, int lo, int hi) {
    return max(lo, min(hi, x));
}

__device__ __forceinline__ int floor_log2_u32(uint32_t x) {
    return 31 - __clz(x);
}

__device__ __forceinline__ int fast_round_shift(int q, int e) {
    if (e <= 0) return q;
    float r = ldexpf(q, -e);
    return __float2int_rn(r);
}

__device__ __forceinline__ int udiv_s(int n, int S) {
    // The value of S is: 1, 2, 3, 4
    if (n < 0) return 0;
    // dispatch case
    switch (S) {
        case 1: return (int)n;
        case 2: return (int)(n >> 1);
        case 3: {
            // (3) magic number div
            uint32_t hi = __umulhi(n, 0xAAAAAAABu);
            return (int)(hi >> 1);
        }
        case 4: return (int)(n >> 2);
        default: return (int)(n / S);
    }
}

__device__ __forceinline__
float convert_single(float value, const QuantParams& p, int* out_m, int* out_e) {
    value = clampf(value, -1.f, 1.f);
    if (fabsf(value) < 1e-10f) {
        if (out_m) *out_m = 0;
        if (out_e) *out_e = 0;
        return 0.f;
    }

    int q = __float2int_rn(value * p.scale_factor);
    if (q ==  p.limit) q -= 1;
    if (q == -p.limit) q += 1;

    uint32_t aq = (uint32_t)abs(q);
    int sig = (aq == 0) ? 0 : floor_log2_u32(aq);

    int exponent_step;
    if (p.group_up) {
        exponent_step = udiv_s(sig, p.S);
        if (exponent_step >= p.N) exponent_step = p.N - 1;
    } else {
        exponent_step = udiv_s(sig - p.M + p.S, p.S);
        if (exponent_step < 0) exponent_step = 0;
    }

    int e = p.S * exponent_step;
    int m = fast_round_shift(q, e);
    m = clampi(m, p.mantissa_min, p.mantissa_max);

    if (out_m) *out_m = m;
    if (out_e) *out_e = e;

    return ldexpf(m, e) * p.inv_sf;
}

__device__ __forceinline__
float convert_multi(float value, const QuantParams& p, int* out_count, int* out_m, int* out_e) {
    value = clampf(value, -1.f, 1.f);
    if (fabsf(value) < 1e-10f) {
        if (out_count) *out_count = 0;
        return 0.f;
    }

    int q = __float2int_rn(value * p.scale_factor);
    if (q ==  p.limit) q -= 1;
    if (q == -p.limit) q += 1;

    uint32_t aq = (uint32_t)abs(q);
    int sig = (aq == 0) ? 0 : floor_log2_u32(aq);

    int exponent_step;
    if (p.group_up) {
        exponent_step = udiv_s(sig, p.S);
        if (exponent_step >= p.N) exponent_step = p.N - 1;
    } else {
        exponent_step = udiv_s(sig - p.M + p.S, p.S);
        if (exponent_step < 0) exponent_step = 0;
    }

    int count = 0;
    int remaining = q;
    float acc = 0.f;
    for (int try_step = exponent_step; try_step >= 0; --try_step) {
        if (remaining == 0) break;
        if (count >= MAX_PAIRS) break;

        int e = p.S * try_step;
        int m = fast_round_shift(remaining, e);
        m = clampi(m, p.mantissa_min, p.mantissa_max);

        // groups > 0 mask
        if (try_step > 0) {
            m = (m >> p.mask_bits) << p.mask_bits;
        }

        if (m != 0) {
            out_m[count] = m;
            out_e[count] = e;
            count++;

            int represented = m * (1 << e);
            remaining -= represented;

            // sum
            acc += ldexpf(m, e);
        }
    }

    if (out_count) *out_count = count;
    return acc * p.inv_sf;
}

__global__
void quant_convert_kernel(
    const float* __restrict__ in,
    float* __restrict__ out_quant,
    // single output
    int* __restrict__ out_mantissa_single,
    int* __restrict__ out_exponent_single,
    // multi-pair output
    int* __restrict__ out_count,
    int* __restrict__ out_mantissa_multi, // layout: [numle * MAX_PAIRS]
    int* __restrict__ out_exponent_multi, // layout: [numle * MAX_PAIRS]
    int n,
    QuantParams p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = in[idx];

    if (p.mask_bits == 0) {
        int m, e;
        float qv = convert_single(x, p, &m, &e);
        out_quant[idx] = qv;
        if (out_mantissa_single) out_mantissa_single[idx] = m;
        if (out_exponent_single) out_exponent_single[idx] = e;
    } else {
        int* mptr = out_mantissa_multi ? (out_mantissa_multi + idx * MAX_PAIRS) : nullptr;
        int* eptr = out_exponent_multi ? (out_exponent_multi + idx * MAX_PAIRS) : nullptr;
        int c = 0;
        float qv = convert_multi(x, p, &c, mptr, eptr);
        out_quant[idx] = qv;
        if (out_count) out_count[idx] = c;
    }
}

void multi_afp_convert(
    const float* d_in,
    float* d_out,

    // mask_bits == 0
    int* d_m_single,
    int* d_e_single,

    // mask_bits > 0
    int* d_count,
    int* d_m_multi,
    int* d_e_multi,

    // numel of array
    int n,

    // Quant Params
    int M,
    int S,
    int N,
    int group_up,
    int mask_bits,
    int mantissa_min,
    int mantissa_max,

    cudaStream_t stream = 0
) {
    // pre-compute limit
    int limit = 1 << (M + (S * (N - 1)));
    float scale = limit - 1;
    QuantParams p {
        M, S, N,
        mask_bits,
        group_up,
        mantissa_min,
        mantissa_max,
        limit,
        scale,
        float(1.f / scale)
    };

    // kernel launch
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    quant_convert_kernel<<<blocks, threads, 0, stream>>>(
        d_in, d_out,
        d_m_single, d_e_single,
        d_count, d_m_multi, d_e_multi,
        n, p
    );
}