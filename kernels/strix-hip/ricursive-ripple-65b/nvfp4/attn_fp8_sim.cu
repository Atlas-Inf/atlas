// SPDX-License-Identifier: AGPL-3.0-only

// Ripple FP8 attention, accuracy slice (ATLAS_RIPPLE_ATTN_FP8_SIM=1): round-trip
// each BF16 attention projection row through FP8 E4M3 with a per-row f32 scale,
// in place, at load. The quantizer is quantize_bf16_to_fp8's (dense_gemv_fp8w.cu):
// scale = amax / 448, x * (1 / scale) clamped to +-448, the same software e4m3
// encode. The dequantized value is rounded once more to BF16, so this is never
// more accurate than the FP8 kernels it stands in for. The point is to answer
// ST-995 before the FP8 decode / verify / prefill routes exist.
//
// Grid: (N, 1, 1)  Block: (256, 1, 1). One block per [K] row.

#include <cuda_bf16.h>

__device__ __forceinline__ float sim_dec_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

__device__ __forceinline__ unsigned char sim_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}

extern "C" __global__ void fp8_rowscale_roundtrip(
    __nv_bfloat16* __restrict__ w,   // [N, K] BF16, rewritten in place
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.x;
    if (row >= N) return;
    __nv_bfloat16* r = w + (unsigned long long)row * K;
    float m = 0.0f;
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(r[k])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, o));
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) mx = fmaxf(mx, sm[i]);
        sm[0] = mx;
    }
    __syncthreads();
    const float scale = sm[0] > 0.0f ? sm[0] / 448.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = fminf(fmaxf(__bfloat162float(r[k]) * inv, -448.0f), 448.0f);
        r[k] = __float2bfloat16(sim_dec_fp8(sim_enc_fp8(v)) * scale);
    }
}

// ATLAS_RIPPLE_ATTN_FP8_SIM=g128: the same round trip with one scale per row per
// 128-wide K group (amax / 448 over the group), the 1x128 granularity block-FP8
// checkpoints use along K. Outlier channels then only coarsen their own group.
// Grid: (N, ceil(K/128), 1)  Block: (128, 1, 1). One thread per element.
extern "C" __global__ void fp8_g128_roundtrip(
    __nv_bfloat16* __restrict__ w,   // [N, K] BF16, rewritten in place
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.x, k = blockIdx.y * 128 + threadIdx.x;
    if (row >= N) return;
    __nv_bfloat16* r = w + (unsigned long long)row * K;
    float m = (k < K) ? fabsf(__bfloat162float(r[k])) : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, o));
    __shared__ float sm[4];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
    __syncthreads();
    const float mx = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
    const float scale = mx > 0.0f ? mx / 448.0f : 1.0f;
    const float inv = 1.0f / scale;
    if (k < K) {
        const float v = fminf(fmaxf(__bfloat162float(r[k]) * inv, -448.0f), 448.0f);
        r[k] = __float2bfloat16(sim_dec_fp8(sim_enc_fp8(v)) * scale);
    }
}
