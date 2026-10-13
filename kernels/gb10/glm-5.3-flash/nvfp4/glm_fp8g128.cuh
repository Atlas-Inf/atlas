// SPDX-License-Identifier: AGPL-3.0-only
// GLM NoPE-512 latent in the `fp8_g128` paged KV layout: a block of
// `block_size` tokens stores [block_size][512] FP8-E4M3 values, then
// [block_size][4] FP32 scales, one per 128 dims (amax/448, amax floored at
// 1e-4) — the packing the native sparse prefill's `prep_kv` produces.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define GLM_FP8G128_TOKEN_BYTES 528u

// Token `token`'s values and scales, given its physical block.
__device__ __forceinline__ const unsigned char* glm_fp8g128_values(
    const void* cache, unsigned int physical, unsigned int offset, unsigned int block_size) {
    return static_cast<const unsigned char*>(cache)
        + (unsigned long long)physical * block_size * GLM_FP8G128_TOKEN_BYTES + offset * 512u;
}
__device__ __forceinline__ const float* glm_fp8g128_scales(
    const void* cache, unsigned int physical, unsigned int offset, unsigned int block_size) {
    return reinterpret_cast<const float*>(static_cast<const unsigned char*>(cache)
        + (unsigned long long)physical * block_size * GLM_FP8G128_TOKEN_BYTES
        + block_size * 512u + offset * 16u);
}

// Eight consecutive E4M3 codes (same group) -> BF16 as bf16(half(q) * scale).
__device__ __forceinline__ uint4 glm_fp8g128_dequant8(uint2 codes, float scale) {
    const __nv_fp8x2_storage_t* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&codes);
    __align__(16) __nv_bfloat16 out[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(pairs[j], __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        out[2 * j] = __float2bfloat16(f.x * scale);
        out[2 * j + 1] = __float2bfloat16(f.y * scale);
    }
    return *reinterpret_cast<const uint4*>(out);
}

// One CTA of 128 threads per token; thread `tid` owns element `g * 128 + tid`
// of group `g`. Returns the E4M3 code; `scale` receives the group's scale.
__device__ __forceinline__ __nv_fp8_storage_t glm_fp8g128_quantize(
    float v, float* amax, float* scale) {
    const int tid = threadIdx.x;
    amax[tid] = fabsf(v);
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) amax[tid] = fmaxf(amax[tid], amax[tid + s]);
        __syncthreads();
    }
    float m = amax[0];
    __syncthreads();  // amax is reused by the next group
    if (!(m > 1.0e-4f)) m = 1.0e-4f;
    *scale = __fmul_rn(m, static_cast<float>(1.0 / 448.0));
    return __nv_cvt_float_to_fp8(v / *scale, __NV_SATFINITE, __NV_E4M3);
}
