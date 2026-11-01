// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the common `reshape_and_cache.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../common/reshape_and_cache.cu"

#include "glm_fp8g128.cuh"

// ATLAS_GLM_LATENT_QDQ=1: round each cached BF16 latent through the FP8-G128
// format in place, to measure an FP8 latent cache's quality on every reader
// without porting them. `cache` rows are [slot, 512] BF16.
extern "C" __global__ void glm_latent_qdq_fp8g128(
    __nv_bfloat16* __restrict__ cache,
    const long long* __restrict__ slots) {
    __shared__ float amax[128];
    const long long slot = slots[blockIdx.x];
    if (slot < 0) return;
    __nv_bfloat16* row = cache + (size_t)slot * 512u;
    const int tid = threadIdx.x;
    for (int g = 0; g < 4; ++g) {
        float sc;
        const __nv_fp8_storage_t q =
            glm_fp8g128_quantize(__bfloat162float(row[g * 128 + tid]), amax, &sc);
        const __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
        row[g * 128 + tid] = __float2bfloat16(__half2float(__half(h)) * sc);
    }
}

// Write GLM latents into a `fp8_g128` paged cache (see glm_fp8g128.cuh).
// `key` rows are BF16 with `key_stride` elements between tokens.
extern "C" __global__ void glm_latent_cache_write_fp8g128(
    const __nv_bfloat16* __restrict__ key,
    unsigned char* __restrict__ cache,
    const long long* __restrict__ slots,
    unsigned int block_size,
    unsigned int key_stride) {
    __shared__ float amax[128];
    const long long slot = slots[blockIdx.x];
    if (slot < 0) return;
    const unsigned int physical = (unsigned int)(slot / block_size);
    const unsigned int offset = (unsigned int)(slot % block_size);
    unsigned char* values = const_cast<unsigned char*>(glm_fp8g128_values(cache, physical, offset, block_size));
    float* scales = const_cast<float*>(glm_fp8g128_scales(cache, physical, offset, block_size));
    const __nv_bfloat16* src = key + (size_t)blockIdx.x * key_stride;
    const int tid = threadIdx.x;
    for (int g = 0; g < 4; ++g) {
        float sc;
        values[g * 128 + tid] =
            glm_fp8g128_quantize(__bfloat162float(src[g * 128 + tid]), amax, &sc);
        if (tid == 0) scales[g] = sc;
    }
}

// Dequantize one owner's logical tokens [0, tokens) into contiguous BF16
// rows `out[t][512]` for the BF16 prefill kernels. One CTA of 64 threads
// per token, eight values per thread.
extern "C" __global__ void glm_latent_dequant_fp8g128(
    const unsigned char* __restrict__ cache,
    const unsigned int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ out,
    unsigned int tokens,
    unsigned int block_size) {
    const unsigned int t = blockIdx.x;
    if (t >= tokens) return;
    const unsigned int physical = block_table[t / block_size], offset = t % block_size;
    const unsigned int col = threadIdx.x * 8u;
    const uint2 codes = *reinterpret_cast<const uint2*>(
        glm_fp8g128_values(cache, physical, offset, block_size) + col);
    const float scale = glm_fp8g128_scales(cache, physical, offset, block_size)[col / 128u];
    *reinterpret_cast<uint4*>(out + (size_t)t * 512u + col) = glm_fp8g128_dequant8(codes, scale);
}
