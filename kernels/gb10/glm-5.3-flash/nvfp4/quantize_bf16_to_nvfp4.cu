// SPDX-License-Identifier: AGPL-3.0-only
// Shared by GLM and Qwen: retain the activation quantizer and add a weight
// entry point with CUDA's nearest-even E4M3 and E2M1 conversion contract.
#include "../../common/quantize_bf16_to_nvfp4.cu"
#include <cuda_fp4.h>

extern "C" __global__ void quantize_bf16_to_nvfp4_weight_rne(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    atlas_pdl_enter();
    const unsigned row = blockIdx.x;
    if (row >= N) return;
    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_packed = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* row_scale = scale_out + (unsigned long long)row * (K / 16);
    const float inv_scale2 = scale2 > 0.0f ? 1.0f / scale2 : 0.0f;
    for (unsigned g = threadIdx.x; g < K / 16; g += blockDim.x) {
        float group_max = 0.0f;
        #pragma unroll
        for (unsigned j = 0; j < 16; ++j)
            group_max = fmaxf(group_max, fabsf(__bfloat162float(row_in[g * 16 + j])));
        const float raw_scale = group_max > 0.0f ? group_max * inv_scale2 / 6.0f : 0.0f;
        const __nv_fp8_e4m3 rounded_scale(raw_scale);
        row_scale[g] = *reinterpret_cast<const unsigned char*>(&rounded_scale);
        const float effective_scale = (float)rounded_scale * scale2;
        const float inv_eff = effective_scale > 0.0f ? 1.0f / effective_scale : 0.0f;
        #pragma unroll
        for (unsigned j = 0; j < 16; j += 2) {
            const float2 pair = make_float2(
                __bfloat162float(row_in[g * 16 + j]) * inv_eff,
                __bfloat162float(row_in[g * 16 + j + 1]) * inv_eff);
            row_packed[g * 8 + j / 2] =
                __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
        }
    }
}
