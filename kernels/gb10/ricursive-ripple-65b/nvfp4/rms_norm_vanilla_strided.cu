// SPDX-License-Identifier: AGPL-3.0-only

// Strided vanilla RMSNorm: `rms_norm_vanilla`'s math (out = x * w / RMS(x), no
// `(1 + w)` offset) over `common/rms_norm.cu::rms_norm_strided`'s addressing,
// so the per-head Q/K norms of all multi-row verify rows run in ONE launch
// for plain-weight models (Ricursive) instead of one per row and head group.
//
// Grid: (rows_per_group, num_groups, 1)   Block: (min(hidden_size, 1024), 1, 1)

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

__device__ __forceinline__ float rmsvs_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

extern "C" __global__ void rms_norm_vanilla_strided(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps,
    unsigned int row_stride                     // elements between GROUPS
) {
    atlas_pdl_enter();
    const unsigned int tid = threadIdx.x;
    const unsigned long long base =
        (unsigned long long)blockIdx.y * row_stride + (unsigned long long)blockIdx.x * hidden_size;
    const __nv_bfloat16* x = input + base;
    __nv_bfloat16* out = output + base;

    float ss = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        ss += v * v;
    }
    ss = rmsvs_warp_sum(ss);
    __shared__ float warp_sums[32];
    if (tid % 32 == 0) warp_sums[tid / 32] = ss;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        v = rmsvs_warp_sum(v);
        if (tid == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x)
        out[i] = __float2bfloat16(__bfloat162float(x[i]) * rms * __bfloat162float(weight[i]));
}
