// SPDX-License-Identifier: AGPL-3.0-only

#ifndef ATLAS_GDN_REDUCE_WYN_CUH
#define ATLAS_GDN_REDUCE_WYN_CUH

static __device__ __forceinline__ float atlas_wyn_warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

static __device__ __forceinline__ float atlas_wyn_block_sum(
    float value,
    float* warp_sums,
    unsigned int thread
) {
    value = atlas_wyn_warp_sum(value);
    const unsigned int warp = thread / 32;
    const unsigned int lane = thread & 31;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    if (thread < 4) {
        float sum = warp_sums[thread];
        sum += __shfl_down_sync(0x0fu, sum, 2);
        sum += __shfl_down_sync(0x0fu, sum, 1);
        if (thread == 0) warp_sums[0] = sum;
    }
    __syncthreads();
    return warp_sums[0];
}

#endif
