// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// The publisher rounds both the incoming residual sum and the normalized
// component to BF16 before applying the learned RMS weight.
template<bool Add, bool Save>
__device__ void normalize_ling_row(const __nv_bfloat16* input, __nv_bfloat16* updated,
    const __nv_bfloat16* branch, const __nv_bfloat16* weight, __nv_bfloat16* output,
    __nv_bfloat16* residual, unsigned width, float eps, float* warp_sums) {
    const unsigned row = blockIdx.x * width;
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        __nv_bfloat16 value = input[row + i];
        if (Add) {
            value = __float2bfloat16(__bfloat162float(value) + __bfloat162float(branch[row + i]));
            updated[row + i] = value;
        }
        if (Save) residual[row + i] = value;
        const float x = __bfloat162float(value);
        sum = fmaf(x, x, sum);
    }
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[threadIdx.x] : 0.0f;
        for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (threadIdx.x == 0) warp_sums[0] = rsqrtf(sum / width + eps);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = __bfloat162float(__float2bfloat16(__bfloat162float(input[row + i]) * warp_sums[0]));
        output[row + i] = __float2bfloat16(normalized * __bfloat162float(weight[i]));
    }
}

extern "C" __global__ void ling_rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* y, unsigned n, float eps) {
    __shared__ float sums[32];
    normalize_ling_row<false,false>(x, nullptr, nullptr, w, y, nullptr, n, eps, sums);
}
extern "C" __global__ void ling_rms_norm_residual(const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* y, __nv_bfloat16* residual, unsigned n, float eps) {
    __shared__ float sums[32];
    normalize_ling_row<false,true>(x, nullptr, nullptr, w, y, residual, n, eps, sums);
}
extern "C" __global__ void ling_residual_add_rms_norm(__nv_bfloat16* x, const __nv_bfloat16* src,
    const __nv_bfloat16* w, __nv_bfloat16* y, __nv_bfloat16* residual, unsigned n, float eps) {
    __shared__ float sums[32];
    normalize_ling_row<true,true>(x, x, src, w, y, residual, n, eps, sums);
}
