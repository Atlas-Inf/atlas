// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Row-major original E4M3 weights and dynamic per-token E4M3 activations.
// Each warp computes one output row, accumulating the 128-wide products
// before applying the activation and weight scale. Handles the partial last
// output block in kv_a_proj_with_mqa (576 rows).
extern "C" __global__ void ling_fp8_linear(const __nv_fp8_e4m3* x, const float* xs,
    const __nv_fp8_e4m3* w, const void* ws, __nv_bfloat16* y,
    unsigned int n, unsigned int k, unsigned int bf16_scales) {
    const unsigned int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= n) return;
    const unsigned int lane = threadIdx.x % 32, token = blockIdx.y, groups = k / 128;
    float acc = 0.f;
    for (unsigned int group = 0; group < groups; ++group) {
        const unsigned int col = group * 128 + lane;
        float block = 0.f;
        #pragma unroll
        for (unsigned int i = 0; i < 128; i += 32) {
            block += float(x[(size_t)token * k + col+i]) * float(w[(size_t)row * k + col+i]);
        }
        for (int stride = 16; stride; stride /= 2)
            block += __shfl_down_sync(0xffffffff, block, stride);
        if (!lane) {
            const size_t si = (size_t)(row/128)*groups + group;
            float s = bf16_scales ? __bfloat162float(static_cast<const __nv_bfloat16*>(ws)[si])
                                  : static_cast<const float*>(ws)[si];
            acc += block * (xs[(size_t)token * groups + group] * s);
        }
    }
    if (!lane) y[(size_t)token*n + row] = __float2bfloat16(acc);
}
