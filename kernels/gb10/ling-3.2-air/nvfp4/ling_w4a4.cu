// SPDX-License-Identifier: AGPL-3.0-only

#include "../../inkling-2-mini/nvfp4/inkling2_moe_gemm.cu"
#include "../../inkling-2-mini/nvfp4/inkling2_moe_prefill.cu"

// Ling keeps one device pointer per expert instead of a contiguous weight tensor.
extern "C" __global__ void __launch_bounds__(256, 2) ling_w4a4(
    const unsigned char* x, const unsigned char* xs, const float* outer,
    const unsigned char* codes, const unsigned char* scales, const float* global,
    __nv_bfloat16* out, const int* offsets, const int* tiles, const int* rows,
    unsigned int n, unsigned int k) {
    ink_moe_lowp<false, true>(x, xs, outer, codes, scales, global, out, offsets, tiles, rows, n, k);
}
