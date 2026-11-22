// SPDX-License-Identifier: AGPL-3.0-only
// Router logits for up to MAX_M decode or verify rows from one pass over the router
// weight, unrounded FP32: the common batched GEMV's body with an FP32 output, so each
// row is the chain dense_gemv_bf16_fp32out runs for one row. Kept in this target, not
// in common/, where a model's shadow of dense_gemv_bf16_batchm.cu would have to declare it.

#include "../../common/dense_gemv_bf16_batchm.cu"

extern "C" __global__ void dense_gemv_bf16_batchm_f32out(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride
) {
    dense_gemv_bf16_batchm_body(A, B, C, M, N, K, out_stride);
}
