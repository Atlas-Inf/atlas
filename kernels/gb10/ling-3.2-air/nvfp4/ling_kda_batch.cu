// SPDX-License-Identifier: AGPL-3.0-only
#include "../../glm-5.3-flash/nvfp4/kda_recurrent_body.cuh"

// Each request has a separately allocated recurrent state. Pass its address
// directly, just as the single-row graph does. Atlas keys the batch graph by
// the ordered sequence slots that own these addresses.
extern "C" __global__ void ling_kda_batch(
    const __nv_bfloat16* qkv, const __nv_bfloat16* gate, const __nv_bfloat16* beta,
    const float* a_log, const float* bias, __nv_bfloat16* output,
    float* s0, float* s1, float* s2, float* s3, float* s4, float* s5, float* s6, float* s7,
    unsigned heads, unsigned dim, float lower_bound, unsigned row_stride, unsigned row_base) {
    // Grid row y advances state y by one token: row `y * row_stride + row_base`. Decode
    // passes (1, 0); a verify with two rows per sequence passes (2, 0) and then (2, 1).
    float* const states[] = {s0, s1, s2, s3, s4, s5, s6, s7};
    const unsigned row = blockIdx.y * row_stride + row_base;
    const unsigned offset = row * heads * dim;
    atlas_kda_recurrent_body(qkv + 3 * offset, gate + offset, beta + row * heads, a_log, bias,
                             states[blockIdx.y], output + offset, 1, heads, dim, lower_bound);
}
