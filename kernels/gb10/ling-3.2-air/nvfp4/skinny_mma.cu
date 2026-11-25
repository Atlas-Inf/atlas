// SPDX-License-Identifier: AGPL-3.0-only
#include "../../qwen3.6-27b/nvfp4/skinny_mma.cu"

// The expert plan maps MMA columns back to independent token/slot pairs.
// Gate and up read the original token row; down reads the routed slot row.
extern "C" __global__ void ling_expert_mma(
    const __nv_bfloat16* input, const unsigned long long* wp,
    const unsigned long long* sp, const float* global, __nv_bfloat16* output,
    const unsigned* plan, unsigned n, unsigned k, unsigned topk,
    unsigned stride) {
    // moe_union_match stores one list of up to 32 token/slot pairs per expert.
    constexpr unsigned experts = 512, max_matches = 32;
    if (blockIdx.y >= plan[0]) return;
    const unsigned expert = plan[1 + blockIdx.y];
    const unsigned count = plan[1 + experts + expert];
    const unsigned* tokens = plan + 1 + 2 * experts + expert * max_matches;
    const unsigned* slots = tokens + experts * max_matches;
    const unsigned warps = blockDim.x / 32;
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned g = lane / 4, c = lane % 4;
    const float factor = global[expert] * 256.0f;
    const SkPart p{reinterpret_cast<const unsigned char*>(wp[expert]),
                   reinterpret_cast<const unsigned char*>(sp[expert]), nullptr,
                   factor, output, n, SkCol{0,0}};
    __shared__ float partial[SK_WARPS][4][32];
    for (unsigned base = 0; base < count; base += 8) {
        const unsigned match = base + g;
        const unsigned row = match < count ? tokens[match] : 0;
        const unsigned slot = match < count ? slots[match] : 0;
        const unsigned source = stride ? (row * topk + slot) : row;
        unsigned part, first;
        const SkTile wt = sk_tile_at<false>(p, p, n / 16, k,
            blockIdx.x, g, part, first);
        float sum[1][4] = {};
        for (unsigned chunk = warp; chunk < k / 256; chunk += warps) {
            SkW packed;
            sk_load_w<false>(packed, wt, chunk, c);
            uint4 x[1][8];
            sk_load_a<false,1>(x, input + (size_t)source * k, k,
                               match < count ? 1 : 0, 0, chunk, c);
            sk_mma_w4<1>(sum, packed, x);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) partial[warp][i][lane] = sum[0][i];
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned m = base + 2 * c + (i & 1);
                if (m < count) {
                    float v = 0.0f;
                    #pragma unroll
                    for (unsigned w = 0; w < warps; ++w)
                        v = __fadd_rn(v, partial[w][i][lane]);
                    const unsigned dest = tokens[m] * topk + slots[m];
                    output[(size_t)dest * n + first + (i >= 2 ? 8 : 0)] =
                        __float2bfloat16_rn(__fmul_rn(v, factor));
                }
            }
        }
        __syncthreads();
    }
}
