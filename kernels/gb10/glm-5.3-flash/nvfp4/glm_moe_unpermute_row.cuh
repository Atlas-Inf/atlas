// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// One token's sparse EP reduce over its local routes, written to `dst` (the
// token's output row, wherever it lives): the body of the `_vec8` unpermute
// kernels (moe_permute.cu) and of their RS producers
// (moe_unpermute_fused.cu), so every caller sums the same products in the
// same slot order. A block of 256 threads, each 8 consecutive columns at a
// time; hidden_size % 8 == 0 (% 64 for E4M3), topk <= 8. `WRITE_THROUGH`
// stores with st.global.wt, for a row the NIC reads from pinned memory.

namespace glm_unpermute {

constexpr unsigned int MAX_TOPK = 8;

struct Routes {
    unsigned int rows[MAX_TOPK];
    float weights[MAX_TOPK];
    bool live[MAX_TOPK];
};

// The token's routes in slot order, each live if local. Every index is a
// compile-time slot, so the arrays stay in registers.
__device__ __forceinline__ Routes load_routes(
    unsigned int token,
    const int* __restrict__ token_to_perm,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int topk,
    unsigned int local_expert_start,
    unsigned int local_expert_end
) {
    Routes r;
    #pragma unroll
    for (unsigned int k = 0; k < MAX_TOPK; k++) {
        const unsigned int slot = token * topk + k;
        const int expert = k < topk ? topk_ids[slot] : -1;
        r.live[k] = k < topk && expert >= (int)local_expert_start && expert < (int)local_expert_end;
        r.rows[k] = r.live[k] ? (unsigned int)token_to_perm[slot] : 0u;
        r.weights[k] = r.live[k] ? topk_weights[slot] : 0.0f;
    }
    return r;
}

template <bool WRITE_THROUGH>
__device__ __forceinline__ void store8(__nv_bfloat16* dst, const float (&acc)[8]) {
    uint4 out;
    __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
    #pragma unroll
    for (int j = 0; j < 8; j++) o[j] = __float2bfloat16(acc[j]);
    if (WRITE_THROUGH)
        __stwt(reinterpret_cast<uint4*>(dst), out);
    else
        *reinterpret_cast<uint4*>(dst) = out;
}

// BF16 expert rows.
template <bool WRITE_THROUGH>
__device__ __forceinline__ void row_vec8(
    const Routes& r,
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ dst,
    unsigned int hidden_size
) {
    for (unsigned int c = threadIdx.x * 8; c < hidden_size; c += blockDim.x * 8) {
        uint4 raw[MAX_TOPK];
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++)
            if (r.live[i]) raw[i] = *(const uint4*)&expert_output[r.rows[i] * hidden_size + c];
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++) {
            if (!r.live[i]) continue;
            const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw[i]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float value = __bfloat162float(v[j]);
                acc[j] += r.weights[i] * value;
            }
        }
        store8<WRITE_THROUGH>(&dst[c], acc);
    }
}

// E4M3 expert rows (ATLAS_GLM_MOE_DOWN_E4M3): bytes [rows, hidden] and FP32
// scales [rows, hidden/64]; each route's value is scale * E4M3.
template <bool WRITE_THROUGH>
__device__ __forceinline__ void row_vec8_e4m3(
    const Routes& r,
    const unsigned char* __restrict__ expert_output,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ dst,
    unsigned int hidden_size
) {
    for (unsigned int c = threadIdx.x * 8; c < hidden_size; c += blockDim.x * 8) {
        uint2 raw[MAX_TOPK];
        float sc[MAX_TOPK];
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++)
            if (r.live[i]) {
                raw[i] = *(const uint2*)&expert_output[(unsigned long long)r.rows[i] * hidden_size + c];
                sc[i] = r.weights[i] * scales[(unsigned long long)r.rows[i] * (hidden_size / 64) + c / 64];
            }
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++) {
            if (!r.live[i]) continue;
            const unsigned int w[2] = {raw[i].x, raw[i].y};
            #pragma unroll
            for (int h = 0; h < 4; h++) {
                unsigned int f2;
                const unsigned short pair = (unsigned short)(w[h >> 1] >> ((h & 1) * 16));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f2) : "h"(pair));
                const __half2 hv = *reinterpret_cast<const __half2*>(&f2);
                acc[h * 2] += sc[i] * __low2float(hv);
                acc[h * 2 + 1] += sc[i] * __high2float(hv);
            }
        }
        store8<WRITE_THROUGH>(&dst[c], acc);
    }
}

}  // namespace glm_unpermute
