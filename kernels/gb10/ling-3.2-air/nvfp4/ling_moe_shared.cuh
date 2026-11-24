// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cuda_bf16.h>
#include "../../common/atlas_pdl.cuh"
#include "ling_moe_quant.cuh"

// Ling-3.2-air shared expert: BF16, no compression
// (mlp.shared_experts.{gate,up,down}_proj.weight, moe_shared_expert_intermedi-
// ate_size 768), summed alongside the routed output
// (modeling_bailing_moe_v3_vl.py BailingMoeV3SparseMoeBlock.forward :555).
// The shared expert's swiglu limit rides share_expert_swiglu_limit_list
// (passed per launch as `limit`, 0 = none).
//
// Decode batches are M rows (1 single-stream decode, more for MTP draft and
// verify on layer 42), far under the width that would make tensor cores pay:
// a warp-per-channel dot kernel streams each weight byte exactly once (per
// CTA: 8 warps = 8 channels, weight rows of 4 bf16 per lane iteration —
// 256-byte contiguous per warp instruction) and holds the M row accumulators
// of its channel in registers, so one weight pass serves every row. The
// register budget is why MR (the row bound) is a template: the host launches
// the smallest bucket of {1, 4, 8, 16, 32} that covers M (split larger, or
// call repeatedly with row offsets, if ever needed).
//
// Numerics: dot products accumulate fp32 (torch BF16 linear's opmath)
// with a warp reduction; the gate/up pair applies the exact Bailing chain —
// silu round-tripped through BF16, gate clamped to max=limit, up to
// [-limit, limit], product round-tripped through BF16 (ling_moe_quant.cuh) —
// and `ling_moe_shared_down` rounds the down projection to BF16 once.

// Gate/up fused: H[r, c] = clamp-swiglu(dot(gate_w[c], x[r]), dot(up_w[c],
// x[r]), limit). Grid (I/8), block 256: warp w owns channels
// blockIdx.x * 8 + w for all M rows.
template <int MR>
__device__ __forceinline__ void lm_shared_gate_up_impl(
    const __nv_bfloat16* __restrict__ X,      // [M, K]
    const __nv_bfloat16* __restrict__ gate_w, // [I, K]
    const __nv_bfloat16* __restrict__ up_w,   // [I, K]
    __nv_bfloat16* __restrict__ H,            // [M, I]
    unsigned int M, unsigned int I, unsigned int K, float limit
) {
    atlas_pdl_enter();
    if (M == 0 || M > (unsigned int)MR || K % 128u != 0u) return;
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int warp = threadIdx.x / 32u;
    const unsigned int c = blockIdx.x * 8u + warp;
    if (c >= I) return;
    const __nv_bfloat16* wg = gate_w + (unsigned long long)c * K;
    const __nv_bfloat16* wu = up_w + (unsigned long long)c * K;

    float acc_g[MR], acc_u[MR];
    #pragma unroll
    for (int r = 0; r < MR; ++r) { acc_g[r] = 0.0f; acc_u[r] = 0.0f; }

    for (unsigned int k0 = 0; k0 < K; k0 += 128u) {
        const unsigned int k = k0 + lane * 4u;
        const uint2 wgp = *(const uint2*)&wg[k];
        const uint2 wup = *(const uint2*)&wu[k];
        const float wgf[4] = {
            __bfloat162float(((const __nv_bfloat16*)&wgp)[0]),
            __bfloat162float(((const __nv_bfloat16*)&wgp)[1]),
            __bfloat162float(((const __nv_bfloat16*)&wgp)[2]),
            __bfloat162float(((const __nv_bfloat16*)&wgp)[3]),
        };
        const float wuf[4] = {
            __bfloat162float(((const __nv_bfloat16*)&wup)[0]),
            __bfloat162float(((const __nv_bfloat16*)&wup)[1]),
            __bfloat162float(((const __nv_bfloat16*)&wup)[2]),
            __bfloat162float(((const __nv_bfloat16*)&wup)[3]),
        };
        #pragma unroll
        for (unsigned int r = 0; r < MR; ++r) {
            if (r >= M) break;
            const uint2 xp = *(const uint2*)&X[(unsigned long long)r * K + k];
            const float xf[4] = {
                __bfloat162float(((const __nv_bfloat16*)&xp)[0]),
                __bfloat162float(((const __nv_bfloat16*)&xp)[1]),
                __bfloat162float(((const __nv_bfloat16*)&xp)[2]),
                __bfloat162float(((const __nv_bfloat16*)&xp)[3]),
            };
            float sg = 0.0f, su = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) { sg += xf[j] * wgf[j]; su += xf[j] * wuf[j]; }
            acc_g[r] += sg;
            acc_u[r] += su;
        }
    }

    #pragma unroll
    for (unsigned int r = 0; r < MR; ++r) {
        if (r >= M) break;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_g[r] += __shfl_xor_sync(0xFFFFFFFFu, acc_g[r], off);
            acc_u[r] += __shfl_xor_sync(0xFFFFFFFFu, acc_u[r], off);
        }
    }
    if (lane != 0) return;
    #pragma unroll
    for (unsigned int r = 0; r < MR; ++r) {
        if (r >= M) break;
        const float g = __bfloat162float(__float2bfloat16(acc_g[r]));
        const float u = __bfloat162float(__float2bfloat16(acc_u[r]));
        const float s = ling_silu_bf16(g, limit);
        H[(unsigned long long)r * I + c] = __float2bfloat16(ling_gate_up_mul(s, u, limit));
    }
}

// Down: Y[r, c] = dot(w[c], H[r]). Grid (N/8), block 256, same ownership.
template <int MR>
__device__ __forceinline__ void lm_shared_down_impl(
    const __nv_bfloat16* __restrict__ H,     // [M, K] (= I of the gate/up)
    const __nv_bfloat16* __restrict__ w,     // [N, K]
    __nv_bfloat16* __restrict__ Y,           // [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    atlas_pdl_enter();
    if (M == 0 || M > (unsigned int)MR || K % 128u != 0u) return;
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int warp = threadIdx.x / 32u;
    const unsigned int c = blockIdx.x * 8u + warp;
    if (c >= N) return;
    const __nv_bfloat16* wc = w + (unsigned long long)c * K;

    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; ++r) acc[r] = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += 128u) {
        const unsigned int k = k0 + lane * 4u;
        const uint2 wp = *(const uint2*)&wc[k];
        const float wf[4] = {
            __bfloat162float(((const __nv_bfloat16*)&wp)[0]),
            __bfloat162float(((const __nv_bfloat16*)&wp)[1]),
            __bfloat162float(((const __nv_bfloat16*)&wp)[2]),
            __bfloat162float(((const __nv_bfloat16*)&wp)[3]),
        };
        #pragma unroll
        for (unsigned int r = 0; r < MR; ++r) {
            if (r >= M) break;
            const uint2 hp = *(const uint2*)&H[(unsigned long long)r * K + k];
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                s += __bfloat162float(((const __nv_bfloat16*)&hp)[j]) * wf[j];
            acc[r] += s;
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < MR; ++r) {
        if (r >= M) break;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
    }
    if (lane != 0) return;
    #pragma unroll
    for (unsigned int r = 0; r < MR; ++r) {
        if (r >= M) break;
        Y[(unsigned long long)r * N + c] = __float2bfloat16(acc[r]);
    }
}
