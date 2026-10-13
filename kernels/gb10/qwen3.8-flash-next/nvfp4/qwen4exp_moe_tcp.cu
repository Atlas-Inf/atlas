// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next routed-MoE PREFILL on tensor cores with exactly the TC
// decode numerics (ATLAS_QWEN4EXP_PREFILL_MOE_BF16=1): a row's gate/up/SiLU
// and down bytes are the ones qwen4exp_moe_c8_tc.cu writes for it in decode
// (verify rows and serial decode under ATLAS_QWEN4EXP_MOE_TC), so prefill and
// decode agree on the routed MoE. Replaces moe_prefill_q38's routed chain
// (a->e4m3, gate_up_silu, down: E4M3 weights and activations, ~7% off the
// exact math on real inputs; this is ~0.3%, scripts/dev/qwen4exp_moe_fidelity.cu).
//
// The per-output operation sequence is the decode kernels' (qwen4exp_moe_tc.cuh):
//   gate/up  four K quarters (5 k128 super-blocks each), each its own MMA
//            chain from zero -- per super-block the k32 pairs j = 0..3, MMA 0
//            then MMA 1 -- summed ((q0 + q1) + q2) + q3, * scale2, BF16; then
//            SiLU * up (tc_silu_up), stored BF16 (decode rounds the same FP32
//            to BF16 when down loads it)
//   down     one chain over all of K, * scale2, BF16
// Only the work layout differs: a warp runs TCP_*_MT m16 tiles x 4 n8 tiles,
// decoding each weight once per 16 MT rows; MMA rows are independent
// (scripts/dev/qwen4exp_moe_tcp_bench.cu compares every sampled row with the
// decode kernel, byte for byte).
//
// Layout as moe_prefill_q38's routed chain: rows sorted by expert
// (expert_offsets [E + 1], sorted_token_ids [S] -> input token), act [S, 640]
// BF16, C [S, 2560] BF16 (expert_down_out); the served [N, K/2] weight tables
// (gate_ptrs / up_ptrs / down_ptrs); a null table is a remote expert, skipped.
// Grids: gate/up (640 / 32, m-blocks of 2 * 16 * TCP_GU_MT rows, E), down
// (2560 / 64, m-blocks of 128 rows, E), block 128; CTAs stride the blocks.

#include "qwen4exp_moe_c8.cuh"
#include "qwen4exp_moe_tc.cuh"

#define TCP_DN_NT 4  // n8 tiles a down warp
#ifndef TCP_GU_NT
#define TCP_GU_NT 4  // n8 tiles a gate/up warp
#endif
#ifndef TCP_GU_MT
#define TCP_GU_MT 2  // m16 tiles a gate/up warp (4: register spills, measured slower)
#endif
#ifndef TCP_DN_MT
#define TCP_DN_MT 2  // m16 tiles a down warp (4: +12% down)
#endif

// acc[mt][nt] += one super-block sb of the warp's n8 tiles (B rows `b[nt]`
// at row g, scales `s[nt]`) against the rows at `arow[mt][0 / 1]` (rows g and
// g + 8 of m-tile mt; null: zero).
template <unsigned K, unsigned TCP_MT, unsigned TCP_NT>
__device__ __forceinline__ void tcp_sb(float (&acc)[TCP_MT][TCP_NT][4], const unsigned char* const (&b)[TCP_NT],
                                       const unsigned char* const (&s)[TCP_NT],
                                       const __nv_bfloat16* const (&arow)[TCP_MT][2], unsigned sb, unsigned t) {
    uint4 w[TCP_NT];
    unsigned short e[TCP_NT];
#pragma unroll
    for (unsigned nt = 0; nt < TCP_NT; nt++) {
        w[nt] = *(const uint4*)(b[nt] + 64 * sb);
        e[nt] = *(const unsigned short*)(s[nt] + 8 * sb);
    }
#pragma unroll
    for (unsigned j = 0; j < 4; j++) {
        unsigned bq[TCP_NT][4];
#pragma unroll
        for (unsigned nt = 0; nt < TCP_NT; nt++) {
            const unsigned q[4] = {w[nt].x, w[nt].y, w[nt].z, w[nt].w};
            tc_b8(q[j], (unsigned char)(e[nt] >> (8 * (j >> 1))), bq[nt]);
        }
        const unsigned k = sb * 128 + 32 * t + 8 * j;
#pragma unroll
        for (unsigned mt = 0; mt < TCP_MT; mt++) {
            const uint4 z = make_uint4(0, 0, 0, 0);
            unsigned a[4], a8[4];
            tc_a8(arow[mt][0] ? *(const uint4*)(arow[mt][0] + k) : z, a);
            tc_a8(arow[mt][1] ? *(const uint4*)(arow[mt][1] + k) : z, a8);
#pragma unroll
            for (unsigned nt = 0; nt < TCP_NT; nt++) {
                tc_mma(acc[mt][nt], a[0], a8[0], a[1], a8[1], bq[nt][0], bq[nt][1]);
                tc_mma(acc[mt][nt], a[2], a8[2], a[3], a8[3], bq[nt][2], bq[nt][3]);
            }
        }
    }
}

// ── gate/up + SiLU: warp w = projection w & 1, rows WR (w >> 1) of the
// 2 WR-row block, outputs n0 .. n0 + 31 ──
template <bool CLAMP>
__device__ __forceinline__ void tcp_gate_up(
    const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs, const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals, __nv_bfloat16* __restrict__ act,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids, unsigned num_experts
) {
    constexpr unsigned K = C8_H, ROW = K / 2, K16 = K / 16, SB = K / 128, QSB = SB / 4;
    constexpr unsigned TCP_MT = TCP_GU_MT, TCP_NT = TCP_GU_NT, NO = 8 * TCP_NT, WR = 16 * TCP_MT, BR = 2 * WR;  // rows a warp, a block
    const unsigned e = blockIdx.z;
    if (blockDim.x != 128) __trap();
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e], m_rows = expert_offsets[e + 1] - m_start;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, proj = warp & 1u, mh = warp >> 1;
    const unsigned g = lane >> 2, t = lane & 3u, n0 = blockIdx.x * NO;
    const unsigned char* B = (const unsigned char*)(proj ? up_packed_ptrs : gate_packed_ptrs)[e];
    const unsigned char* S = (const unsigned char*)(proj ? up_scale_ptrs : gate_scale_ptrs)[e];
    if (m_rows <= 0 || gate_packed_ptrs[e] == 0 || up_packed_ptrs[e] == 0) return;
    const float s2 = (proj ? up_scale2_vals : gate_scale2_vals)[e];
    const unsigned char* b[TCP_NT];
    const unsigned char* s[TCP_NT];
#pragma unroll
    for (unsigned nt = 0; nt < TCP_NT; nt++) {
        b[nt] = B + (size_t)(n0 + 8 * nt + g) * ROW + 16 * t;
        s[nt] = S + (size_t)(n0 + 8 * nt + g) * K16 + 2 * t;
    }
    __shared__ unsigned short s_gu[2][BR][NO];
#pragma unroll 1
    for (int m0 = blockIdx.y * BR; m0 < m_rows; m0 += gridDim.y * BR) {
        const __nv_bfloat16* arow[TCP_MT][2];
#pragma unroll
        for (unsigned mt = 0; mt < TCP_MT; mt++)
#pragma unroll
            for (unsigned h = 0; h < 2; h++) {
                const int r = m0 + WR * mh + 16 * mt + g + 8 * h;
                arow[mt][h] = r < m_rows ? A + (size_t)sorted_token_ids[m_start + r] * K : nullptr;
            }
        float acc[TCP_MT][TCP_NT][4] = {}, sum[TCP_MT][TCP_NT][4];
#pragma unroll 1
        for (unsigned sb = 0; sb < SB; sb++) {
            if (sb && sb % QSB == 0) {  // a K quarter ends: fold it in, start the next from zero
#pragma unroll
                for (unsigned mt = 0; mt < TCP_MT; mt++)
#pragma unroll
                    for (unsigned nt = 0; nt < TCP_NT; nt++)
#pragma unroll
                        for (unsigned i = 0; i < 4; i++) {
                            sum[mt][nt][i] = sb == QSB ? acc[mt][nt][i] : sum[mt][nt][i] + acc[mt][nt][i];
                            acc[mt][nt][i] = 0.f;
                        }
            }
            tcp_sb<K, TCP_MT, TCP_NT>(acc, b, s, arow, sb, t);
        }
#pragma unroll
        for (unsigned mt = 0; mt < TCP_MT; mt++)
#pragma unroll
            for (unsigned nt = 0; nt < TCP_NT; nt++)
#pragma unroll
                for (unsigned i = 0; i < 4; i++) {
                    const unsigned r = WR * mh + 16 * mt + g + 8 * (i >> 1), c = 8 * nt + 2 * t + (i & 1);
                    s_gu[proj][r][c] = __bfloat16_as_ushort(__float2bfloat16((sum[mt][nt][i] + acc[mt][nt][i]) * s2));
                }
        __syncthreads();
        for (unsigned i = threadIdx.x; i < BR * NO; i += blockDim.x) {
            const int r = m0 + (int)(i / NO);
            if (r < m_rows)
                act[(size_t)(m_start + r) * C8_I + n0 + i % NO] =
                    __float2bfloat16(tc_silu_up<CLAMP>(s_gu[0][i / NO][i % NO], s_gu[1][i / NO][i % NO], true));
        }
        __syncthreads();
    }
}

#define TCP_GU_ARGS                                                            \
    const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gp, \
    const unsigned long long* __restrict__ gs, const float* __restrict__ g2,  \
    const unsigned long long* __restrict__ upk, const unsigned long long* __restrict__ us, \
    const float* __restrict__ u2, __nv_bfloat16* __restrict__ act,            \
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids, \
    unsigned int num_experts
#define TCP_GU_PASS A, gp, gs, g2, upk, us, u2, act, expert_offsets, sorted_token_ids, num_experts

extern "C" __global__ void __launch_bounds__(128, 2) qwen4exp_moe_tcp_gate_up(TCP_GU_ARGS) {
    tcp_gate_up<true>(TCP_GU_PASS);
}

// Without the routed SwiGLU clamp (with ATLAS_QWEN4EXP_MOE_NO_CLAMP).
extern "C" __global__ void __launch_bounds__(128, 2) qwen4exp_moe_tcp_gate_up_nc(TCP_GU_ARGS) {
    tcp_gate_up<false>(TCP_GU_PASS);
}

// ── down: warp w = rows 64 (w >> 1) of the 128-row block, outputs
// n0 + 32 (w & 1) .. + 31; one chain over K ──
extern "C" __global__ void __launch_bounds__(128, 2) qwen4exp_moe_tcp_down(
    const __nv_bfloat16* __restrict__ act, const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, const int* __restrict__ expert_offsets, unsigned int num_experts
) {
    constexpr unsigned K = C8_I, ROW = K / 2, K16 = K / 16, SB = K / 128, TCP_MT = TCP_DN_MT, TCP_NT = TCP_DN_NT, WR = 16 * TCP_MT, BR = 2 * WR;
    const unsigned e = blockIdx.z;
    if (blockDim.x != 128) __trap();
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e], m_rows = expert_offsets[e + 1] - m_start;
    const unsigned char* B = (const unsigned char*)packed_ptrs[e];
    if (m_rows <= 0 || B == nullptr) return;
    const unsigned char* S = (const unsigned char*)scale_ptrs[e];
    const float s2 = scale2_vals[e];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, mh = warp >> 1;
    const unsigned g = lane >> 2, t = lane & 3u, n0 = blockIdx.x * 64 + 32 * (warp & 1u);
    const unsigned char* b[TCP_NT];
    const unsigned char* s[TCP_NT];
#pragma unroll
    for (unsigned nt = 0; nt < TCP_NT; nt++) {
        b[nt] = B + (size_t)(n0 + 8 * nt + g) * ROW + 16 * t;
        s[nt] = S + (size_t)(n0 + 8 * nt + g) * K16 + 2 * t;
    }
#pragma unroll 1
    for (int m0 = blockIdx.y * BR; m0 < m_rows; m0 += gridDim.y * BR) {
        const __nv_bfloat16* arow[TCP_MT][2];
#pragma unroll
        for (unsigned mt = 0; mt < TCP_MT; mt++)
#pragma unroll
            for (unsigned h = 0; h < 2; h++) {
                const int r = m0 + WR * mh + 16 * mt + g + 8 * h;
                arow[mt][h] = r < m_rows ? act + (size_t)(m_start + r) * K : nullptr;
            }
        float acc[TCP_MT][TCP_NT][4] = {};
#pragma unroll 1
        for (unsigned sb = 0; sb < SB; sb++) tcp_sb<K, TCP_MT, TCP_NT>(acc, b, s, arow, sb, t);
#pragma unroll
        for (unsigned mt = 0; mt < TCP_MT; mt++)
#pragma unroll
            for (unsigned h = 0; h < 2; h++) {
                const int r = m0 + WR * mh + 16 * mt + g + 8 * h;
                if (r >= m_rows) continue;
#pragma unroll
                for (unsigned nt = 0; nt < TCP_NT; nt++)
                    *(unsigned*)(C + (size_t)(m_start + r) * C8_H + n0 + 8 * nt + 2 * t) =
                        tc_bf16x2(acc[mt][nt][2 * h] * s2, acc[mt][nt][2 * h + 1] * s2);
            }
    }
}
