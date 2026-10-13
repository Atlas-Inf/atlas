// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next routed-MoE PREFILL GEMMs, bit-identical twins of
// `moe_w4a16_fused_gate_up_t_k64` + `moe_silu_mul` and of
// `moe_w4a16_grouped_gemm_ptrtable_t_k64` (moe_w4a16_grouped_gemm.cu, the
// qwen3.6-35b-a3b file this target symlinks). Opt-in:
// ATLAS_QWEN4EXP_PREFILL_MOE=1.
//
// WHAT THE DEFAULT DOES. Per 64-row x 128-column tile and 64-wide K step, its
// 128 threads (4 warps) stage A (BF16, gathered through sorted_token_ids) and
// the packed E2M1 weights + FP8 group scales, then every thread dequantizes
// one column -- LUT[nibble] * (scale * scale2) -> E4M3 -- into a shared FP8
// tile, a barrier, and the warps run mma.sync.m16n8k32.e4m3 with A converted
// BF16 -> E4M3 on the fly, by every one of the 10 (gate/up) or 20 (down)
// column-tile CTAs that read the row. Dequant and MMA alternate across
// barriers, so the tensor cores idle while the tile converts: on GB10
// (8192-token chunk, 256 local experts) it runs at 33 TFLOP/s of the 210 an
// FP8 mma.sync reaches, and stubbing the dequant alone
// (MOE_PROBE_CHEAP_DEQUANT) takes it to 60.
//
// WHAT THIS DOES, the same arithmetic in three changes:
//  1. A is converted to E4M3 ONCE (`moe_q38_a_to_e4m3` for the gate/up input;
//     the gate/up epilogue writes the down input in E4M3 directly), so the
//     GEMMs stage half the A bytes and run no conversion in the K loop.
//  2. A 128-row tile with 8 warps: each weight tile is dequantized once for
//     twice the rows, and the FP8 weight tile is double-buffered so the next
//     step's dequant issues while this step's MMAs drain (no data dependence).
//  3. Gate and up of one 64-column intermediate slice share a CTA, so
//     SiLU(gate) * up is formed in the epilogue: the [rows, 2 x 640] BF16
//     gate/up intermediates and the `moe_silu_mul` pass are gone.
//
// BIT-IDENTICAL, element by element:
//  * the FP8 weight is `cvt.rn.satfinite.e4m3x2(LUT[q] * (float(s) * s2))`,
//    the default's expression;
//  * the FP8 activation is `cvt.rn.satfinite(float(bf16))` of the same BF16
//    value the default converts in its K loop (for the down input, the BF16
//    `moe_silu_mul` would have stored);
//  * every output is ONE FP32 accumulator over the k32 MMAs in increasing k,
//    k = 0, 32, 64, ... -- the default's chain (its K64 step issues the two
//    k32 halves in order); a K step only regroups which barrier interval an
//    MMA falls in;
//  * gate/up are rounded to BF16 (`__float2bfloat16`, as the default stores
//    them) and `moe_silu_mul`'s `g * (1 / (1 + __expf(-g))) * u` follows.
// `scripts/dev/qwen4exp_moe_prefill_bench.cu` compares every byte of the
// down output (and of the SiLU*up activation, through its E4M3 image).
//
// MEASURED (GB10, 256 local experts, top-10 of 512 with a lognormal skew):
//   16000-token chunk: default 14.45 gate_up + 1.28 silu + 8.77 down = 24.5 ms
//                      q38      0.52 a->e4m3 + 10.05 gate_up+silu + 5.90 down = 16.5 ms
//   8192-token chunk:  13.75 -> 9.7 ms.
// Where the rest goes (8192 tokens, gate_up+silu 6.04 ms): without the MMAs
// (Q38_PROBE_NO_MMA) 5.90, without the dequant (Q38_PROBE_NO_DEQUANT) 4.77,
// without both 4.49 -- the staging pipeline, not the tensor cores, is the
// floor now; the expert weights alone are 471 MB a layer (2.2 ms of DRAM).
// BK = 64 (fewer barriers, one CTA per SM) measured 10-20% slower.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define Q38_BM 128                 // rows per CTA (8 warps x 16)
#ifndef Q38_BK
#define Q38_BK 32                  // K per pipeline step: 32 or 64 (k32 MMAs)
#endif
#ifndef Q38_STAGES
#define Q38_STAGES 3               // raw (A + packed weight) ring depth (2: +3%, 4: +2%)
#endif
#ifndef Q38_MINB
#define Q38_MINB 2                 // CTAs per SM the register budget targets
#endif
#define Q38_GROUP 16               // NVFP4 scale group along K
#define Q38_A_STRIDE (Q38_BK + 16) // FP8 A rows (16-byte aligned, conflict-free u32 reads)
#define Q38_BP_STRIDE (128 + 16)   // packed-byte rows
#define Q38_F8_STRIDE (Q38_BK + 16) // FP8 weight rows

__device__ __constant__ float Q38_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void q38_cp16(void* dst_smem, const void* src_gmem, bool pred) {
    const unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    const unsigned int n = pred ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src_gmem), "r"(n));
}

// Two floats -> two E4M3 bytes (`lo` in the low byte): the default's
// conversion instruction and pairing.
__device__ __forceinline__ unsigned short q38_e4m3x2(float lo, float hi) {
    unsigned short pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(hi), "f"(lo));
    return pair;
}

__device__ __forceinline__ float q38_bf16_bits_to_f32(unsigned short bf) {
    float f;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(bf));
    return f;
}

// ── A -> E4M3, elementwise, four per thread ──
// `moe_bf16x4_to_e4m3x4` (the default's K-loop conversion) over [n] BF16.
// Grid: (ceil(n / 4 / 256)), Block 256; n % 4 == 0.
extern "C" __global__ void moe_q38_a_to_e4m3(
    const __nv_bfloat16* __restrict__ a,
    unsigned char* __restrict__ a8,
    const unsigned int n
) {
    const unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint2 w = *reinterpret_cast<const uint2*>(a + i);
    const unsigned short h0 = q38_e4m3x2(q38_bf16_bits_to_f32((unsigned short)(w.x & 0xFFFFu)),
                                         q38_bf16_bits_to_f32((unsigned short)(w.x >> 16)));
    const unsigned short h1 = q38_e4m3x2(q38_bf16_bits_to_f32((unsigned short)(w.y & 0xFFFFu)),
                                         q38_bf16_bits_to_f32((unsigned short)(w.y >> 16)));
    *reinterpret_cast<unsigned int*>(a8 + i) = ((unsigned int)h1 << 16) | (unsigned int)h0;
}

// Shared memory of one CTA (static).
struct Q38Smem {
    unsigned char a[Q38_STAGES][Q38_BM][Q38_A_STRIDE];
    unsigned char bp[Q38_STAGES][Q38_BK / 2][Q38_BP_STRIDE];
    unsigned char bs[Q38_STAGES][Q38_BK / Q38_GROUP][Q38_BP_STRIDE];
    unsigned char f8[2][128][Q38_F8_STRIDE];
    int tok[Q38_BM];
    float s2[2];
    float lut[16];
};

// One 128-row x 128-column x K tile over E4M3 A ([*, K] bytes, row `tok[r]`).
// Columns 0..63 come from (B0, S0, s2[0]) at column offset n0, 64..127 from
// (B1, S1, s2[1]) at n1 (the gate and up halves; for the down projection both
// halves are one matrix at n0 and n0 + 64). `N` is each weight's column count
// (row stride of the packed / scale planes). On return acc[nt] holds n-tile
// nt (0..7 the first half, 8..15 the second) for this warp's 16 rows.
__device__ __forceinline__ void q38_tile(
    Q38Smem& sm,
    float (&acc)[16][4],
    const unsigned char* __restrict__ A8, unsigned int K,
    unsigned int cta_m_local, unsigned int M_eff,
    const unsigned char* B0, const unsigned char* S0, unsigned int n0,
    const unsigned char* B1, const unsigned char* S1, unsigned int n1,
    unsigned int N
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31u;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    #pragma unroll
    for (int i = 0; i < 16; ++i) { acc[i][0] = 0.0f; acc[i][1] = 0.0f; acc[i][2] = 0.0f; acc[i][3] = 0.0f; }
    const unsigned int steps = K / Q38_BK;

    // Loads of step `s` into raw buffer `b`; always one commit group (empty
    // past the last step) so the wait counts below stay uniform.
    auto issue = [&](unsigned int s, unsigned int b) {
        if (s < steps) {
            const unsigned int kb = s * Q38_BK;
            // A: 128 rows x BK bytes.
            #pragma unroll
            for (unsigned int c = tid; c < Q38_BM * (Q38_BK / 16); c += 256) {
                const unsigned int row = c / (Q38_BK / 16);
                const unsigned int col = (c % (Q38_BK / 16)) * 16;
                const bool valid = (cta_m_local + row) < M_eff;
                const unsigned int a_row = (unsigned int)sm.tok[row];
                q38_cp16(&sm.a[b][row][col], &A8[(unsigned long long)a_row * K + kb + col], valid);
            }
            // Packed weights: BK/2 kp-rows x 128 columns; scales: BK/16
            // groups x 128 columns.
            #pragma unroll
            for (unsigned int c = tid; c < (Q38_BK / 2) * 8; c += 256) {
                const unsigned int kp = c >> 3;
                const unsigned int nc = (c & 7u) << 4;            // 0..112, step 16
                const unsigned char* Bh = nc < 64 ? B0 : B1;
                const unsigned int gn = (nc < 64 ? n0 : n1) + (nc & 63u);
                q38_cp16(&sm.bp[b][kp][nc], &Bh[(unsigned long long)((kb >> 1) + kp) * N + gn], true);
            }
            if (tid < (Q38_BK / Q38_GROUP) * 8) {
                const unsigned int grp = tid >> 3;
                const unsigned int nc = (tid & 7u) << 4;
                const unsigned char* Sh = nc < 64 ? S0 : S1;
                const unsigned int gn = (nc < 64 ? n0 : n1) + (nc & 63u);
                q38_cp16(&sm.bs[b][grp][nc], &Sh[(unsigned long long)(kb / Q38_GROUP + grp) * N + gn], true);
            }
        }
        asm volatile("cp.async.commit_group;");
    };
    // Dequantize raw buffer `rb` into FP8 buffer `fb`: thread -> one column and
    // BK/32 of the BK/16 scale groups (the default's per-element expression
    // and pairing).
    auto dequant = [&](unsigned int rb, unsigned int fb) {
#ifdef Q38_PROBE_NO_DEQUANT
        // MEASUREMENT PROBE ONLY -- WRONG OUTPUT. One byte read and written.
        sm.f8[fb][tid & 127u][(tid >> 7) * 16] = sm.bp[rb][0][tid & 127u];
        return;
#endif
        const unsigned int col = tid & 127u;
        const float s2 = sm.s2[col >> 6];
        #pragma unroll
        for (unsigned int gi = 0; gi < Q38_BK / 32; ++gi) {
            const unsigned int grp = (tid >> 7) * (Q38_BK / 32) + gi;
            __nv_fp8_e4m3 f;
            *(unsigned char*)&f = sm.bs[rb][grp][col];
            const float sv = (float)f * s2;
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j) {
                const unsigned int kp = grp * 8 + j;
                const unsigned char packed = sm.bp[rb][kp][col];
                *(unsigned short*)&sm.f8[fb][col][kp * 2] =
                    q38_e4m3x2(sm.lut[packed & 0xF] * sv, sm.lut[packed >> 4] * sv);
            }
        }
    };

    // Pipeline over a ring of Q38_STAGES raw buffers and two FP8 weight
    // buffers, per step s: MMA(s) from a[s % R] / f8[s & 1]; past one barrier
    // the loads of step s+R into the raw buffer MMA(s) just freed, and the
    // dequant of step s+1 into the other FP8 buffer -- issued while step s's
    // MMAs drain the tensor pipe -- then a second barrier before MMA(s+1).
    #pragma unroll
    for (unsigned int i = 0; i < Q38_STAGES; ++i) issue(i, i);
    asm volatile("cp.async.wait_group %0;" ::"n"(Q38_STAGES - 1));   // step 0
    __syncthreads();
    dequant(0, 0);
    __syncthreads();
    const unsigned int r0 = warp * 16 + g, r1 = r0 + 8;
    for (unsigned int s = 0; s < steps; ++s) {
        const unsigned int rb = s % Q38_STAGES, fb = s & 1u;
        #pragma unroll
        for (unsigned int kk = 0; kk < Q38_BK; kk += 32) {
            const unsigned int a0 = *(const unsigned int*)&sm.a[rb][r0][kk + 4 * t4];
            const unsigned int a1 = *(const unsigned int*)&sm.a[rb][r1][kk + 4 * t4];
            const unsigned int a2 = *(const unsigned int*)&sm.a[rb][r0][kk + 16 + 4 * t4];
            const unsigned int a3 = *(const unsigned int*)&sm.a[rb][r1][kk + 16 + 4 * t4];
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const unsigned int nc = nt * 8 + g;
                const unsigned int b0 = *(const unsigned int*)&sm.f8[fb][nc][kk + 4 * t4];
                const unsigned int b1 = *(const unsigned int*)&sm.f8[fb][nc][kk + 16 + 4 * t4];
#ifdef Q38_PROBE_NO_MMA
                // MEASUREMENT PROBE ONLY -- WRONG OUTPUT. Fragments still loaded.
                acc[nt][0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1);
#else
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
#endif
            }
        }
        if (s + 1 < steps) {
            asm volatile("cp.async.wait_group %0;" ::"n"(Q38_STAGES - 2));   // raw(s+1), own part
        }
        __syncthreads();       // a[rb] / f8[fb] read by every warp; raw(s+1) visible
        issue(s + Q38_STAGES, rb);
        if (s + 1 < steps) {
            dequant((s + 1) % Q38_STAGES, fb ^ 1u);
            __syncthreads();   // f8[fb^1] complete before MMA(s+1)
        }
    }
    asm volatile("cp.async.wait_group 0;");   // drain the empty tail groups
}

// Per-CTA setup for rows [cta_m_local, cta_m_local + BM) of an expert: the
// gather table (identity when `sorted_token_ids` is null).
__device__ __forceinline__ void q38_rows(Q38Smem& sm, const int* __restrict__ sorted_token_ids,
                                         unsigned int m_start, unsigned int cta_m_local, unsigned int M_eff) {
    if (threadIdx.x < Q38_BM) {
        const unsigned int r = threadIdx.x;
        const unsigned int cta_m = m_start + cta_m_local;
        sm.tok[r] = (sorted_token_ids && cta_m_local + r < M_eff) ? sorted_token_ids[cta_m + r]
                                                                  : (int)(cta_m + r);
    }
}

// ── gate + up + SiLU*mul ──
// act8[row, n] = e4m3(bf16(silu(bf16(A x gate)) * bf16(A x up))) for rows
// [m_start, m_start + M) of one weight set; A8 rows through
// `sorted_token_ids` (identity when null). CTAs stride the 128-row tiles.
__device__ __forceinline__ void q38_gate_up_silu_rows(
    Q38Smem& sm,
    const unsigned char* __restrict__ A8,
    const unsigned char* Bg, const unsigned char* Sg, float s2g,
    const unsigned char* Bu, const unsigned char* Su, float s2u,
    unsigned char* __restrict__ act8,
    const int* __restrict__ sorted_token_ids,
    unsigned int m_start, unsigned int M_eff,
    unsigned int N, unsigned int K
) {
    const unsigned int n0 = blockIdx.x * 64;
    if (threadIdx.x == 0) { sm.s2[0] = s2g; sm.s2[1] = s2u; }
    if (threadIdx.x < 16) sm.lut[threadIdx.x] = Q38_E2M1_LUT[threadIdx.x];
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    for (unsigned int cta_m_local = blockIdx.y * Q38_BM; cta_m_local < M_eff;
         cta_m_local += gridDim.y * Q38_BM) {
        __syncthreads();   // previous tile's readers of sm.tok / buffers are done
        q38_rows(sm, sorted_token_ids, m_start, cta_m_local, M_eff);
        __syncthreads();
        float acc[16][4];
        q38_tile(sm, acc, A8, K, cta_m_local, M_eff, Bg, Sg, n0, Bu, Su, n0, N);
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned int c0 = n0 + nt * 8 + t4 * 2;
            #pragma unroll
            for (unsigned int half = 0; half < 2; ++half) {
                const unsigned int lr = warp * 16 + g + half * 8;
                if (cta_m_local + lr >= M_eff) continue;
                float o[2];
                #pragma unroll
                for (unsigned int j = 0; j < 2; ++j) {
                    const float gv = __bfloat162float(__float2bfloat16(acc[nt][half * 2 + j]));
                    const float uv = __bfloat162float(__float2bfloat16(acc[nt + 8][half * 2 + j]));
                    const float sig = 1.0f / (1.0f + __expf(-gv));
                    o[j] = __bfloat162float(__float2bfloat16(gv * sig * uv));
                }
                const unsigned int row = m_start + cta_m_local + lr;
                *(unsigned short*)&act8[(size_t)row * N + c0] = q38_e4m3x2(o[0], o[1]);
            }
        }
    }
}

// ── down ──
// C[row, n] = bf16(act x down) for rows [m_start, m_start + M) of one weight
// set (act8 not gathered). CTAs stride the 128-row tiles.
__device__ __forceinline__ void q38_down_rows(
    Q38Smem& sm,
    const unsigned char* __restrict__ act8,
    const unsigned char* B, const unsigned char* S, float s2,
    __nv_bfloat16* __restrict__ C,
    unsigned int m_start, unsigned int M_eff,
    unsigned int N, unsigned int K
) {
    const unsigned int n0 = blockIdx.x * 128;
    if (threadIdx.x == 0) { sm.s2[0] = s2; sm.s2[1] = s2; }
    if (threadIdx.x < 16) sm.lut[threadIdx.x] = Q38_E2M1_LUT[threadIdx.x];
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    for (unsigned int cta_m_local = blockIdx.y * Q38_BM; cta_m_local < M_eff;
         cta_m_local += gridDim.y * Q38_BM) {
        __syncthreads();
        q38_rows(sm, nullptr, m_start, cta_m_local, M_eff);
        __syncthreads();
        float acc[16][4];
        q38_tile(sm, acc, act8, K, cta_m_local, M_eff, B, S, n0, B, S, n0 + 64, N);
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const unsigned int c0 = n0 + nt * 8 + t4 * 2;
            #pragma unroll
            for (unsigned int half = 0; half < 2; ++half) {
                const unsigned int lr = warp * 16 + g + half * 8;
                if (cta_m_local + lr >= M_eff) continue;
                const unsigned int row = m_start + cta_m_local + lr;
                *(unsigned int*)&C[(size_t)row * N + c0] =
                      (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc[nt][half * 2]))
                    | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc[nt][half * 2 + 1])) << 16);
            }
        }
    }
}

// Grouped (routed experts): weights by expert pointer tables, rows by
// expert_offsets; a null table is a remote (EP) expert and is skipped, as in
// the default kernels. Grid: (N / 64, m-tiles (strided), num_experts).
extern "C" __global__ void __launch_bounds__(256, Q38_MINB) moe_q38_gate_up_silu(
    const unsigned char* __restrict__ A8,                // [tokens, K] E4M3
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ act8,                    // [rows, N] E4M3
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) Q38Smem sm;
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int M_expert = expert_offsets[e + 1] - m_start;
    if (M_expert <= 0) return;
    const unsigned char* Bg = (const unsigned char*)gate_packed_ptrs[e];
    const unsigned char* Bu = (const unsigned char*)up_packed_ptrs[e];
    if (Bg == nullptr || Bu == nullptr) return;
    q38_gate_up_silu_rows(sm, A8, Bg, (const unsigned char*)gate_scale_ptrs[e], gate_scale2_vals[e],
                          Bu, (const unsigned char*)up_scale_ptrs[e], up_scale2_vals[e], act8,
                          sorted_token_ids, (unsigned int)m_start, (unsigned int)M_expert, N, K);
}

// Grid: (N / 128, m-tiles (strided), num_experts).
extern "C" __global__ void __launch_bounds__(256, Q38_MINB) moe_q38_down(
    const unsigned char* __restrict__ act8,              // [rows, K] E4M3
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                       // [rows, N]
    const int* __restrict__ expert_offsets,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) Q38Smem sm;
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int M_expert = expert_offsets[e + 1] - m_start;
    if (M_expert <= 0) return;
    const unsigned char* B = (const unsigned char*)B_packed_ptrs[e];
    if (B == nullptr) return;
    q38_down_rows(sm, act8, B, (const unsigned char*)B_scale_ptrs[e], scale2_vals[e], C,
                  (unsigned int)m_start, (unsigned int)M_expert, N, K);
}

// Dense (the shared expert, `w4a16_gemm_t` x 3 + `moe_silu_mul` in the
// default): one weight set over rows [0, M), scalar scale2.
// Grid: (N / 64, ceil(M / 128)).
extern "C" __global__ void __launch_bounds__(256, Q38_MINB) moe_q38_dense_gate_up_silu(
    const unsigned char* __restrict__ A8,                // [M, K] E4M3
    const unsigned char* __restrict__ gate_packed,       // [K/2, N]
    const unsigned char* __restrict__ gate_scale,        // [K/16, N]
    const float gate_scale2,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float up_scale2,
    unsigned char* __restrict__ act8,                    // [M, N] E4M3
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) Q38Smem sm;
    q38_gate_up_silu_rows(sm, A8, gate_packed, gate_scale, gate_scale2, up_packed, up_scale,
                          up_scale2, act8, nullptr, 0u, M, N, K);
}

// Grid: (N / 128, ceil(M / 128)).
extern "C" __global__ void __launch_bounds__(256, Q38_MINB) moe_q38_dense_down(
    const unsigned char* __restrict__ act8,              // [M, K] E4M3
    const unsigned char* __restrict__ B_packed,          // [K/2, N]
    const unsigned char* __restrict__ B_scale,           // [K/16, N]
    const float scale2,
    __nv_bfloat16* __restrict__ C,                       // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) Q38Smem sm;
    q38_down_rows(sm, act8, B_packed, B_scale, scale2, C, 0u, M, N, K);
}

// ── router weight -> BF16 ──
// The router GEMM ran `w4a16_gemm` (w4a16_gemm.cu: 64x64 tiles, K step 16,
// no pipelining; 6.7 TFLOP/s at 16K tokens x 512 experts on GB10). Its B
// values are `bf16((LUT[q] * float(scale)) * scale2)` and its products one
// FP32 accumulator per output over m16n8k16 MMAs in increasing k -- exactly
// what `dense_gemm_bf16_pipelined` computes over a BF16 [N, K] weight. This
// writes that weight from the [N, K/2] packed / [N, K/16] scale planes (the
// layout w4a16_gemm reads: row per output column `n`, low nibble = even k), so
// the router logits, and the routing, are byte-identical.
// Grid: (ceil(N * K / 2 / 256)), Block 256: one packed byte (two k) a thread.
extern "C" __global__ void moe_q38_router_dequant(
    const unsigned char* __restrict__ B_packed,   // [N, K/2]
    const unsigned char* __restrict__ B_scale,    // [N, K/16] FP8 E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ out,              // [N, K]
    const unsigned int N,
    const unsigned int K
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;   // packed byte
    const unsigned int half_k = K / 2;
    if (i >= N * half_k) return;
    const unsigned int n = i / half_k;
    const unsigned int kp = i - n * half_k;
    const unsigned char packed = B_packed[i];
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = B_scale[(size_t)n * (K / Q38_GROUP) + (2 * kp) / Q38_GROUP];
    const float lo = Q38_E2M1_LUT[packed & 0xF] * (float)f * scale2;
    const float hi = Q38_E2M1_LUT[packed >> 4] * (float)f * scale2;
    *reinterpret_cast<unsigned int*>(out + (size_t)n * K + 2 * kp) =
          (unsigned int)__bfloat16_as_ushort(__float2bfloat16(lo))
        | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

// ── unpermute + top-k weighted reduce, local experts only ──
// `moe_unpermute_reduce_indexed` (common/moe_permute.cu) sums all top-k routes
// of a token, `acc += w * float(row[c])` in slot order, and under EP reads
// the remote experts' rows as zeros -- which is why the default memsets the
// gate, up and down outputs (1.2 GB a layer at a 16K chunk) before every MoE
// prefill. This sums only the routes whose expert is in [local_start,
// local_end), in the same slot order with the same expression. A skipped
// route would have added `w * +0.0 = +0.0` to an accumulator that is never
// -0.0 (it starts at +0.0 and a round-to-nearest sum is -0.0 only when both
// operands are), so the output bytes are identical and the remote rows never
// need writing. (glm-5.3-flash's `moe_unpermute_reduce_indexed_ep` is the
// same idea; its 16-byte twin caps top-k at 8, this one takes any top-k.)
//
// Eight columns per thread, one 16-byte load per local route.
// Grid: (num_tokens), Block: (hidden / 8); hidden % 8 == 0.
extern "C" __global__ void moe_q38_unpermute_local(
    const __nv_bfloat16* __restrict__ expert_output,  // [rows, hidden]
    __nv_bfloat16* __restrict__ output,                // [tokens, hidden]
    const int* __restrict__ token_to_perm,             // [tokens, topk]
    const int* __restrict__ topk_ids,                  // [tokens, topk]
    const float* __restrict__ topk_weights,            // [tokens, topk]
    const unsigned int hidden,
    const unsigned int num_tokens,
    const unsigned int topk,
    const unsigned int local_start,
    const unsigned int local_end
) {
    const unsigned int token = blockIdx.x;
    const unsigned int c = threadIdx.x * 8u;
    if (token >= num_tokens || c >= hidden) return;
    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
    for (unsigned int k = 0; k < topk; ++k) {
        const unsigned int slot = token * topk + k;
        const int expert = topk_ids[slot];
        if (expert < (int)local_start || expert >= (int)local_end) continue;
        const int perm_row = token_to_perm[slot];
        const float w = topk_weights[slot];
        const uint4 raw = *reinterpret_cast<const uint4*>(expert_output + (size_t)perm_row * hidden + c);
        const unsigned int wd[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wd[j] & 0xFFFFu)));
            const float hi = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wd[j] >> 16)));
            acc[2 * j] += w * lo;
            acc[2 * j + 1] += w * hi;
        }
    }
    uint4 o;
    unsigned int* ow = reinterpret_cast<unsigned int*>(&o);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        ow[j] = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc[2 * j]))
              | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc[2 * j + 1])) << 16);
    *reinterpret_cast<uint4*>(output + (size_t)token * hidden + c) = o;
}
