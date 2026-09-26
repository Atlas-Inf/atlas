// SPDX-License-Identifier: AGPL-3.0-only

// QSA indexer scoring on TENSOR CORES.
//
// `qsa_score_rows_exact` is the worst-performing kernel in a long-context
// prefill: 926 ms for 1.55 TFLOP = 1.67 TFLOP/s, against 26.6 for MoE and 30.1
// for the dense tensor-core attention on this same box (TTFT_GAP.md 31c). Its
// tile was swept in both directions and moved it 0.10 s -- below the noise
// floor. Like the scalar attention kernel before it, its time is in its
// INSTRUCTION CLASS, not its schedule.
//
// WHAT IT COMPUTES, unchanged:
//     score[r][b] = (sum over h of max(dot(q[r][h][:], k[b][:]), 0)) * rsqrt(hd)
// with `b >= (first_pos + r + 1) / ratio` masked to -1e30.
//
// MAPPING. One warp per indexer head (n_heads == 4 == warps), each computing a
// full [BM=16, BN=32] score tile for its head with `mma.sync.m16n8k16`, then a
// shared-memory reduction sums the four heads IN HEAD ORDER, matching the
// reference's `acc = fadd(acc, fmaxf(dot, 0))` over h = 0..3.
//
// PRECISION. `block_keys` is already BF16, so only Q is rounded. That was the
// stated objection to this kernel, and it was priced BEFORE the kernel was
// written: `-DQSA_SCORE_PROBE_BF16_Q` rounds Q inside the otherwise-exact
// scorer, and `ppl.py` puts above-bound perplexity at 4.58049 against the F32
// scorer's 4.60664 -- BF16 Q measures BETTER, not worse (34). The MMA also
// reassociates the 128-term contraction, which 30a prices at +0.036%.
//
// NOT bit-identical. Gated behind `ATLAS_QSA_SCORE_TC=1`, default OFF.

#include <cuda_bf16.h>

#define QSC_BM 16
#define QSC_BN 32
#define QSC_PAD 8
#define QSC_MAXH 4

extern "C" __global__ void __launch_bounds__(128, 1)
qsa_score_rows_tc(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys, // [n_blocks, hd]
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rows,
    const unsigned int n_blocks_max
) {
    const unsigned int r0 = blockIdx.x * QSC_BM;
    const unsigned int b0 = blockIdx.y * QSC_BN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3u;
    const unsigned int ldq = hd + QSC_PAD;

    extern __shared__ __nv_bfloat16 qsc_smem[];
    __nv_bfloat16* sQ = qsc_smem;                                  // [H][BM][ldq]
    __nv_bfloat16* sK = sQ + (size_t)n_heads * QSC_BM * ldq;       // [BN][ldq]
    float* sAcc = (float*)(sK + (size_t)QSC_BN * ldq);             // [H][BM][BN]

    // Q -> BF16 in shared. Rounding here is the whole precision change; see the
    // header. Out-of-range rows are zeroed so their scores are harmless.
    for (unsigned int i = tid; i < n_heads * QSC_BM * hd; i += 128) {
        const unsigned int h = i / (QSC_BM * hd);
        const unsigned int rest = i - h * QSC_BM * hd;
        const unsigned int rr = rest / hd;
        const unsigned int d = rest - rr * hd;
        const unsigned int r = r0 + rr;
        const float v = (r < rows) ? q[((size_t)r * n_heads + h) * hd + d] : 0.0f;
        sQ[((size_t)h * QSC_BM + rr) * ldq + d] = __float2bfloat16(v);
    }
    for (unsigned int i = tid; i < QSC_BN * hd; i += 128) {
        const unsigned int bb = i / hd;
        const unsigned int d = i - bb * hd;
        const unsigned int b = b0 + bb;
        sK[(size_t)bb * ldq + d] =
            (b < n_blocks_max) ? block_keys[(size_t)b * hd + d] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // One warp per head. Anything past n_heads has no work but must still reach
    // the barriers below.
    float acc[QSC_BN / 8][4];
    #pragma unroll
    for (int nt = 0; nt < QSC_BN / 8; ++nt) {
        acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
    }
    if (warp_id < n_heads) {
        const unsigned short* sQh =
            (const unsigned short*)(sQ + (size_t)warp_id * QSC_BM * ldq);
        const unsigned short* sKu = (const unsigned short*)sK;
        for (unsigned int ks = 0; ks < hd / 16; ++ks) {
            const unsigned int kb = ks * 16;
            // SM121 workaround: manual A/B register loading -- ldmatrix and
            // ldmatrix.trans both produce wrong results on GB10.
            const unsigned int ar0 = group_id;
            const unsigned int ar1 = group_id + 8;
            const unsigned int ac0 = kb + tid_in_group * 2;
            const unsigned int ac1 = ac0 + 8;
            const unsigned int a0 = *(const unsigned int*)&sQh[ar0 * ldq + ac0];
            const unsigned int a1 = *(const unsigned int*)&sQh[ar1 * ldq + ac0];
            const unsigned int a2 = *(const unsigned int*)&sQh[ar0 * ldq + ac1];
            const unsigned int a3 = *(const unsigned int*)&sQh[ar1 * ldq + ac1];
            #pragma unroll
            for (int nt = 0; nt < QSC_BN / 8; ++nt) {
                const unsigned int n_col = nt * 8 + group_id;
                const unsigned int b0k = ((unsigned int)sKu[n_col * ldq + ac0 + 1] << 16) |
                                          (unsigned int)sKu[n_col * ldq + ac0];
                const unsigned int b1k = ((unsigned int)sKu[n_col * ldq + ac1 + 1] << 16) |
                                          (unsigned int)sKu[n_col * ldq + ac1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0k), "r"(b1k),
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3])
                );
            }
        }
        // ReLU before the cross-head sum, exactly as the reference does.
        float* sA = sAcc + (size_t)warp_id * QSC_BM * QSC_BN;
        #pragma unroll
        for (int nt = 0; nt < QSC_BN / 8; ++nt) {
            const unsigned int c0 = nt * 8 + tid_in_group * 2;
            sA[(size_t)group_id * QSC_BN + c0]           = fmaxf(acc[nt][0], 0.0f);
            sA[(size_t)group_id * QSC_BN + c0 + 1]       = fmaxf(acc[nt][1], 0.0f);
            sA[(size_t)(group_id + 8) * QSC_BN + c0]     = fmaxf(acc[nt][2], 0.0f);
            sA[(size_t)(group_id + 8) * QSC_BN + c0 + 1] = fmaxf(acc[nt][3], 0.0f);
        }
    }
    __syncthreads();

    // Sum the heads in order 0..n_heads-1, matching the reference's
    // `acc = fadd(acc, fmaxf(dot, 0))`, then scale and mask.
    const float inv = rsqrtf((float)hd);
    for (unsigned int i = tid; i < QSC_BM * QSC_BN; i += 128) {
        const unsigned int rr = i / QSC_BN;
        const unsigned int bb = i - rr * QSC_BN;
        const unsigned int r = r0 + rr;
        const unsigned int b = b0 + bb;
        if (r >= rows || b >= n_blocks_max) continue;
        const unsigned int complete = (first_pos + r + 1) / ratio;
        float* out = scores + (size_t)r * score_stride + b;
        if (b >= complete) { *out = -1e30f; continue; }
        float s = 0.0f;
        for (unsigned int h = 0; h < n_heads; ++h) {
            s = __fadd_rn(s, sAcc[((size_t)h * QSC_BM + rr) * QSC_BN + bb]);
        }
        *out = s * inv;
    }
}
