// SPDX-License-Identifier: AGPL-3.0-only

// qwen4_exp attention prefill projections: an FP8 x FP8 GEMM with the
// k-chain of `fp8_fp8_gemm_t_m128` / `fp8_gemm_t_m128` (w4a16_gemm.cu, the
// qwen3.6-35b-a3b file this target symlinks), on a better schedule.
// Opt-in: ATLAS_QWEN4EXP_PREFILL_FP8_W2=1.
//
// WHY. The QSA attention layers' q+gate projection (TP2: 16016 x 6144 x
// 2560, FP8 activations) runs `fp8_fp8_gemm_t_m128` at ~65 TFLOP/s of the
// 210 an FP8 mma.sync reaches on GB10 (7.67 ms a layer on the pair), and
// o_proj (`fp8_gemm_t_m128`, BF16 activations converted in its K loop)
// 3.27 ms. Those kernels run 4 warps over a 128 x 128 tile in two 64-row
// halves: every warp re-reads all 16 n-tiles of B from shared memory for
// each half, and a K step of 32 sits between two barriers with one stage
// in flight.
//
// WHAT. 8 warps on a 2 x 4 grid (each 64 rows x 32 columns: a quarter of B
// and half of A per k32), a 3-stage cp.async ring, E4M3 A in shared memory
// (o_proj converts A once with moe_q38_a_to_e4m3, the same
// cvt.rn.satfinite.e4m3x2 of float(bf16) the default does per fragment).
//
// EXACTNESS. Every output is one FP32 accumulator over the
// `mma.sync.m16n8k32.e4m3` MMAs in increasing k on the same operand bytes --
// the default's chain -- then `__float2bfloat16`. K % 32 == 0 (the default
// zero-fills a partial K step; this refuses it). `scripts/dev/qwen4exp_fp8_gemm_bench.cu`
// compares every output byte.
//
// Grid: (ceil(N / 128), ceil(M / 128), 1), Block: (256, 1, 1), dynamic shared
// memory QF_STAGES * 256 * (QF_BK + 16) bytes; K % QF_BK == 0.

#include <cuda_bf16.h>

#define QF_BM 128
#define QF_BN 128
#ifndef QF_BK
#define QF_BK 128     // 32: 7.2, 64: 5.9, 128: 5.0 ms (q+gate, TP2 16K)
#endif
#ifndef QF_STAGES
#define QF_STAGES 2   // 73.7 KB; 3 stages at BK 128 exceed the 99 KB a CTA may take
#endif
#define QF_STRIDE (QF_BK + 16)   // 16-byte aligned rows, conflict-free u32 fragment reads

__device__ __forceinline__ void qf_cp16(void* dst_smem, const void* src_gmem, bool pred) {
    const unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    const unsigned int n = pred ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src_gmem), "r"(n)
                 : "memory");
}

extern "C" __global__ void __launch_bounds__(256, 2) qwen4exp_fp8_gemm_w2(
    const unsigned char* __restrict__ A8,   // [M, K] E4M3
    const unsigned char* __restrict__ B8,   // [N, K] E4M3
    __nv_bfloat16* __restrict__ C,          // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    extern __shared__ __align__(16) unsigned char qf_smem[];
    auto sa = reinterpret_cast<unsigned char (*)[QF_BM][QF_STRIDE]>(qf_smem);
    auto sb = reinterpret_cast<unsigned char (*)[QF_BN][QF_STRIDE]>(qf_smem + QF_STAGES * QF_BM * QF_STRIDE);
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31u;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int wm = warp & 1u, wn = warp >> 1;
    const unsigned int m0 = blockIdx.y * QF_BM, n0 = blockIdx.x * QF_BN;
    const unsigned int steps = K / QF_BK;

    // BK/32 16-byte copies of A and of B per thread per stage.
    auto issue = [&](unsigned int s) {
        if (s < steps) {
            const unsigned int b = s % QF_STAGES;
            #pragma unroll
            for (unsigned int c = tid; c < QF_BM * (QF_BK / 16); c += 256) {
                const unsigned int r = c / (QF_BK / 16), col = (c % (QF_BK / 16)) * 16;
                const unsigned int kb = s * QF_BK + col;
                const unsigned int ga = m0 + r, gb = n0 + r;
                qf_cp16(&sa[b][r][col], &A8[(unsigned long long)(ga < M ? ga : 0) * K + kb], ga < M);
                qf_cp16(&sb[b][r][col], &B8[(unsigned long long)(gb < N ? gb : 0) * K + kb], gb < N);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    #pragma unroll
    for (unsigned int s = 0; s < QF_STAGES - 1; ++s) issue(s);
    for (unsigned int s = 0; s < steps; ++s) {
        asm volatile("cp.async.wait_group %0;" ::"n"(QF_STAGES - 2) : "memory");
        __syncthreads();   // stage s landed for every thread; stage s-1 is free
        issue(s + QF_STAGES - 1);
        const unsigned int b = s % QF_STAGES;
        #pragma unroll
        for (unsigned int kk = 0; kk < QF_BK; kk += 32) {
            unsigned int bf[4][2];
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                const unsigned int nc = wn * 32 + j * 8 + g;
                bf[j][0] = *(const unsigned int*)&sb[b][nc][kk + 4 * t4];
                bf[j][1] = *(const unsigned int*)&sb[b][nc][kk + 16 + 4 * t4];
            }
            #pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                const unsigned int r0 = wm * 64 + i * 16 + g, r1 = r0 + 8;
                const unsigned int a0 = *(const unsigned int*)&sa[b][r0][kk + 4 * t4];
                const unsigned int a1 = *(const unsigned int*)&sa[b][r1][kk + 4 * t4];
                const unsigned int a2 = *(const unsigned int*)&sa[b][r0][kk + 16 + 4 * t4];
                const unsigned int a3 = *(const unsigned int*)&sa[b][r1][kk + 16 + 4 * t4];
                #pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    float (&c)[4] = acc[i][j];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf[j][0]), "r"(bf[j][1]),
                          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");

    #pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int c0 = n0 + wn * 32 + j * 8 + t4 * 2, c1 = c0 + 1;
            #pragma unroll
            for (unsigned int h = 0; h < 2; ++h) {
                const unsigned int r = m0 + wm * 64 + i * 16 + g + h * 8;
                if (r >= M) continue;
                if (c0 < N) C[(unsigned long long)r * N + c0] = __float2bfloat16(acc[i][j][h * 2]);
                if (c1 < N) C[(unsigned long long)r * N + c1] = __float2bfloat16(acc[i][j][h * 2 + 1]);
            }
        }
    }
}
