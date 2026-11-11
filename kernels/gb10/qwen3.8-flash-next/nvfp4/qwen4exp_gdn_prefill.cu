// SPDX-License-Identifier: AGPL-3.0-only

// qwen4_exp GDN prefill: `dense_gemm_ba_gates_prefill` (common/ssm_preprocess.cu)
// with several tokens and every BA output in one CTA.
// Opt-in: ATLAS_QWEN4EXP_PREFILL_BA_ROWS=1.
//
// WHY. The default launches one 256-thread CTA per (token, group of 4 BA
// outputs): at the TP2 rank shape (48 outputs, K = 2560) a 16016-token chunk is
// 192192 CTAs of ~1100 cycles each, re-reading each token's 5 KB activation row
// 12 times -- 2.09 ms a layer on the pair (75 ms of a cold 16K prefill) for a
// GEMM whose bytes take ~0.3 ms.
//
// WHAT CHANGES. One CTA takes QBA_TOK tokens and every group of 4 outputs at
// once: each thread accumulates its slot of every (token, group), so one
// weight load feeds QBA_TOK tokens (the weight was re-read from L2 for every
// token: ~4 GB a 16K chunk) and one activation load every group. The
// transforms run after one barrier, from shared memory holding every
// output's two warp partials.
//
// EXACTNESS. Every output is computed by the same lanes in the same order:
// output n = 4 g + j belongs to threads [64 j, 64 j + 64) exactly as in the
// default's CTA for group g; each lane's partial runs over kv = lane,
// lane + 64, ... with the same products and adds; the same `__shfl_down_sync`
// tree folds each warp; the two warp partials are added in the same order and
// the same sigmoid / softplus / exp expressions follow. Built with
// `--fmad=false` like the rest of this target; the BF16 -> FP32 widening is
// exact wherever it is hoisted to. `scripts/dev/qwen4exp_ba_gates_bench.cu`
// compares every gate byte against the default kernel: GB10, 16016 tokens at
// the TP2 shape, 2.25 -> 0.97 ms.
//
// Grid: (ceil(M / QBA_TOK), 1, 1), Block: (256, 1, 1); N <= 4 * QBA_MAX_GROUPS
// (48: the TP2 rank shape), K % 8 == 0.

#include <cuda_bf16.h>

#ifndef QBA_TOK
#define QBA_TOK 2      // tokens a CTA (4: 1.04 ms, 1: 1.52 at the TP2 16K shape)
#endif
#ifndef QBA_MINB
#define QBA_MINB 4     // CTAs per SM the register budget targets (62 regs, no spills)
#endif
#define QBA_MAX_GROUPS 12

extern "C" __global__ void __launch_bounds__(256, QBA_MINB) qwen4exp_ba_gates_prefill_rows(
    const __nv_bfloat16* __restrict__ A,  // [M, K_stride] activations
    const __nv_bfloat16* __restrict__ B,  // [N, K] BA weight (row-major)
    const float* __restrict__ A_log,      // [nv]
    const float* __restrict__ dt_bias,    // [nv]
    float* __restrict__ gate_out,         // [M, gate_stride] FP32
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_stride,
    unsigned int gate_stride,
    unsigned int nv,
    unsigned int vheads_per_group
) {
    const unsigned int threads_per_out = 256 / 4;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = threadIdx.x % 32;
    const unsigned int groups = (N + 3) / 4;
    __shared__ float smem[QBA_TOK][QBA_MAX_GROUPS * 4 * 2];
    if (groups > QBA_MAX_GROUPS) return;
    const unsigned int tok0 = blockIdx.x * QBA_TOK;
    const unsigned int K_VEC = K / 8;

    // This thread's slot j = local_out of every group of 4 outputs, for each
    // of the CTA's tokens: one weight load feeds QBA_TOK tokens and one
    // activation load every group, each (token, output) keeping its own
    // accumulator in the default's order.
    float acc[QBA_TOK][QBA_MAX_GROUPS];
    #pragma unroll
    for (unsigned int t = 0; t < QBA_TOK; ++t) {
        #pragma unroll
        for (unsigned int g = 0; g < QBA_MAX_GROUPS; ++g) acc[t][g] = 0.0f;
    }
    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        // Widened once (BF16 -> FP32 is exact), used by every group.
        float af[QBA_TOK][8];
        #pragma unroll
        for (unsigned int t = 0; t < QBA_TOK; ++t) {
            const unsigned int token = tok0 + t < M ? tok0 + t : M - 1;
            const uint4 a_data = ((const uint4*)(A + (unsigned long long)token * K_stride))[kv];
            const __nv_bfloat16* ap = reinterpret_cast<const __nv_bfloat16*>(&a_data);
            #pragma unroll
            for (int e = 0; e < 8; ++e) af[t][e] = __bfloat162float(ap[e]);
        }
        #pragma unroll
        for (unsigned int g = 0; g < QBA_MAX_GROUPS; ++g) {
            const unsigned int n = g * 4 + local_out;
            if (g >= groups || n >= N) continue;
            const uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[kv];
            const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&b_data);
            float bf[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) bf[e] = __bfloat162float(bp[e]);
            #pragma unroll
            for (unsigned int t = 0; t < QBA_TOK; ++t) {
                // The default's order: (lo, hi) of each 32-bit word, words 0..3.
                #pragma unroll
                for (int e = 0; e < 8; ++e) acc[t][g] += af[t][e] * bf[e];
            }
        }
    }
    #pragma unroll
    for (unsigned int t = 0; t < QBA_TOK; ++t) {
        #pragma unroll
        for (unsigned int g = 0; g < QBA_MAX_GROUPS; ++g) {
            const unsigned int n = g * 4 + local_out;
            if (g >= groups || n >= N) continue;   // uniform over the warp
            float v = acc[t][g];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFF, v, offset);
            }
            if (warp_lane == 0) {
                smem[t][n * 2 + (lane / 32)] = v;
            }
        }
    }
    __syncthreads();
    for (unsigned int idx = threadIdx.x; idx < QBA_TOK * N; idx += blockDim.x) {
        const unsigned int t = idx / N, n = idx % N;
        const unsigned int token = tok0 + t;
        if (token >= M) continue;
        float* gate_tok = gate_out + (unsigned long long)token * gate_stride;
        float result = smem[t][n * 2] + smem[t][n * 2 + 1];
        unsigned int group_dim_ba = 2 * vheads_per_group;
        unsigned int within_group = n % group_dim_ba;
        unsigned int group = n / group_dim_ba;
        if (within_group < vheads_per_group) {
            unsigned int vh = group * vheads_per_group + within_group;
            gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
        } else {
            unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
            float a_log_val = A_log[vh];
            float dt_b = dt_bias[vh];
            float A_val = __expf(fminf(a_log_val, 20.0f));
            float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
            gate_tok[vh] = __expf(-A_val * dt);
        }
    }
}
