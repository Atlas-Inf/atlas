// SPDX-License-Identifier: AGPL-3.0-only
//
// GDN in_proj_ba + gates for prefill on BF16 tensor cores: the math of
// `dense_gemm_ba_gates_prefill` (common/ssm_preprocess.cu) — FP32 dot
// products of the BF16 activations and BF16 weights, then
//   beta = sigmoid(b),  gate = exp(-exp(A_log) * softplus(a + dt_bias))
// into the interleaved FP32 [gate(nv) | beta(nv)] rows. That kernel runs one
// CTA per token and four outputs, so every token re-reads its weight rows
// from L2 (1.3 GB at 1k rows for a 1 GFLOP projection: 0.65 ms per layer on
// GB10). Here a CTA computes BA_TOK tokens x BA_OUT outputs with m16n8k16
// BF16 MMA from staged tiles. Products are exact in FP32 and only the
// summation order differs: equal to rounding.
//
// Grid: (N / BA_OUT, ceil(M / BA_TOK)), block 128. N % BA_OUT == 0, K % 64 == 0.

#include "../../common/bf16mma_dequant.cuh"

#define BA_TOK 32
#define BA_OUT 48
#define BA_KS 64
#define BA_LD 72   // bf16 per staged row: 144 B, conflict-free fragment loads

struct __align__(16) BaStage {
    __nv_bfloat16 a[BA_TOK][BA_LD];
    __nv_bfloat16 w[BA_OUT][BA_LD];
};

extern "C" __global__ void __launch_bounds__(128)
gdn_ba_gates_prefill(const __nv_bfloat16* __restrict__ A,   // [M, K_stride]
                     const __nv_bfloat16* __restrict__ B,   // [N, K]
                     const float* __restrict__ A_log,       // [nv]
                     const float* __restrict__ dt_bias,     // [nv]
                     float* __restrict__ gate_out,          // [M, gate_stride]
                     unsigned int M, unsigned int N, unsigned int K, unsigned int K_stride,
                     unsigned int gate_stride, unsigned int nv, unsigned int vheads_per_group)
{
    __shared__ BaStage st[2];
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned int g = lane >> 2, t = lane & 3;
    const unsigned int t0 = blockIdx.y * BA_TOK, n0 = blockIdx.x * BA_OUT;
    // Warp (wt, wn): tokens 16 wt.., outputs 24 wn.. (three n8 tiles).
    const unsigned int wt = warp & 1, wn = warp >> 1;
    const int ksteps = K / BA_KS;

    auto issue = [&](int buf, int s) {
        const unsigned int kb = s * BA_KS;
        for (unsigned int c = tid; c < (BA_TOK + BA_OUT) * 8; c += 128) {
            const unsigned int r = c >> 3, cc = (c & 7) * 8;
            if (r < BA_TOK) {
                const bool ok = t0 + r < M;
                bm_cp16(&st[buf].a[r][cc], A + (unsigned long long)(ok ? t0 + r : 0) * K_stride + kb + cc, ok);
            } else {
                const unsigned int o = r - BA_TOK;
                bm_cp16(&st[buf].w[o][cc], B + (unsigned long long)(n0 + o) * K + kb + cc, true);
            }
        }
    };

    float acc[3][4] = {};
    issue(0, 0);
    bm_commit();
    for (int s = 0; s < ksteps; s++) {
        bm_wait<0>();
        __syncthreads();
        if (s + 1 < ksteps) issue((s + 1) & 1, s + 1);
        bm_commit();
        const BaStage& sb = st[s & 1];
        #pragma unroll
        for (int kk = 0; kk < BA_KS; kk += 16) {
            const __nv_bfloat16* ar = &sb.a[16 * wt + g][kk + 2 * t];
            const unsigned int a0 = *(const unsigned int*)ar;
            const unsigned int a1 = *(const unsigned int*)(ar + 8 * BA_LD);
            const unsigned int a2 = *(const unsigned int*)(ar + 8);
            const unsigned int a3 = *(const unsigned int*)(ar + 8 * BA_LD + 8);
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                const __nv_bfloat16* br = &sb.w[24 * wn + 8 * j + g][kk + 2 * t];
                bm_mma(acc[j], a0, a1, a2, a3, *(const unsigned int*)br, *(const unsigned int*)(br + 8));
            }
        }
    }

    const unsigned int group_dim_ba = 2 * vheads_per_group;
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // acc[j][i]: token 16 wt + g + 8 (i >> 1), output 24 wn + 8 j + 2 t + (i & 1).
            const unsigned int tok = t0 + 16 * wt + g + 8 * (i >> 1);
            const unsigned int n = n0 + 24 * wn + 8 * j + 2 * t + (i & 1);
            if (tok >= M) continue;
            const float result = acc[j][i];
            const unsigned int within_group = n % group_dim_ba, group = n / group_dim_ba;
            float* gate_tok = gate_out + (unsigned long long)tok * gate_stride;
            if (within_group < vheads_per_group) {
                const unsigned int vh = group * vheads_per_group + within_group;
                gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
            } else {
                const unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
                const float A_val = __expf(fminf(A_log[vh], 20.0f));
                const float dt = __logf(1.0f + __expf(fminf(result + dt_bias[vh], 20.0f)));
                gate_tok[vh] = __expf(-A_val * dt);
            }
        }
    }
}
