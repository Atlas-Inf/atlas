// SPDX-License-Identifier: AGPL-3.0-only
// Register-resident variant of common/mamba2_ssm_decode.cu's
// mamba2_ssm_prefill_persistent (omni-local overlay — common PTX untouched).
//
// Persistent kernel's bottleneck: H lived in ~33KB smem per CTA (2-3 CTAs/SM)
// and every token did a global load + two __syncthreads + smem RMW. Here each
// thread keeps its share of H in registers (state s = sub + j*SUB, j <
// STATE/SUB — same ownership and arithmetic order as the reference), and
// x/B/C/dt are staged per TB-token chunk in smem instead of per token.
//
// Omni dims (nemotron_h_omni): STATE=128, head_dim=64, SUB=4, ROWS=64,
// TB=16 — measured 3.5x faster than the smem kernel at seq 8192, byte-identical
// for both y and final H (see microbench RESULTS in the PR).
//
// Grid: (num_heads, batch_size, head_dim/ROWS)  Block: (ROWS*SUB, 1, 1)
// Dynamic smem: TB*(2*STATE + ROWS + 1) floats.
#include <cuda_bf16.h>

#define SSM_REG_STATE 128u
#define SSM_REG_SUB 4u
#define SSM_REG_TB 16u
#define SSM_REG_ROWS 64u

extern "C" __global__ void mamba2_ssm_prefill_reg(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ B_in,
    const __nv_bfloat16* __restrict__ C_in,
    const __nv_bfloat16* __restrict__ dt_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,
    float dt_min,
    float dt_max,
    unsigned int x_stride,
    unsigned int bc_stride,
    unsigned int dt_stride,
    unsigned int y_stride
) {
    constexpr unsigned int STATE = SSM_REG_STATE;
    constexpr unsigned int SUB = SSM_REG_SUB;
    constexpr unsigned int TB = SSM_REG_TB;
    constexpr unsigned int ROWS = SSM_REG_ROWS;
    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int row_blk = blockIdx.z;
    if (head >= num_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hd0 = row_blk * ROWS;
    const unsigned int hd = hd0 + tid / SUB;
    const unsigned int sub = tid % SUB;

    const unsigned int heads_per_group = num_heads / n_groups;
    const unsigned int group = head / heads_per_group;
    const float neg_A = expf(A_log[head]);
    const float D_val = D_param[head];
    const float dt_bias_val = dt_bias[head];
    float* H = h_state + ((unsigned long long)(b * num_heads + head) * head_dim * state_size);

    constexpr unsigned int SJ = STATE / SUB;
    float h[SJ];
    #pragma unroll
    for (unsigned int j = 0; j < SJ; j++)
        h[j] = H[hd * STATE + sub + j * SUB];

    // staging: sdtB[TB][STATE], sC[TB][STATE], sX[TB][ROWS], sdA[TB]
    extern __shared__ float smem[];
    float* sdtB = smem;
    float* sC = sdtB + TB * STATE;
    float* sX = sC + TB * STATE;
    float* sdA = sX + TB * ROWS;
    const unsigned int nthreads = blockDim.x;

    for (unsigned int t0 = 0; t0 < seq_len; t0 += TB) {
        const unsigned int tn = min(TB, seq_len - t0);
        for (unsigned int i = tid; i < tn * STATE; i += nthreads) {
            const unsigned int tt = i / STATE, s = i - tt * STATE;
            const __nv_bfloat16* B_t = B_in + (unsigned long long)(t0 + tt) * bc_stride
                + b * n_groups * state_size + group * state_size;
            const __nv_bfloat16* C_t = C_in + (unsigned long long)(t0 + tt) * bc_stride
                + b * n_groups * state_size + group * state_size;
            float dt_val = (float)dt_raw[(unsigned long long)(t0 + tt) * dt_stride
                + b * num_heads + head] + dt_bias_val;
            dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
            dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
            sdtB[i] = dt_val * (float)B_t[s];
            sC[i] = (float)C_t[s];
        }
        for (unsigned int i = tid; i < tn * ROWS; i += nthreads) {
            const unsigned int tt = i / ROWS, r = i - tt * ROWS;
            sX[i] = (float)x[(unsigned long long)(t0 + tt) * x_stride
                + (unsigned long long)(b * num_heads + head) * head_dim + hd0 + r];
        }
        for (unsigned int tt = tid; tt < tn; tt += nthreads) {
            float dt_val = (float)dt_raw[(unsigned long long)(t0 + tt) * dt_stride
                + b * num_heads + head] + dt_bias_val;
            dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
            dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
            sdA[tt] = expf(-neg_A * dt_val);
        }
        __syncthreads();

        const unsigned int lr = tid / SUB;
        for (unsigned int tt = 0; tt < tn; tt++) {
            const float x_hd = sX[tt * ROWS + lr];
            const float dA = sdA[tt];
            float y = 0.0f;
            #pragma unroll
            for (unsigned int j = 0; j < SJ; j++) {
                const unsigned int s = sub + j * SUB;
                h[j] = dA * h[j] + x_hd * sdtB[tt * STATE + s];
                y += h[j] * sC[tt * STATE + s];
            }
            #pragma unroll
            for (unsigned int off = 1; off < SUB; off <<= 1)
                y += __shfl_down_sync(0xFFFFFFFFu, y, off);
            if (sub == 0u) {
                y += D_val * x_hd;
                output[(unsigned long long)(t0 + tt) * y_stride
                    + (unsigned long long)(b * num_heads + head) * head_dim + hd] =
                    __float2bfloat16(y);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int j = 0; j < SJ; j++)
        H[hd * STATE + sub + j * SUB] = h[j];
}
