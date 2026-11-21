// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 block attention residual: a sublayer input is a learned-query softmax over
// the token's saved block states and its current partial sum. Reference: transformers
// PR #51050 `Inkling2BlockAttentionResidual` and `Inkling2LoopMessageAttentionResidual`
// (Apache-2.0):
//
//   n_j     = rms0(s_j)            (gamma-free RMSNorm, eps 1e-6, fp32), j = blocks..., partial
//   p_{q,j} = softmax_j(q . n_j / sqrt(H))
//   out_q   = sum_j p_{q,j} n_j    (fp32)
//
// Saved blocks are BF16 layer inputs, block j of row r at r * row_stride + j * block_stride; the
// partial is fp32 or BF16 [rows, H]. Queries are fp32 [nq, H] (1 for an attention or MLP
// input, 3 for the q/k/v inputs of a full layer, 4 for a loop message).
// dot(q, n_j) = rstd_j * dot(q, s_j), so the first pass reads each state once for all
// queries and the second builds the outputs.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define INK_MAX_STATES 9   // up to 8 blocks + the partial
#define INK_MAX_Q 4
#define INK_THREADS 256

__device__ __forceinline__ float ink_state(const __nv_bfloat16* __restrict__ blocks, const float* __restrict__ pf,
                                           const __nv_bfloat16* __restrict__ pb, unsigned long long row,
                                           unsigned int row_stride, unsigned long long block_stride,
                                           unsigned int H, unsigned int nb, unsigned int j, unsigned int e) {
    if (j < nb) return __bfloat162float(blocks[row * row_stride + (unsigned long long)j * block_stride + e]);
    return pf != nullptr ? pf[row * H + e] : __bfloat162float(pb[row * H + e]);
}

// Two launches, the hidden dimension split into chunks of INK_THREADS so a decode row
// spreads over the GPU: `_sums` writes each chunk's per-state sum of squares and query
// dots, `_out` adds the chunks up, takes the softmax and writes its chunk of outputs.
// Grid (ceil(H / INK_THREADS), rows) for both.

// sums[row][chunk][state][0] = sum s^2, [1 + q] = sum q . s
extern "C" __global__ void inkling2_block_res_sums(
    const __nv_bfloat16* __restrict__ blocks,   // see row_stride / block_stride
    const float* __restrict__ partial_f32,      // [rows, H] or NULL
    const __nv_bfloat16* __restrict__ partial_bf16, // [rows, H] when partial_f32 is NULL
    const float* __restrict__ queries,          // [nq, H]
    float* __restrict__ sums,                   // [rows, chunks, INK_MAX_STATES, 1 + INK_MAX_Q]
    const unsigned int H,
    const unsigned int nb,                      // saved blocks (the partial is state nb)
    const unsigned int row_stride,              // elements between rows of a block
    const unsigned long long block_stride,      // elements between blocks
    const unsigned int nq
) {
    atlas_pdl_enter();
    const unsigned long long row = blockIdx.y;
    const unsigned int chunk = blockIdx.x, chunks = gridDim.x;
    const unsigned int e = chunk * INK_THREADS + threadIdx.x;
    const unsigned int ns = nb + 1;
    float acc[INK_MAX_STATES][1 + INK_MAX_Q];
#pragma unroll
    for (int j = 0; j < INK_MAX_STATES; j++)
#pragma unroll
        for (int q = 0; q <= INK_MAX_Q; q++) acc[j][q] = 0.0f;
    if (e < H) {
        float qv[INK_MAX_Q];
#pragma unroll
        for (int q = 0; q < INK_MAX_Q; q++) qv[q] = q < (int)nq ? queries[(unsigned long long)q * H + e] : 0.0f;
#pragma unroll
        for (int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= (int)ns) break;
            const float s = ink_state(blocks, partial_f32, partial_bf16, row, row_stride, block_stride, H, nb, j, e);
            acc[j][0] += s * s;
#pragma unroll
            for (int q = 0; q < INK_MAX_Q; q++) acc[j][1 + q] += qv[q] * s;
        }
    }
    __shared__ float red[INK_THREADS / 32][INK_MAX_STATES][1 + INK_MAX_Q];
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
    for (int j = 0; j < INK_MAX_STATES; j++)
#pragma unroll
        for (int q = 0; q <= INK_MAX_Q; q++) {
            float v = acc[j][q];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0) red[warp][j][q] = v;
        }
    __syncthreads();
    constexpr unsigned int PER = INK_MAX_STATES * (1 + INK_MAX_Q);
    if (threadIdx.x < PER) {
        const unsigned int j = threadIdx.x / (1 + INK_MAX_Q), q = threadIdx.x % (1 + INK_MAX_Q);
        float v = 0.0f;
        for (unsigned int w = 0; w < INK_THREADS / 32; w++) v += red[w][j][q];
        sums[(row * chunks + chunk) * PER + threadIdx.x] = v;
    }
}

extern "C" __global__ void inkling2_block_res_out(
    const __nv_bfloat16* __restrict__ blocks,
    const float* __restrict__ partial_f32,
    const __nv_bfloat16* __restrict__ partial_bf16,
    const float* __restrict__ sums,             // from inkling2_block_res_sums
    float* __restrict__ out,                    // [nq, rows, H]
    const unsigned int rows,
    const unsigned int H,
    const unsigned int nb,
    const unsigned int row_stride,
    const unsigned long long block_stride,
    const unsigned int nq,
    const float eps
) {
    atlas_pdl_enter();
    const unsigned long long row = blockIdx.y;
    const unsigned int chunks = gridDim.x;
    const unsigned int ns = nb + 1;
    constexpr unsigned int PER = INK_MAX_STATES * (1 + INK_MAX_Q);
    __shared__ float p[INK_MAX_Q][INK_MAX_STATES];   // softmax weight x rstd
    if (threadIdx.x < nq) {
        const unsigned int q = threadIdx.x;
        float logit[INK_MAX_STATES], rstd[INK_MAX_STATES], mx = -1e30f;
        for (unsigned int j = 0; j < ns; j++) {
            float ss = 0.0f, d = 0.0f;
            for (unsigned int c = 0; c < chunks; c++) {
                const float* t = sums + (row * chunks + c) * PER + j * (1 + INK_MAX_Q);
                ss += t[0];
                d += t[1 + q];
            }
            rstd[j] = rsqrtf(ss / (float)H + eps);
            logit[j] = d * rstd[j] * rsqrtf((float)H);
            mx = fmaxf(mx, logit[j]);
        }
        float sum = 0.0f;
        for (unsigned int j = 0; j < ns; j++) {
            logit[j] = __expf(logit[j] - mx);
            sum += logit[j];
        }
        for (unsigned int j = 0; j < ns; j++) p[q][j] = logit[j] / sum * rstd[j];
    }
    __syncthreads();
    const unsigned int e = blockIdx.x * INK_THREADS + threadIdx.x;
    if (e >= H) return;
    float o[INK_MAX_Q] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int j = 0; j < ns; j++) {
        const float s = ink_state(blocks, partial_f32, partial_bf16, row, row_stride, block_stride, H, nb, j, e);
#pragma unroll
        for (int q = 0; q < INK_MAX_Q; q++) o[q] += (q < (int)nq ? p[q][j] : 0.0f) * s;
    }
    for (unsigned int q = 0; q < nq; q++) out[((unsigned long long)q * rows + row) * H + e] = o[q];
}
