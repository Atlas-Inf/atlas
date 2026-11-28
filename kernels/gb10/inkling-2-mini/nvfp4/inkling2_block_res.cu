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

// Decode rows: two launches, the hidden dimension split into `chunks` CTAs a row
// (gridDim.x), each covering the channels e = chunk * INK_THREADS + threadIdx.x (mod
// chunks * INK_THREADS), so a row spreads over the GPU. `_sums` writes each chunk's
// per-state sum of squares and query dots, `_out` adds the chunks up, takes the softmax
// and writes its channels. Grid (chunks, rows) for both. Prefill rows: `_row`, one CTA a
// row for both passes, 8 channels a thread.

// This chunk's per-state sum of squares and query dots of `row` into sums[0 .. PER)
// (global or shared): [state][0] = sum s^2, [state][1 + q] = sum q . s.
#define INK_RES_PER (INK_MAX_STATES * (1 + INK_MAX_Q))
__device__ __forceinline__ void ink_res_sums(
    const __nv_bfloat16* __restrict__ blocks, const float* __restrict__ partial_f32,
    const __nv_bfloat16* __restrict__ partial_bf16, const float* __restrict__ queries, float* sums,
    const unsigned long long row, const unsigned int chunk, const unsigned int chunks, const unsigned int H,
    const unsigned int nb, const unsigned int row_stride, const unsigned long long block_stride,
    const unsigned int nq) {
    const unsigned int ns = nb + 1;
    float acc[INK_MAX_STATES][1 + INK_MAX_Q];
#pragma unroll
    for (int j = 0; j < INK_MAX_STATES; j++)
#pragma unroll
        for (int q = 0; q <= INK_MAX_Q; q++) acc[j][q] = 0.0f;
    for (unsigned int e = chunk * INK_THREADS + threadIdx.x; e < H; e += chunks * INK_THREADS) {
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
    if (threadIdx.x < INK_RES_PER) {
        const unsigned int j = threadIdx.x / (1 + INK_MAX_Q), q = threadIdx.x % (1 + INK_MAX_Q);
        float v = 0.0f;
        for (unsigned int w = 0; w < INK_THREADS / 32; w++) v += red[w][j][q];
        sums[threadIdx.x] = v;
    }
}

// The softmax weights (times each state's rstd) of `row` from its chunks' sums, into
// p[q][state], then its output channels of this chunk.
__device__ __forceinline__ void ink_res_out(
    const __nv_bfloat16* __restrict__ blocks, const float* __restrict__ partial_f32,
    const __nv_bfloat16* __restrict__ partial_bf16, const float* sums, float* __restrict__ out,
    const unsigned long long row, const unsigned int chunk, const unsigned int chunks, const unsigned int rows,
    const unsigned int H, const unsigned int nb, const unsigned int row_stride,
    const unsigned long long block_stride, const unsigned int nq, const float eps) {
    const unsigned int ns = nb + 1;
    __shared__ float p[INK_MAX_Q][INK_MAX_STATES];   // softmax weight x rstd
    if (threadIdx.x < nq) {
        const unsigned int q = threadIdx.x;
        float logit[INK_MAX_STATES], rstd[INK_MAX_STATES], mx = -1e30f;
        // compile-time state indices keep logit / rstd in registers
#pragma unroll
        for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= ns) break;
            float ss = 0.0f, d = 0.0f;
            for (unsigned int c = 0; c < chunks; c++) {
                const float* t = sums + c * INK_RES_PER + j * (1 + INK_MAX_Q);
                ss += t[0];
                d += t[1 + q];
            }
            rstd[j] = rsqrtf(ss / (float)H + eps);
            logit[j] = d * rstd[j] * rsqrtf((float)H);
            mx = fmaxf(mx, logit[j]);
        }
        float sum = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= ns) break;
            logit[j] = __expf(logit[j] - mx);
            sum += logit[j];
        }
#pragma unroll
        for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= ns) break;
            p[q][j] = logit[j] / sum * rstd[j];
        }
    }
    __syncthreads();
    for (unsigned int e = chunk * INK_THREADS + threadIdx.x; e < H; e += chunks * INK_THREADS) {
        float o[INK_MAX_Q] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int j = 0; j < ns; j++) {
            const float s = ink_state(blocks, partial_f32, partial_bf16, row, row_stride, block_stride, H, nb, j, e);
#pragma unroll
            for (int q = 0; q < INK_MAX_Q; q++) o[q] += (q < (int)nq ? p[q][j] : 0.0f) * s;
        }
#pragma unroll
        for (unsigned int q = 0; q < INK_MAX_Q; q++) {
            if (q >= nq) break;
            out[((unsigned long long)q * rows + row) * H + e] = o[q];
        }
    }
}

// sums[row][chunk][...] (ink_res_sums)
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
    ink_res_sums(blocks, partial_f32, partial_bf16, queries, sums + (row * gridDim.x + blockIdx.x) * INK_RES_PER,
                 row, blockIdx.x, gridDim.x, H, nb, row_stride, block_stride, nq);
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
    ink_res_out(blocks, partial_f32, partial_bf16, sums + row * gridDim.x * INK_RES_PER, out, row, blockIdx.x,
                gridDim.x, rows, H, nb, row_stride, block_stride, nq, eps);
}

// Both passes for prefill rows, one CTA a row and 8 channels a thread: every state's 8
// channels load once (16-byte loads) and stay in registers for the softmax and the
// output, the sums reduce across the CTA. Each output's sum of squares goes to `sq`
// (or nowhere with NULL), sq[q * rows + row], for the loop message gates' rms. With
// `normed`, output q also leaves RMS-normed as inkling2_rmsnorm writes it,
// BF16(w_q * (o * rsqrt(mean(o^2) + norm_eps))) at normed + q * rows * H, so the fp32
// output need not be written (`out` NULL) and read back. `p_out` takes the row's mix
// weights (softmax x rstd, p[q][state]): the fused message gate rebuilds the outputs
// from the states with them, bit for bit, instead of reading fp32 outputs. The same
// formulas as the two decode kernels and inkling2_rmsnorm, summed in its own order.
// grid (rows), block H / 8.
__device__ __forceinline__ void ink_bf16x8(const uint4 u, float* f) {
    const unsigned int w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

extern "C" __global__ void __launch_bounds__(1024) inkling2_block_res_row(
    const __nv_bfloat16* __restrict__ blocks,
    const float* __restrict__ partial_f32,
    const __nv_bfloat16* __restrict__ partial_bf16,
    const float* __restrict__ queries,
    float* __restrict__ out,                    // [nq, rows, H]
    const unsigned int rows,
    const unsigned int H,
    const unsigned int nb,
    const unsigned int row_stride,
    const unsigned long long block_stride,
    const unsigned int nq,
    const float eps,
    float* __restrict__ sq,                     // [nq, rows] or NULL
    const __nv_bfloat16* __restrict__ nw0,      // output q's norm weight (normed only)
    const __nv_bfloat16* __restrict__ nw1,
    const __nv_bfloat16* __restrict__ nw2,
    __nv_bfloat16* __restrict__ normed,         // [nq, rows, H] or NULL
    const float norm_eps,
    float* __restrict__ p_out                   // [rows, INK_MAX_Q, INK_MAX_STATES] or NULL
) {
    atlas_pdl_enter();
    __shared__ float red[32][INK_RES_PER];
    __shared__ float p[INK_MAX_Q][INK_MAX_STATES];   // softmax weight x rstd
    const unsigned long long row = blockIdx.x;
    const unsigned int ns = nb + 1, e0 = threadIdx.x * 8, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int warps = blockDim.x / 32;
    // the row's states: saved blocks as BF16 words, the partial as fp32
    uint4 bs[INK_MAX_STATES - 1];
#pragma unroll
    for (unsigned int j = 0; j < INK_MAX_STATES - 1; j++) {
        if (j >= nb) break;
        bs[j] = *(const uint4*)(blocks + row * row_stride + (unsigned long long)j * block_stride + e0);
    }
    float pt[8];
    if (partial_f32 != nullptr) {
        const float4 a = *(const float4*)(partial_f32 + row * H + e0), b = *(const float4*)(partial_f32 + row * H + e0 + 4);
        pt[0] = a.x; pt[1] = a.y; pt[2] = a.z; pt[3] = a.w; pt[4] = b.x; pt[5] = b.y; pt[6] = b.z; pt[7] = b.w;
    } else {
        ink_bf16x8(*(const uint4*)(partial_bf16 + row * H + e0), pt);
    }
    auto state = [&](unsigned int j, float* f) {
        if (j < nb) ink_bf16x8(bs[j], f);
        else
#pragma unroll
            for (int i = 0; i < 8; i++) f[i] = pt[i];
    };
    float qv[INK_MAX_Q][8];
#pragma unroll
    for (unsigned int q = 0; q < INK_MAX_Q; q++)
#pragma unroll
        for (int i = 0; i < 8; i++) qv[q][i] = q < nq ? queries[(unsigned long long)q * H + e0 + i] : 0.0f;

    // per state: sum of squares and each query's dot, reduced over the CTA
#pragma unroll
    for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
        if (j >= ns) break;
        float f[8], acc[1 + INK_MAX_Q] = {};
        state(j, f);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[0] += f[i] * f[i];
#pragma unroll
            for (unsigned int q = 0; q < INK_MAX_Q; q++) acc[1 + q] += qv[q][i] * f[i];
        }
#pragma unroll
        for (unsigned int q = 0; q <= INK_MAX_Q; q++) {
            float v = acc[q];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0) red[warp][j * (1 + INK_MAX_Q) + q] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const unsigned int q = threadIdx.x;
        float logit[INK_MAX_STATES], rstd[INK_MAX_STATES], mx = -1e30f;
#pragma unroll
        for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= ns) break;
            float ss = 0.0f, d = 0.0f;
            for (unsigned int w = 0; w < warps; w++) {
                ss += red[w][j * (1 + INK_MAX_Q)];
                d += red[w][j * (1 + INK_MAX_Q) + 1 + q];
            }
            rstd[j] = rsqrtf(ss / (float)H + eps);
            logit[j] = d * rstd[j] * rsqrtf((float)H);
            mx = fmaxf(mx, logit[j]);
        }
        float sum = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= ns) break;
            logit[j] = __expf(logit[j] - mx);
            sum += logit[j];
        }
#pragma unroll
        for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
            if (j >= ns) break;
            p[q][j] = logit[j] / sum * rstd[j];
            if (p_out != nullptr) p_out[(row * INK_MAX_Q + q) * INK_MAX_STATES + j] = p[q][j];
        }
    }
    __syncthreads();

    // outputs from the registers
    float o[INK_MAX_Q][8] = {};
#pragma unroll
    for (unsigned int j = 0; j < INK_MAX_STATES; j++) {
        if (j >= ns) break;
        float f[8];
        state(j, f);
#pragma unroll
        for (unsigned int q = 0; q < INK_MAX_Q; q++) {
            const float w = q < nq ? p[q][j] : 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) o[q][i] += w * f[i];
        }
    }
    float ss[INK_MAX_Q];
#pragma unroll
    for (unsigned int q = 0; q < INK_MAX_Q; q++) {
        ss[q] = 0.0f;
        if (q >= nq) break;
        if (out != nullptr) {
            float* dst = out + ((unsigned long long)q * rows + row) * H + e0;
            *(float4*)dst = make_float4(o[q][0], o[q][1], o[q][2], o[q][3]);
            *(float4*)(dst + 4) = make_float4(o[q][4], o[q][5], o[q][6], o[q][7]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) ss[q] += o[q][i] * o[q][i];
    }
    if (sq == nullptr && normed == nullptr) return;
    __syncthreads();   // red is reused
    __shared__ float tot[INK_MAX_Q];
#pragma unroll
    for (unsigned int q = 0; q < INK_MAX_Q; q++) {
        float v = ss[q];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) red[warp][q] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        float v = 0.0f;
        for (unsigned int w = 0; w < warps; w++) v += red[w][threadIdx.x];
        tot[threadIdx.x] = v;
        if (sq != nullptr) sq[(unsigned long long)threadIdx.x * rows + row] = v;
    }
    if (normed == nullptr) return;
    __syncthreads();
#pragma unroll
    for (unsigned int q = 0; q < INK_MAX_Q; q++) {
        if (q >= nq) break;
        const __nv_bfloat16* w = q == 0 ? nw0 : (q == 1 ? nw1 : nw2);
        const float rstd = rsqrtf(tot[q] / (float)H + norm_eps);
        float wf[8];
        ink_bf16x8(*(const uint4*)(w + e0), wf);
        unsigned int pk[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            // inkling2_rmsnorm: (w * fp32(x) * rstd), one cast at the end
            const __nv_bfloat162 b = __floats2bfloat162_rn(wf[2 * i] * (o[q][2 * i] * rstd),
                                                           wf[2 * i + 1] * (o[q][2 * i + 1] * rstd));
            pk[i] = *(const unsigned int*)&b;
        }
        *(uint4*)(normed + ((unsigned long long)q * rows + row) * H + e0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
}
