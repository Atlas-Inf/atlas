// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 short causal convolutions (kernel size k <= 4), for any mix of decode,
// verify and prefill rows. Reference: transformers PR #51050
// `Inkling2ShortConvolution` and `Inkling2DynamicShortConvolution` (Apache-2.0).
//
//   static:  y[t,c] = (sum_lag w[c, k-1-lag] x[t-lag, c] + x[t,c]) / sqrt(k + 1)
//            (w is the checkpoint's [C, 1, k] depthwise weight, current token last)
//   dynamic: y[t,c] = (sum_lag (s[lag, c] + d[t, c / hd, lag]) x[t-lag, c] + x[t,c]) / sqrt(2k + 1)
//            (s [k, C] static taps, current token first; d [rows, heads * k] per-token
//            taps projected from the layer input, shared within a head of hd channels)
//
// No activation (Atlas's causal_conv1d kernels fuse SiLU). Math in fp32; a
// sequence's state is fp32 [C, k-1], oldest input first, as transformers caches it,
// at states[s] + state_off (floats): each sequence keeps its own allocation.
//
// Rows are flat [rows, C] (row stride x_stride / out_stride elements). Sequence s
// owns rows s * rows_per_seq .. (s + 1) * rows_per_seq and the state at states[s]. A row reads
// earlier inputs of its own sequence from x, and the ones before the sequence's
// first row from the state. `inkling2_sconv` writes y only; `inkling2_sconv_state`
// then rolls each sequence's state forward over its rows (stream order keeps the
// two apart, so no row reads a state another thread is replacing). With one row a
// sequence (decode), `roll` has `inkling2_sconv` roll the state itself: the thread of
// (sequence, channel) is that state's only reader, and it has read it.
//
// The post-attention and post-MLP convs feed the residual stream directly: with
// `res_f32` the output y (BF16-rounded) is added as r = (restart ? 0 : res_x) + y into
// fp32 `res_f32`, with `res_bf16` as BF16(res_r + y); `out` is then not written.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define INK_MAX_K 4

extern "C" __global__ void inkling2_sconv(
    const __nv_bfloat16* __restrict__ x,        // [rows, x_stride]
    __nv_bfloat16* __restrict__ out,            // [rows, out_stride]
    float* const* __restrict__ states,          // [num_seqs] state bases
    const unsigned int state_off,               // floats from a base to this conv's [C, k-1]
    const unsigned int rows_per_seq,
    const __nv_bfloat16* __restrict__ w,        // static: [C, k]; dynamic: [k, C]
    const __nv_bfloat16* __restrict__ dyn,      // dynamic: [rows, dyn_stride]; NULL = static
    const unsigned int rows,
    const unsigned int channels,
    const unsigned int k,
    const unsigned int x_stride,
    const unsigned int out_stride,
    const unsigned int dyn_stride,
    const unsigned int head_dim,                // channels per dynamic-tap head
    const unsigned int roll,                    // 1: roll the state too (rows_per_seq == 1)
    const __nv_bfloat16* __restrict__ res_x,    // residual add (res_f32): the stream's BF16 rows
    const float* __restrict__ res_r,            // layer output (res_bf16): the fp32 residual
    float* __restrict__ res_f32,                // NULL, or r = (restart ? 0 : res_x) + y
    __nv_bfloat16* __restrict__ res_bf16,       // NULL, or BF16(res_r + y)
    const unsigned int restart
) {
    atlas_pdl_enter();
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int r = blockIdx.y;
    if (c >= channels || r >= rows) return;
    const int s = r / rows_per_seq;
    const int first = s * rows_per_seq;
    float* st = states[s] + state_off + (unsigned long long)c * (k - 1);
    const float xc = __bfloat162float(x[(unsigned long long)r * x_stride + c]);

    float acc = 0.0f;
    for (unsigned int lag = 0; lag < k; lag++) {
        const int src = (int)r - (int)lag;
        // inputs before the sequence's first row come from the state (newest last)
        const float xv = src >= first ? __bfloat162float(x[(unsigned long long)src * x_stride + c])
                                      : st[(int)(k - 1) - (first - src)];
        float tap;
        if (dyn == nullptr) {
            tap = __bfloat162float(w[(unsigned long long)c * k + (k - 1 - lag)]);
        } else {
            tap = __bfloat162float(w[(unsigned long long)lag * channels + c])
                + __bfloat162float(dyn[(unsigned long long)r * dyn_stride + (c / head_dim) * k + lag]);
        }
        acc += tap * xv;
    }
    const float denom = dyn == nullptr ? (float)(k + 1) : (float)(2 * k + 1);
    const __nv_bfloat16 y = __float2bfloat16((acc + xc) * rsqrtf(denom));
    const unsigned long long o = (unsigned long long)r * out_stride + c;
    if (res_f32 != nullptr) res_f32[o] = (restart ? 0.0f : __bfloat162float(res_x[o])) + __bfloat162float(y);
    else if (res_bf16 != nullptr) res_bf16[o] = __float2bfloat16(res_r[o] + __bfloat162float(y));
    else out[o] = y;
    if (roll) {
        // as inkling2_sconv_state with one row: shift, the new input last
        for (unsigned int j = 0; j + 2 < k; j++) st[j] = st[j + 1];
        st[k - 2] = xc;
    }
}

// state <- the last k-1 inputs of each sequence (its rows, then older state)
extern "C" __global__ void inkling2_sconv_state(
    const __nv_bfloat16* __restrict__ x,
    float* const* __restrict__ states,
    const unsigned int state_off,
    const unsigned int rows_per_seq,
    const unsigned int num_seqs,
    const unsigned int channels,
    const unsigned int k,
    const unsigned int x_stride
) {
    atlas_pdl_enter();
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int s = blockIdx.y;
    if (c >= channels || s >= num_seqs) return;
    const int first = s * rows_per_seq, end = first + rows_per_seq;
    float* st = states[s] + state_off + (unsigned long long)c * (k - 1);
    float old[INK_MAX_K - 1];
    for (unsigned int j = 0; j + 1 < k; j++) old[j] = st[j];
    const int n = end - first;
    for (unsigned int j = 0; j + 1 < k; j++) {
        // new position j holds the input (k-1-j) steps before the end
        const int back = (int)(k - 1 - j);
        st[j] = back <= n ? __bfloat162float(x[(unsigned long long)(end - back) * x_stride + c])
                          : old[j + n];
    }
}

// Verify rollback: one sequence's state (fp32 [C, k-1] at `state`, already restored to its
// pre-verify value) rolled forward over the first `rows` of its journaled conv inputs
// (BF16 [rows, C]), as inkling2_sconv_state would have. grid (ceil(C / 256)), block 256.
extern "C" __global__ void inkling2_sconv_rewind(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ state,
    const unsigned int channels,
    const unsigned int k,
    const unsigned int rows
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels || rows == 0) return;
    float* st = state + (unsigned long long)c * (k - 1);
    float old[INK_MAX_K - 1];
    for (unsigned int j = 0; j + 1 < k; j++) old[j] = st[j];
    const int n = (int)rows;
    for (unsigned int j = 0; j + 1 < k; j++) {
        const int back = (int)(k - 1 - j);
        st[j] = back <= n ? __bfloat162float(x[(unsigned long long)(n - back) * channels + c]) : old[j + n];
    }
}
