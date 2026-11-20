// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the token with the largest logit, without the whole
// head. A greedy step asks nothing else of its logits, and the head is 331
// MB a rank that a step reads at the speed of memory.
//
// A rank holds a quarter of the vocabulary, its rows of the head in BF16
// and a copy of them in FP8 (the draft head's). A step reads the copy,
// which gives every token a coarse logit, and a bound on how far the head's
// own logit can lie from it (`head_bound.rs`: |exact - coarse| <= |x| *
// bound[v], before the head's value is rounded to BF16). Then:
//
//   dsv41_head_low    a rank's best lower bound of a row, coarse - |x| *
//                     bound, and |x|. The ranks gather these: some token's
//                     logit is at least the largest of them.
//   dsv41_head_keep   a rank's tokens whose upper bound, coarse + |x| *
//                     bound, reaches that, compared as BF16 rounds them:
//                     rounding keeps order, so a token left out is strictly
//                     below the row's largest logit after it too, and can
//                     neither be it nor tie with it.
//   dsv41_head_exact  the kept tokens' logits by `dense_gemv_bf16`'s own
//                     additions in its order, so their bytes are the
//                     head's, and a block's best of them.
//   dsv41_head_pick   the best of every rank's, which the ranks gathered:
//                     the largest value, and the lowest token among equals,
//                     as `argmax_bf16` over the whole row picks.
//
// The token with the largest logit is always kept (its upper bound is at
// least its logit, which is at least any lower bound), with every token
// that ties with it, so the pick is the whole head's argmax. Where the
// bound rules little out (a flat row), more of the head is read, up to all
// of it, a token at a time.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_HEAD_THREADS 1024

// A block's sum, or its largest, of what its threads hold in `part`.
__device__ __forceinline__ void dsv41_head_reduce(float* part, bool largest) {
    for (unsigned int step = DSV41_HEAD_THREADS / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            const float other = part[threadIdx.x + step];
            part[threadIdx.x] = largest ? fmaxf(part[threadIdx.x], other) : part[threadIdx.x] + other;
        }
        __syncthreads();
    }
}

// x [rows, K] BF16; coarse [rows, n] FP32, this rank's tokens of each row;
// bound [n]. low[2 * row] = the row's best lower bound here, low[2 * row +
// 1] = |x|; count[row] = 0, for `dsv41_head_keep`. Grid rows, block 1024.
extern "C" __global__ void __launch_bounds__(DSV41_HEAD_THREADS) dsv41_head_low(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ coarse,
    const float* __restrict__ bound,
    float* __restrict__ low,
    unsigned int* __restrict__ count,
    unsigned int n,
    unsigned int K) {
    atlas_pdl_enter();
    __shared__ float part[DSV41_HEAD_THREADS];
    const unsigned int t = threadIdx.x, row = blockIdx.x;
    const __nv_bfloat16* of_row = x + (size_t)row * K;
    float squares = 0.0f;
    for (unsigned int k = t; k < K; k += DSV41_HEAD_THREADS) {
        const float v = __bfloat162float(of_row[k]);
        squares += v * v;
    }
    part[t] = squares;
    __syncthreads();
    dsv41_head_reduce(part, false);
    const float norm = sqrtf(part[0]);
    __syncthreads();
    const float* c = coarse + (size_t)row * n;
    float best = -INFINITY;
    for (unsigned int v = t; v < n; v += DSV41_HEAD_THREADS) best = fmaxf(best, c[v] - norm * bound[v]);
    part[t] = best;
    __syncthreads();
    dsv41_head_reduce(part, true);
    if (t == 0) {
        low[2 * row] = part[0];
        low[2 * row + 1] = norm;
        count[row] = 0u;
    }
}

// The tokens of this rank a row cannot rule out, in list[row * n ..] in no
// order, count[row] of them. low [parts, low_rows, 2]: every rank's
// `dsv41_head_low` of the row. Grid rows, block 1024.
extern "C" __global__ void __launch_bounds__(DSV41_HEAD_THREADS) dsv41_head_keep(
    const float* __restrict__ coarse,
    const float* __restrict__ bound,
    const float* __restrict__ low,
    unsigned int* __restrict__ list,
    unsigned int* __restrict__ count,
    unsigned int n,
    unsigned int parts,
    unsigned int low_rows) {
    atlas_pdl_enter();
    const unsigned int row = blockIdx.x;
    float reached = -INFINITY;
    for (unsigned int p = 0; p < parts; ++p) reached = fmaxf(reached, low[((size_t)p * low_rows + row) * 2]);
    reached = dsv41_bf16(reached);
    const float norm = low[(size_t)row * 2 + 1];
    const float* c = coarse + (size_t)row * n;
    for (unsigned int v = threadIdx.x; v < n; v += DSV41_HEAD_THREADS) {
        if (dsv41_bf16(c[v] + norm * bound[v]) >= reached) list[(size_t)row * n + atomicAdd(count + row, 1u)] = v;
    }
}

#define DSV41_HEAD_PER 4u
#define DSV41_HEAD_LANES 64u

// The head's logits of a row's kept tokens, and block (b, row)'s best of
// them in found[(row * blocks + b) * 2 ..] as (value bits, token), token
// DSV41_NO_TOKEN where it had none. A block takes four tokens at a time,
// every `blocks * 4`th from its own, 64 threads a token: the lanes, the
// order of a lane's additions and the sums across lanes and warps are
// `dense_gemv_bf16`'s (common/dense_gemv_bf16.cu), which this file's flags
// compile as that one's do, so a value's bytes are the head's.
//
// x [rows, K] BF16, head [n, K] BF16 (this rank's rows; the first is token
// `first`), K a multiple of 8. Grid (blocks, rows), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HEAD_PER * DSV41_HEAD_LANES) dsv41_head_exact(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ head,
    const unsigned int* __restrict__ list,
    const unsigned int* __restrict__ count,
    unsigned int* __restrict__ found,
    unsigned int n,
    unsigned int K,
    unsigned int first) {
    atlas_pdl_enter();
    __shared__ float halves[DSV41_HEAD_PER * 2u];
    __shared__ float values[DSV41_HEAD_PER];
    __shared__ unsigned int tokens[DSV41_HEAD_PER];
    const unsigned int row = blockIdx.y;
    const unsigned int group = threadIdx.x / DSV41_HEAD_LANES, lane = threadIdx.x % DSV41_HEAD_LANES;
    const unsigned int have = count[row] < n ? count[row] : n;
    const uint4* a_vec = (const uint4*)(x + (size_t)row * K);
    float best = 0.0f;
    unsigned int best_token = DSV41_NO_TOKEN;
    for (unsigned int at = blockIdx.x * DSV41_HEAD_PER; at < have; at += gridDim.x * DSV41_HEAD_PER) {
        const bool mine = at + group < have;
        unsigned int v = 0u;
        if (mine) {
            v = list[(size_t)row * n + at + group];
            const uint4* b_vec = (const uint4*)(head + (size_t)v * K);
            float acc = 0.0f;
            for (unsigned int kv = lane; kv < K / 8u; kv += DSV41_HEAD_LANES) {
                const uint4 a_data = a_vec[kv];
                const uint4 b_data = b_vec[kv];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                    *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                    *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                    acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                    acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
            if ((threadIdx.x & 31u) == 0u) halves[group * 2u + lane / 32u] = acc;
        }
        __syncthreads();
        if (mine && lane == 0u) {
            const float value = dsv41_bf16(halves[group * 2u] + halves[group * 2u + 1u]);
            // A logit that is not a number is never an argmax's pick.
            if (value == value && dsv41_candidate_before(value, first + v, best, best_token)) {
                best = value;
                best_token = first + v;
            }
        }
        __syncthreads();
    }
    if (lane == 0u) {
        values[group] = best;
        tokens[group] = best_token;
    }
    __syncthreads();
    if (threadIdx.x != 0u) return;
    for (unsigned int g = 1u; g < DSV41_HEAD_PER; ++g) {
        if (dsv41_candidate_before(values[g], tokens[g], best, best_token)) {
            best = values[g];
            best_token = tokens[g];
        }
    }
    unsigned int* out = found + ((size_t)row * gridDim.x + blockIdx.x) * 2u;
    out[0] = __float_as_uint(best);
    out[1] = best_token;
}

// What a row of logits begins with when the row was not computed, only its
// largest picked: four BF16 values that are not numbers, in this order, then
// the step that picked and its token. No head writes this, so whoever reads
// the row can tell, and a head that runs later writes over it.
#define DSV41_HEAD_MARK 0x7fc1u

// out[row] = the token of the row's largest logit, from every rank's
// blocks' best (found [parts, found_rows, blocks, 2]); token 0 where no
// rank kept any, as an argmax over a row of no numbers picks. With `mark`
// (rows of `mark_stride` BF16 values, the logits the step did not compute),
// each row's mark, `step` and token too. Grid 1, a thread a row.
extern "C" __global__ void dsv41_head_pick(
    const unsigned int* __restrict__ found,
    unsigned int* __restrict__ out,
    unsigned short* __restrict__ mark,
    unsigned int rows,
    unsigned int blocks,
    unsigned int parts,
    unsigned int found_rows,
    unsigned int mark_stride,
    unsigned int step) {
    atlas_pdl_enter();
    const unsigned int row = threadIdx.x;
    if (row >= rows) return;
    float best = 0.0f;
    unsigned int best_token = DSV41_NO_TOKEN;
    for (unsigned int p = 0; p < parts; ++p) {
        const unsigned int* of_row = found + ((size_t)p * found_rows + row) * blocks * 2u;
        for (unsigned int b = 0; b < blocks; ++b) {
            const float value = __uint_as_float(of_row[2u * b]);
            if (dsv41_candidate_before(value, of_row[2u * b + 1u], best, best_token)) {
                best = value;
                best_token = of_row[2u * b + 1u];
            }
        }
    }
    const unsigned int token = best_token == DSV41_NO_TOKEN ? 0u : best_token;
    out[row] = token;
    if (!mark) return;
    unsigned short* m = mark + (size_t)row * mark_stride;
    for (unsigned int i = 0; i < 4u; ++i) m[i] = (unsigned short)(DSV41_HEAD_MARK + i);
    m[4] = (unsigned short)step;
    m[5] = (unsigned short)(step >> 16);
    m[6] = (unsigned short)token;
    m[7] = (unsigned short)(token >> 16);
}
