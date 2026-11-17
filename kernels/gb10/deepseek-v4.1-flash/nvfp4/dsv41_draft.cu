// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash, the draft head's own steps (`DSparkBlock` and
// `forward_spec` in the release's `inference/model.py`). A draft is a guess
// that the model then verifies, so these follow the reference's arithmetic
// without holding to its last bit: a draft that differs is a draft that is
// refused, never a wrong token.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

// out[t, d] = BF16(mean over the hc copies of streams[t, j, d]): what the
// draft head reads of the main model at a layer, a row a token. streams [T,
// hc, H] BF16; out rows are `out_stride` values apart (a token's three layers
// lie side by side in one row of the head's input). One thread a value.
extern "C" __global__ void dsv41_stream_mean(
    const __nv_bfloat16* __restrict__ streams,
    __nv_bfloat16* __restrict__ out,
    unsigned int H,
    unsigned int hc,
    unsigned int out_stride,
    unsigned int tokens) {
    atlas_pdl_enter();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)tokens * H) return;
    const size_t t = i / H, d = i % H;
    float sum = 0.0f;
    for (unsigned int j = 0; j < hc; ++j)
        sum += __bfloat162float(streams[(t * hc + j) * H + d]);
    out[t * out_stride + d] = __float2bfloat16(sum / (float)hc);
}

// A projection the ranks computed in parts, a row at a time: out[r, p *
// part_n + j] = parts[p, r, j]. parts [P, rows, part_n] BF16 as the ranks
// gather them, out [rows, P * part_n]. One thread a value.
extern "C" __global__ void dsv41_parts_rows(
    const __nv_bfloat16* __restrict__ parts,
    __nv_bfloat16* __restrict__ out,
    unsigned int rows,
    unsigned int part_n,
    unsigned int P) {
    atlas_pdl_enter();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)P * part_n;
    if (i >= rows * n) return;
    const size_t r = i / n, at = i % n, p = at / part_n, j = at % part_n;
    out[i] = parts[(p * rows + r) * part_n + j];
}

#define DSV41_MARKOV_THREADS 256
#define DSV41_MARKOV_NONE 0xffffffffu

// Whether candidate (value, token) comes before (best, best_token): the
// larger value, and among equals the lower token. Every rank orders the same
// candidates the same way.
__device__ __forceinline__ bool dsv41_markov_before(
    float value, unsigned int token, float best, unsigned int best_token) {
    return token != DSV41_MARKOV_NONE &&
           (best_token == DSV41_MARKOV_NONE || value > best || (value == best && token < best_token));
}

// The block's best of the candidates its threads hold, in entry 0 of
// `values` and `tokens` (shared, a thread an entry).
__device__ __forceinline__ void dsv41_markov_best(float* values, unsigned int* tokens) {
    for (unsigned int step = DSV41_MARKOV_THREADS / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            const float other = values[threadIdx.x + step];
            const unsigned int other_token = tokens[threadIdx.x + step];
            if (dsv41_markov_before(other, other_token, values[threadIdx.x], tokens[threadIdx.x])) {
                values[threadIdx.x] = other;
                tokens[threadIdx.x] = other_token;
            }
        }
        __syncthreads();
    }
}

// One row of a draft block through the Markov head, on a rank's share of the
// vocabulary (`DSparkBlock.forward_head`): a token's logit gains what the
// Markov head gives it after the token before, head[v] . embed[before], and
// the row's draft is the largest. The token before is the block's first
// token for the first row, and for the others the draft of the row before,
// which the ranks agree on from what each found in its share: `before`
// holds every rank's candidates of that row, `count` of them, as (value
// bits, token). A block leaves its own best of this row in mine[blockIdx.x]
// the same way, for the ranks to gather; and block 0 leaves the draft of the
// row before in `chosen`. With no `logits` the launch only does that: it
// closes the chain.
//
// logits [n] FP32, head [n, rank] BF16 (this rank's rows, the first of them
// token `first`), embed [vocab, rank] BF16. Grid ceil(n / 256), block 256,
// a thread a token; rank at most 256.
extern "C" __global__ void __launch_bounds__(DSV41_MARKOV_THREADS) dsv41_markov_round(
    const float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ head,
    const __nv_bfloat16* __restrict__ embed,
    const unsigned int* __restrict__ before,
    unsigned int* __restrict__ mine,
    unsigned int* __restrict__ chosen,
    unsigned int n,
    unsigned int rank,
    unsigned int first,
    unsigned int count,
    unsigned int token,
    unsigned int vocab) {
    atlas_pdl_enter();
    __shared__ float values[DSV41_MARKOV_THREADS];
    __shared__ unsigned int tokens[DSV41_MARKOV_THREADS];
    __shared__ float from[DSV41_MARKOV_THREADS];
    const unsigned int t = threadIdx.x;
    unsigned int last = token;
    if (before) {
        float best = 0.0f;
        unsigned int best_token = DSV41_MARKOV_NONE;
        for (unsigned int i = t; i < count; i += DSV41_MARKOV_THREADS) {
            const float value = __uint_as_float(before[2 * i]);
            if (dsv41_markov_before(value, before[2 * i + 1], best, best_token)) {
                best = value;
                best_token = before[2 * i + 1];
            }
        }
        values[t] = best;
        tokens[t] = best_token;
        __syncthreads();
        dsv41_markov_best(values, tokens);
        last = tokens[0];
        __syncthreads();
    }
    if (chosen && blockIdx.x == 0 && t == 0) *chosen = last;
    if (!logits) return;
    // A row that found no token ends the chain for the host; the rows after
    // it still run, from any token.
    const __nv_bfloat16* row = embed + (size_t)(last < vocab ? last : 0u) * rank;
    for (unsigned int k = t; k < rank; k += DSV41_MARKOV_THREADS) from[k] = __bfloat162float(row[k]);
    __syncthreads();
    const unsigned int v = blockIdx.x * DSV41_MARKOV_THREADS + t;
    float value = 0.0f;
    unsigned int mine_token = DSV41_MARKOV_NONE;
    if (v < n) {
        const __nv_bfloat16* to = head + (size_t)v * rank;
        float gain = 0.0f;
        for (unsigned int k = 0; k < rank; ++k) gain += from[k] * __bfloat162float(to[k]);
        value = logits[v] + gain;
        // A logit that is not a number is no candidate.
        if (value == value) mine_token = first + v;
    }
    values[t] = value;
    tokens[t] = mine_token;
    __syncthreads();
    dsv41_markov_best(values, tokens);
    if (t == 0) {
        mine[2 * blockIdx.x] = __float_as_uint(values[0]);
        mine[2 * blockIdx.x + 1] = tokens[0];
    }
}
