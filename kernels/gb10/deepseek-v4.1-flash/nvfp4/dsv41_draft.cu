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
#define DSV41_MARKOV_WARPS (DSV41_MARKOV_THREADS / 32)

// The warp's best of the candidates its lanes hold, in every lane.
__device__ __forceinline__ void dsv41_markov_warp_best(float& value, unsigned int& token) {
    #pragma unroll
    for (unsigned int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, value, o);
        const unsigned int other_token = __shfl_xor_sync(0xffffffffu, token, o);
        if (dsv41_candidate_before(other, other_token, value, token)) {
            value = other;
            token = other_token;
        }
    }
}

// The block's best of the candidates its warps hold, each the same in all
// of a warp's lanes: in every thread. `values` and `tokens` are shared, an
// entry a warp.
__device__ __forceinline__ void dsv41_markov_block_best(
    float& value, unsigned int& token, float* values, unsigned int* tokens) {
    const unsigned int lane = threadIdx.x & 31u;
    if (lane == 0) {
        values[threadIdx.x >> 5] = value;
        tokens[threadIdx.x >> 5] = token;
    }
    __syncthreads();
    value = lane < DSV41_MARKOV_WARPS ? values[lane] : 0.0f;
    token = lane < DSV41_MARKOV_WARPS ? tokens[lane] : DSV41_NO_TOKEN;
    dsv41_markov_warp_best(value, token);
}

// Eight BF16 values of a row, as a lane holds them.
__device__ __forceinline__ void dsv41_markov_eight(const __nv_bfloat16* at, float* v) {
    const uint4 w = *reinterpret_cast<const uint4*>(at);
    dsv41_unpack2(w.x, v[0], v[1]);
    dsv41_unpack2(w.y, v[2], v[3]);
    dsv41_unpack2(w.z, v[4], v[5]);
    dsv41_unpack2(w.w, v[6], v[7]);
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
// token `first`), embed [vocab, rank] BF16; rank a multiple of 8, at most
// 256. Grid ceil(n / 256), block 256: a block takes 256 tokens, a warp 32 of
// them one after another, and a lane eight values of each one's row, so a
// warp reads a row in one load of whole cache lines and the row before's
// eight values stay in a lane's registers. (A thread a token read 32 rows a
// load, a value each: 97 us a round of the release's head against this
// form's time, with the head in the cache.)
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
    __shared__ float was[DSV41_MARKOV_WARPS], values[DSV41_MARKOV_WARPS];
    __shared__ unsigned int was_tokens[DSV41_MARKOV_WARPS], tokens[DSV41_MARKOV_WARPS];
    const unsigned int t = threadIdx.x, lane = t & 31u;
    unsigned int last = token;
    if (before) {
        float best = 0.0f;
        unsigned int best_token = DSV41_NO_TOKEN;
        for (unsigned int i = t; i < count; i += DSV41_MARKOV_THREADS) {
            const float value = __uint_as_float(before[2 * i]);
            if (dsv41_candidate_before(value, before[2 * i + 1], best, best_token)) {
                best = value;
                best_token = before[2 * i + 1];
            }
        }
        dsv41_markov_warp_best(best, best_token);
        dsv41_markov_block_best(best, best_token, was, was_tokens);
        last = best_token;
    }
    if (chosen && blockIdx.x == 0 && t == 0) *chosen = last;
    if (!logits) return;
    // A row that found no token ends the chain for the host; the rows after
    // it still run, from any token.
    const bool live = 8u * lane < rank;
    float from[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (live) dsv41_markov_eight(embed + (size_t)(last < vocab ? last : 0u) * rank + 8u * lane, from);
    float value = 0.0f;
    unsigned int mine_token = DSV41_NO_TOKEN;
    const unsigned int base = blockIdx.x * DSV41_MARKOV_THREADS + (t & ~31u);
    for (unsigned int v = base; v < base + 32u && v < n; ++v) {
        float gain = 0.0f;
        if (live) {
            float to[8];
            dsv41_markov_eight(head + (size_t)v * rank + 8u * lane, to);
            #pragma unroll
            for (unsigned int k = 0; k < 8; ++k) gain += from[k] * to[k];
        }
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) gain += __shfl_xor_sync(0xffffffffu, gain, o);
        const float now = logits[v] + gain;
        // A logit that is not a number is no candidate.
        if (now == now && dsv41_candidate_before(now, first + v, value, mine_token)) {
            value = now;
            mine_token = first + v;
        }
    }
    dsv41_markov_block_best(value, mine_token, values, tokens);
    if (t == 0) {
        mine[2 * blockIdx.x] = __float_as_uint(value);
        mine[2 * blockIdx.x + 1] = mine_token;
    }
}
