// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash, the draft head's own steps (`DSparkBlock` and
// `forward_spec` in the release's `inference/model.py`). A draft is a guess
// that the model then verifies, so these follow the reference's arithmetic
// without holding to its last bit: a draft that differs is a draft that is
// refused, never a wrong token.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_TOP_THREADS 1024

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
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)P * part_n;
    if (i >= rows * n) return;
    const size_t r = i / n, at = i % n, p = at / part_n, j = at % part_n;
    out[i] = parts[(p * rows + r) * part_n + j];
}

// The `keep` largest of each row of logits and where they are: best[r, j] =
// (the value's bits, first + its column), largest first, the lower column
// first among equals. logits [rows, n] FP32, a block a row. A round a value:
// every thread takes the largest of its columns that comes after the last
// one kept, and the block the largest of those.
extern "C" __global__ void __launch_bounds__(DSV41_TOP_THREADS) dsv41_draft_top(
    const float* __restrict__ logits,
    unsigned int* __restrict__ best,
    unsigned int n,
    unsigned int first,
    unsigned int keep) {
    __shared__ float values[DSV41_TOP_THREADS];
    __shared__ unsigned int columns[DSV41_TOP_THREADS];
    const float* row = logits + (size_t)blockIdx.x * n;
    const unsigned int none = 0xffffffffu;
    float last = 0.0f;
    unsigned int last_at = none;
    for (unsigned int j = 0; j < keep; ++j) {
        float value = -INFINITY;
        unsigned int at = none;
        for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
            const float x = row[i];
            const bool after = last_at == none || x < last || (x == last && i > last_at);
            if (after && (at == none || x > value)) {
                value = x;
                at = i;
            }
        }
        values[threadIdx.x] = value;
        columns[threadIdx.x] = at;
        __syncthreads();
        for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
            if (threadIdx.x < step) {
                const float other = values[threadIdx.x + step];
                const unsigned int other_at = columns[threadIdx.x + step];
                const float mine = values[threadIdx.x];
                const unsigned int mine_at = columns[threadIdx.x];
                const bool better = other_at != none &&
                    (mine_at == none || other > mine || (other == mine && other_at < mine_at));
                if (better) {
                    values[threadIdx.x] = other;
                    columns[threadIdx.x] = other_at;
                }
            }
            __syncthreads();
        }
        last = values[0];
        last_at = columns[0];
        if (threadIdx.x == 0) {
            unsigned int* to = best + ((size_t)blockIdx.x * keep + j) * 2;
            to[0] = __float_as_uint(last);
            to[1] = last_at == none ? none : first + last_at;
        }
        // Every thread has read the round's winner before the next one's
        // candidates replace it.
        __syncthreads();
        if (last_at != none) continue;
        // Nothing comes after it: the rest of the row is none.
        if (threadIdx.x == 0)
            for (unsigned int rest = j + 1; rest < keep; ++rest)
                best[((size_t)blockIdx.x * keep + rest) * 2 + 1] = none;
        return;
    }
}
