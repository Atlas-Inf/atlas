// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash attention over gathered keys.
//
// A query attends to at most 128 window keys and 512 selected compressed
// keys, each one shared latent of 512 values that serves as key and value,
// plus a per-head sink that takes probability and contributes nothing.
//
// This follows the schedule of the reference kernel (`sparse_attn` in the
// release's `inference/kernel.py`), because the model's next step depends on
// it: the output is quantized to FP8 before the output projection, and that
// turns a 1e-3 difference in the attention into 1e-2 after it. The schedule:
// keys are taken 64 at a time in the order of the index row; a running
// maximum rescales what has been summed; the weights that multiply the keys
// are rounded to BF16, the ones summed for the denominator are not; an
// absent key (-1) scores -inf.
//
// Two forms with the same bytes.
//
// `dsv41_sparse_attn` is the plain one: a block per (query, head), in which
// a thread per key scores its key, the steps' maxima become running maxima,
// a thread per key takes its weight, and a thread per output value adds the
// keys in order.
//
// Every head of a query reads the same keys, and in that form each head
// reads all of them twice: 16 heads over 640 keys is 21 MB a token and
// layer, which is what it then spends its time on. The other form is three
// launches in which the heads share the reads:
//
//   dsv41_attn_logits    a block is 16 keys by 16 heads; the 16 threads of a
//                        key read it together.
//   dsv41_attn_schedule  a block per (query, head): the schedule above,
//                        from the logits to the weights.
//   dsv41_attn_output    a block is 16 output values by 16 heads; the 16
//                        threads of a value read each key together.
//
// A thread adds what the plain form's thread added, in its order: the 512
// products of a (head, key), and the keys of a (head, value).
//
// With `rope` the rotations on either side happen here as well: the last
// `rope` values of q are turned to the query's position before it scores
// anything, and those of the output turned back, as `dsv41_rotate` would
// have done in place before and after.
//
// A query's bytes depend only on its own q and on its index row: not on the
// other queries in the launch. Callers build the row the same way for a
// decode step and for a position inside a prefill chunk, so the two agree
// bit for bit.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_DIM 512
#define DSV41_MAX_KEYS 640
#define DSV41_KEYS_PER_STEP 64
#define DSV41_MAX_STEPS (DSV41_MAX_KEYS / DSV41_KEYS_PER_STEP)
#define DSV41_THREADS 640
// Heads a block of the shared form takes together, and the keys or output
// values it takes with them.
#define DSV41_GROUP 16
#define DSV41_GROUP_THREADS (DSV41_GROUP * DSV41_GROUP)
// What the shared form keeps of a (query, head) between its launches, FP32:
// a logit a key, which the schedule replaces by the key's weight; then what
// each step rescales the sum by; then the denominator.
#define DSV41_RECORD_SHRINK DSV41_MAX_KEYS
#define DSV41_RECORD_DENOMINATOR (DSV41_MAX_KEYS + DSV41_MAX_STEPS)
#define DSV41_RECORD 656

__device__ __forceinline__ unsigned int dsv41_steps(unsigned int k) {
    return (k + DSV41_KEYS_PER_STEP - 1) / DSV41_KEYS_PER_STEP;
}

// Key `j` of a query's index row `ids`: its row of 512 values, or none.
__device__ __forceinline__ const __nv_bfloat16* dsv41_key(
    const __nv_bfloat16* keys_a, const __nv_bfloat16* keys_b, const int* ids, unsigned int j,
    unsigned int k, unsigned int split) {
    const int id = j < k ? ids[j] : -1;
    return id < 0 ? nullptr : (j < split ? keys_a : keys_b) + (size_t)id * DSV41_DIM;
}

// A (query, head)'s logit for one key: q . key in value order, or -inf.
__device__ __forceinline__ float dsv41_logit(
    const float* qs, const __nv_bfloat16* key, float scale) {
    if (key == nullptr) return -INFINITY;
    float s = 0.0f;
    for (unsigned int d = 0; d < DSV41_DIM; ++d) s += qs[d] * __bfloat162float(key[d]);
    return s * scale;
}

// A (query, head)'s output value `d`: the keys in order, rescaled at each
// step, over the denominator.
__device__ __forceinline__ float dsv41_attended(
    const __nv_bfloat16* const* row, const float* weight, const float* shrink, float denominator,
    unsigned int steps, unsigned int d) {
    float a = 0.0f;
    for (unsigned int u = 0; u < steps; ++u) {
        a *= shrink[u];
        for (unsigned int j = u * DSV41_KEYS_PER_STEP; j < (u + 1) * DSV41_KEYS_PER_STEP; ++j)
            if (row[j] != nullptr) a += weight[j] * __bfloat162float(row[j][d]);
    }
    return a / denominator;
}

// The schedule of one (query, head), by a block of 640 threads over shared
// arrays: `logit` [steps * 64] in, and out as exp(logit - its step's running
// maximum); `weight` the same rounded to BF16, which is what multiplies a
// key; `shrink` [steps] what each step rescales the sum so far by;
// `denominator`. `step_top` and `step_sum` are scratch. Ends on a barrier.
__device__ __forceinline__ void dsv41_schedule(
    float* logit, float* weight, float* step_top, float* step_sum, float* shrink,
    float* denominator, unsigned int steps, float sink) {
    const unsigned int t = threadIdx.x;
    // A thread per step: the largest logit of the step.
    if (t < steps) {
        float top = -INFINITY;
        for (unsigned int j = 0; j < DSV41_KEYS_PER_STEP; ++j)
            top = fmaxf(top, logit[t * DSV41_KEYS_PER_STEP + j]);
        step_top[t] = top;
    }
    __syncthreads();
    // The running maximum up to and including each step, from the -1e30 the
    // schedule starts at, and what it rescales the sum so far by.
    if (t < steps) {
        float before = -1e30f;
        for (unsigned int u = 0; u < t; ++u) before = fmaxf(before, step_top[u]);
        const float top = fmaxf(before, step_top[t]);
        shrink[t] = expf(before - top);
        // Kept aside until every thread has read `step_top`.
        step_sum[t] = top;
    }
    __syncthreads();
    if (t < steps) step_top[t] = step_sum[t];
    __syncthreads();
    if (t < steps * DSV41_KEYS_PER_STEP) {
        const float e = expf(logit[t] - step_top[t / DSV41_KEYS_PER_STEP]);
        logit[t] = e;
        weight[t] = __bfloat162float(__float2bfloat16(e));
    }
    __syncthreads();
    // A thread per step: the step's weights added in key order, unrounded.
    if (t < steps) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < DSV41_KEYS_PER_STEP; ++j)
            sum += logit[t * DSV41_KEYS_PER_STEP + j];
        step_sum[t] = sum;
    }
    __syncthreads();
    if (t == 0) {
        float total = 0.0f;
        for (unsigned int u = 0; u < steps; ++u) total = total * shrink[u] + step_sum[u];
        const float peak = steps == 0 ? -1e30f : step_top[steps - 1];
        *denominator = total + expf(sink - peak);
    }
    __syncthreads();
}

// The plain form. Grid (m, h), block 640.
//
// q      [m, h, 512] BF16
// keys_a rows of 512 BF16, read by index columns [0, split)
// keys_b rows of 512 BF16, read by index columns [split, k)
// idx    [m, k] row numbers, -1 for none; k <= 640
// sink   [h] FP32 logits
// out    [m, h, 512] BF16
extern "C" __global__ void __launch_bounds__(DSV41_THREADS) dsv41_sparse_attn(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys_a,
    const __nv_bfloat16* __restrict__ keys_b,
    const int* __restrict__ idx,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int k,
    unsigned int split,
    float scale) {
    atlas_pdl_enter();
    const unsigned int query = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
    const unsigned int steps = dsv41_steps(k);
    __shared__ float qs[DSV41_DIM];
    __shared__ const __nv_bfloat16* row[DSV41_MAX_KEYS];
    __shared__ float logit[DSV41_MAX_KEYS];
    __shared__ float weight[DSV41_MAX_KEYS];
    __shared__ float step_top[DSV41_MAX_STEPS];
    __shared__ float step_sum[DSV41_MAX_STEPS];
    __shared__ float shrink[DSV41_MAX_STEPS];
    __shared__ float denominator;

    const __nv_bfloat16* qv = q + ((size_t)query * h + head) * DSV41_DIM;
    if (t < DSV41_DIM) qs[t] = __bfloat162float(qv[t]);
    __syncthreads();
    if (t < steps * DSV41_KEYS_PER_STEP) {
        row[t] = dsv41_key(keys_a, keys_b, idx + (size_t)query * k, t, k, split);
        logit[t] = dsv41_logit(qs, row[t], scale);
    }
    __syncthreads();
    dsv41_schedule(logit, weight, step_top, step_sum, shrink, &denominator, steps, sink[head]);
    if (t >= DSV41_DIM) return;
    out[((size_t)query * h + head) * DSV41_DIM + t] =
        __float2bfloat16(dsv41_attended(row, weight, shrink, denominator, steps, t));
}

// The shared form, first launch: records [m, h, DSV41_RECORD] FP32, a logit
// a key. With rope > 0, q is turned to positions [m] (u32) by freqs
// [rope / 2] (FP32 radians a position) first. h is a multiple of 16.
// Grid (4 * steps, m * h / 16), block 256: thread (key, head).
extern "C" __global__ void __launch_bounds__(DSV41_GROUP_THREADS) dsv41_attn_logits(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys_a,
    const __nv_bfloat16* __restrict__ keys_b,
    const int* __restrict__ idx,
    float* __restrict__ records,
    unsigned int h,
    unsigned int k,
    unsigned int split,
    float scale,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rope) {
    atlas_pdl_enter();
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int query = blockIdx.y / groups, first = (blockIdx.y % groups) * DSV41_GROUP;
    const unsigned int t = threadIdx.x, in_group = t % DSV41_GROUP;
    const unsigned int j = blockIdx.x * DSV41_GROUP + t / DSV41_GROUP;
    __shared__ float qs[DSV41_GROUP][DSV41_DIM];

    const __nv_bfloat16* qv = q + ((size_t)query * h + first) * DSV41_DIM;
    for (unsigned int i = t; i < DSV41_GROUP * DSV41_DIM; i += DSV41_GROUP_THREADS)
        qs[i / DSV41_DIM][i % DSV41_DIM] = __bfloat162float(qv[i]);
    __syncthreads();
    // A thread per rotary pair of a head, as many of them as it takes.
    const unsigned int pairs = rope / 2, turned = DSV41_DIM - rope;
    for (unsigned int i = t; i < DSV41_GROUP * pairs; i += DSV41_GROUP_THREADS) {
        float* pair = qs[i / pairs] + turned + 2 * (i % pairs);
        dsv41_rotate_pair(pair[0], pair[1], (float)positions[query] * freqs[i % pairs], false,
                          pair[0], pair[1]);
    }
    __syncthreads();
    if (j >= dsv41_steps(k) * DSV41_KEYS_PER_STEP) return;
    const __nv_bfloat16* key = dsv41_key(keys_a, keys_b, idx + (size_t)query * k, j, k, split);
    records[((size_t)query * h + first + in_group) * DSV41_RECORD + j] =
        dsv41_logit(qs[in_group], key, scale);
}

// The shared form, second launch: each record's logits replaced by the
// keys' weights, and its shrinks and denominator filled in. sink [h] FP32.
// Grid (m, h), block 640.
extern "C" __global__ void __launch_bounds__(DSV41_THREADS) dsv41_attn_schedule(
    float* __restrict__ records,
    const float* __restrict__ sink,
    unsigned int h,
    unsigned int k) {
    atlas_pdl_enter();
    const unsigned int query = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
    const unsigned int steps = dsv41_steps(k);
    float* record = records + ((size_t)query * h + head) * DSV41_RECORD;
    __shared__ float logit[DSV41_MAX_KEYS];
    __shared__ float weight[DSV41_MAX_KEYS];
    __shared__ float step_top[DSV41_MAX_STEPS];
    __shared__ float step_sum[DSV41_MAX_STEPS];
    __shared__ float shrink[DSV41_MAX_STEPS];
    __shared__ float denominator;
    if (t < steps * DSV41_KEYS_PER_STEP) logit[t] = record[t];
    __syncthreads();
    dsv41_schedule(logit, weight, step_top, step_sum, shrink, &denominator, steps, sink[head]);
    if (t < steps * DSV41_KEYS_PER_STEP) record[t] = weight[t];
    if (t < steps) record[DSV41_RECORD_SHRINK + t] = shrink[t];
    if (t == 0) record[DSV41_RECORD_DENOMINATOR] = denominator;
}

// The shared form, third launch: out [m, h, 512] BF16 from the records and
// the keys, its last `rope` values turned back. rope is a multiple of 32.
// Grid (32, m * h / 16), block 256: thread (output value, head).
extern "C" __global__ void __launch_bounds__(DSV41_GROUP_THREADS) dsv41_attn_output(
    const __nv_bfloat16* __restrict__ keys_a,
    const __nv_bfloat16* __restrict__ keys_b,
    const int* __restrict__ idx,
    const float* __restrict__ records,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int k,
    unsigned int split,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rope) {
    atlas_pdl_enter();
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int query = blockIdx.y / groups, first = (blockIdx.y % groups) * DSV41_GROUP;
    const unsigned int t = threadIdx.x, head = first + t % DSV41_GROUP;
    const unsigned int d = blockIdx.x * DSV41_GROUP + t / DSV41_GROUP;
    const unsigned int steps = dsv41_steps(k);
    __shared__ const __nv_bfloat16* row[DSV41_MAX_KEYS];
    __shared__ float value[DSV41_GROUP_THREADS];

    for (unsigned int j = t; j < steps * DSV41_KEYS_PER_STEP; j += DSV41_GROUP_THREADS)
        row[j] = dsv41_key(keys_a, keys_b, idx + (size_t)query * k, j, k, split);
    __syncthreads();
    const float* record = records + ((size_t)query * h + head) * DSV41_RECORD;
    const __nv_bfloat16 mine = __float2bfloat16(dsv41_attended(
        row, record, record + DSV41_RECORD_SHRINK, record[DSV41_RECORD_DENOMINATOR], steps, d));
    __nv_bfloat16* o = out + ((size_t)query * h + head) * DSV41_DIM;
    // A block's 16 values are all turned or none of them.
    const unsigned int turned = DSV41_DIM - rope;
    if (blockIdx.x * DSV41_GROUP < turned) {
        o[d] = mine;
        return;
    }
    value[t] = __bfloat162float(mine);
    __syncthreads();
    // The thread of a pair's first value turns the pair; its second value is
    // the same head's next.
    if ((d - turned) % 2 != 0) return;
    float re, im;
    dsv41_rotate_pair(value[t], value[t + DSV41_GROUP],
                      (float)positions[query] * freqs[(d - turned) / 2], true, re, im);
    o[d] = __float2bfloat16(re);
    o[d + 1] = __float2bfloat16(im);
}
