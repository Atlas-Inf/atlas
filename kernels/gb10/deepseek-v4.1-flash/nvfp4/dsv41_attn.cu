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
// One block per (query, head), built for latency: every stage that the
// schedule does not order runs at once. A thread per key scores its key;
// the steps' maxima become running maxima in one pass; a thread per key
// takes its weight; and then a thread per output value adds the keys in
// order, which is the only long chain (640 steps). The sums are the ones
// the step-by-step form made, in the same order, so the bytes are the same.
//
// A query's bytes depend only on its own q and on its index row: not on the
// other queries in the launch. Callers build the row the same way for a
// decode step and for a position inside a prefill chunk, so the two agree
// bit for bit.

#include <cuda_bf16.h>

#define DSV41_DIM 512
#define DSV41_MAX_KEYS 640
#define DSV41_KEYS_PER_STEP 64
#define DSV41_MAX_STEPS (DSV41_MAX_KEYS / DSV41_KEYS_PER_STEP)
#define DSV41_THREADS 640

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
    const unsigned int query = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
    const unsigned int steps = (k + DSV41_KEYS_PER_STEP - 1) / DSV41_KEYS_PER_STEP;
    __shared__ float qs[DSV41_DIM];
    __shared__ const __nv_bfloat16* row[DSV41_MAX_KEYS];
    // A key's logit, then exp(logit - its step's running maximum).
    __shared__ float logit[DSV41_MAX_KEYS];
    // The same, rounded to BF16: what multiplies the key.
    __shared__ float weight[DSV41_MAX_KEYS];
    __shared__ float step_top[DSV41_MAX_STEPS];
    __shared__ float step_sum[DSV41_MAX_STEPS];
    __shared__ float shrink[DSV41_MAX_STEPS];
    __shared__ float denominator;

    const __nv_bfloat16* qv = q + ((size_t)query * h + head) * DSV41_DIM;
    if (t < DSV41_DIM) qs[t] = __bfloat162float(qv[t]);
    __syncthreads();

    // A thread per key: its row, and its logit, summed in value order.
    const int* ids = idx + (size_t)query * k;
    if (t < steps * DSV41_KEYS_PER_STEP) {
        const int id = t < k ? ids[t] : -1;
        const __nv_bfloat16* key =
            id < 0 ? nullptr : (t < split ? keys_a : keys_b) + (size_t)id * DSV41_DIM;
        float s = -INFINITY;
        if (key != nullptr) {
            s = 0.0f;
            for (unsigned int d = 0; d < DSV41_DIM; ++d) s += qs[d] * __bfloat162float(key[d]);
            s *= scale;
        }
        row[t] = key;
        logit[t] = s;
    }
    __syncthreads();

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
        // Kept in the upper half until every thread has read `step_top`.
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
        denominator = total + expf(sink[head] - peak);
    }
    __syncthreads();

    // A thread per output value: the keys in order, rescaled at each step.
    if (t >= DSV41_DIM) return;
    float a = 0.0f;
    for (unsigned int u = 0; u < steps; ++u) {
        a *= shrink[u];
        for (unsigned int j = u * DSV41_KEYS_PER_STEP; j < (u + 1) * DSV41_KEYS_PER_STEP; ++j)
            if (row[j] != nullptr) a += weight[j] * __bfloat162float(row[j][t]);
    }
    out[((size_t)query * h + head) * DSV41_DIM + t] = __float2bfloat16(a / denominator);
}
