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
// One block per (query, head). A query's bytes depend only on its own q and
// on its index row: not on the other queries in the launch. Callers build
// the row the same way for a decode step and for a position inside a prefill
// chunk, so the two agree bit for bit.

#include <cuda_bf16.h>

#define DSV41_DIM 512
#define DSV41_MAX_KEYS 640
#define DSV41_KEYS_PER_STEP 64
#define DSV41_THREADS 256

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
    __shared__ float qs[DSV41_DIM];
    __shared__ const __nv_bfloat16* row[DSV41_KEYS_PER_STEP];
    __shared__ float logit[DSV41_KEYS_PER_STEP];
    __shared__ float weight[DSV41_KEYS_PER_STEP];
    __shared__ float shrink;
    __shared__ float peak;
    __shared__ float total;

    const __nv_bfloat16* qv = q + ((size_t)query * h + head) * DSV41_DIM;
    for (unsigned int d = t; d < DSV41_DIM; d += DSV41_THREADS) qs[d] = __bfloat162float(qv[d]);
    if (t == 0) {
        peak = -1e30f;
        total = 0.0f;
    }
    const int* ids = idx + (size_t)query * k;
    // Each thread carries two output values across the steps.
    float acc[DSV41_DIM / DSV41_THREADS] = {0.0f, 0.0f};
    __syncthreads();

    for (unsigned int first = 0; first < k; first += DSV41_KEYS_PER_STEP) {
        if (t < DSV41_KEYS_PER_STEP) {
            const unsigned int c = first + t;
            const int id = c < k ? ids[c] : -1;
            const __nv_bfloat16* key =
                id < 0 ? nullptr : (c < split ? keys_a : keys_b) + (size_t)id * DSV41_DIM;
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
        if (t == 0) {
            float top = peak;
            for (unsigned int j = 0; j < DSV41_KEYS_PER_STEP; ++j) top = fmaxf(top, logit[j]);
            shrink = expf(peak - top);
            peak = top;
            float sum = 0.0f;
            for (unsigned int j = 0; j < DSV41_KEYS_PER_STEP; ++j) {
                const float e = expf(logit[j] - top);
                sum += e;
                weight[j] = __bfloat162float(__float2bfloat16(e));
            }
            total = total * shrink + sum;
        }
        __syncthreads();
        for (unsigned int i = 0; i < DSV41_DIM / DSV41_THREADS; ++i) {
            const unsigned int d = t + i * DSV41_THREADS;
            float a = acc[i] * shrink;
            for (unsigned int j = 0; j < DSV41_KEYS_PER_STEP; ++j)
                if (row[j] != nullptr) a += weight[j] * __bfloat162float(row[j][d]);
            acc[i] = a;
        }
        __syncthreads();
    }

    const float denominator = total + expf(sink[head] - peak);
    __nv_bfloat16* ov = out + ((size_t)query * h + head) * DSV41_DIM;
    for (unsigned int i = 0; i < DSV41_DIM / DSV41_THREADS; ++i)
        ov[t + i * DSV41_THREADS] = __float2bfloat16(acc[i] / denominator);
}
