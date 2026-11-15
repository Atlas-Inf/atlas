// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash attention over gathered keys.
//
// A query attends to at most 128 window keys and 512 selected compressed
// keys, each one shared latent of 512 values that serves as key and value,
// plus a per-head sink that takes probability and contributes nothing. The
// reference is `sparse_attn` in the release's `inference/kernel.py`.
//
// One block per (query, head). Every sum that reaches the output runs in the
// order of the index row, and an absent key (-1) adds an exact zero, so a
// query's bytes depend only on its own q and on its keys in order: not on the
// other queries in the launch, and not on where the row's padding sits. A
// decode step and the same position inside a prefill chunk agree bit for bit.
//
// The softmax is exact in FP32. The reference rounds its weights to BF16,
// 64 keys at a time; that leaves it about 1.3e-3 from this kernel
// (sparkseek/results/2026-11-15-reference-layers).

#include <cuda_bf16.h>

#define DSV41_DIM 512
#define DSV41_MAX_KEYS 640
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
    __shared__ float weight[DSV41_MAX_KEYS];
    __shared__ const __nv_bfloat16* row[DSV41_MAX_KEYS];
    __shared__ float top[DSV41_THREADS];
    __shared__ float total;

    const __nv_bfloat16* qv = q + ((size_t)query * h + head) * DSV41_DIM;
    for (unsigned int d = t; d < DSV41_DIM; d += DSV41_THREADS) qs[d] = __bfloat162float(qv[d]);
    const int* ids = idx + (size_t)query * k;
    for (unsigned int c = t; c < k; c += DSV41_THREADS) {
        const int id = ids[c];
        row[c] = id < 0 ? nullptr : (c < split ? keys_a : keys_b) + (size_t)id * DSV41_DIM;
    }
    __syncthreads();

    // Logits, one key per thread at a time, each a sum in dimension order.
    float mine = -INFINITY;
    for (unsigned int c = t; c < k; c += DSV41_THREADS) {
        float s = -INFINITY;
        if (row[c] != nullptr) {
            s = 0.0f;
            for (unsigned int d = 0; d < DSV41_DIM; ++d) s += qs[d] * __bfloat162float(row[c][d]);
            s *= scale;
        }
        weight[c] = s;
        mine = fmaxf(mine, s);
    }
    top[t] = mine;
    __syncthreads();
    for (unsigned int width = DSV41_THREADS / 2; width > 0; width >>= 1) {
        if (t < width) top[t] = fmaxf(top[t], top[t + width]);
        __syncthreads();
    }
    const float peak = fmaxf(top[0], sink[head]);

    for (unsigned int c = t; c < k; c += DSV41_THREADS) weight[c] = expf(weight[c] - peak);
    __syncthreads();
    if (t == 0) {
        float sum = 0.0f;
        for (unsigned int c = 0; c < k; ++c) sum += weight[c];
        total = sum + expf(sink[head] - peak);
    }
    __syncthreads();

    __nv_bfloat16* ov = out + ((size_t)query * h + head) * DSV41_DIM;
    for (unsigned int d = t; d < DSV41_DIM; d += DSV41_THREADS) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < k; ++c)
            if (row[c] != nullptr) acc += weight[c] * __bfloat162float(row[c][d]);
        ov[d] = __float2bfloat16(acc / total);
    }
}
