// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash, the indexer's scores (`Indexer.forward` in the
// release's `inference/model.py`): a small side attention that scores every
// compressed position a query can see, so the query keeps the best
// `index_topk` of them.
//
//   score[t] = BF16(sum_h BF16(relu(BF16(q_h . key_t)) * w_h))
//
// q is `heads` FP4-rounded query heads, key_t one FP4-rounded index key per
// compressed position, w_h a weight per head from the attention's input.
// Every product of two FP4 values is exact, so the dot product has one value
// whatever order it is summed in; this sums it in FP64 and rounds once. The
// reference's prefill computes exactly this. (Its decode path does not when
// the number of keys is odd: cuBLAS then takes a BF16 path that is a few
// steps off. That is its library's rounding, not the model's.)
//
// The kernel leaves the sum over heads in FP32, so ranks that each hold some
// of the heads can add their parts before the one rounding to BF16.
//
// Plain form: one thread per compressed position.

#include <cuda_bf16.h>

#define DSV41_INDEX_THREADS 256
#define DSV41_INDEX_MAX_Q 4096
#define DSV41_INDEX_MAX_HEADS 64

static __device__ __forceinline__ float dsv41_index_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// q [heads, width] BF16, keys [groups, width] BF16, weights [heads] BF16,
// partial [groups] FP32. A head's weight is BF16(weights[h] * weight_scale).
extern "C" __global__ void __launch_bounds__(DSV41_INDEX_THREADS) dsv41_index_score(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ weights,
    float* __restrict__ partial,
    unsigned int heads,
    unsigned int width,
    unsigned int groups,
    float weight_scale) {
    __shared__ float qs[DSV41_INDEX_MAX_Q];
    __shared__ float ws[DSV41_INDEX_MAX_HEADS];
    for (unsigned int i = threadIdx.x; i < heads * width; i += DSV41_INDEX_THREADS)
        qs[i] = __bfloat162float(q[i]);
    for (unsigned int h = threadIdx.x; h < heads; h += DSV41_INDEX_THREADS)
        ws[h] = dsv41_index_bf16(__bfloat162float(weights[h]) * weight_scale);
    __syncthreads();
    const unsigned int t = blockIdx.x * DSV41_INDEX_THREADS + threadIdx.x;
    if (t >= groups) return;
    const __nv_bfloat16* key = keys + (size_t)t * width;
    float total = 0.0f;
    for (unsigned int h = 0; h < heads; ++h) {
        double dot = 0.0;
        for (unsigned int d = 0; d < width; ++d)
            dot += (double)qs[h * width + d] * (double)__bfloat162float(key[d]);
        const float head = fmaxf(dsv41_index_bf16((float)dot), 0.0f);
        total += dsv41_index_bf16(head * ws[h]);
    }
    partial[t] = total;
}
