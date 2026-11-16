// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash MoE, as the release's `inference/model.py` computes it
// (`Gate`, `Expert`, `MoE`):
//
//   scores  = sqrt(softplus(x . gate^T))                    FP32, x not quantized
//   chosen  = the top 6 of (scores + bias)
//   weights = scores[chosen] / (their sum + 1e-20) * route_scale
//   expert  = w2(fp8(BF16(weight * silu(min(g, 10)) * clamp(u, -10, 10))))
//             with g = BF16(w1(fp8(x))), u = BF16(w3(fp8(x)))
//   y       = BF16(sum over chosen experts in expert order + shared expert)
//
// Every projection sees an FP8 activation, and its output is rounded to BF16
// before anything reads it. These kernels are the plain statement of that:
// one thread per output row, sums in column order. They are the reference
// the faster tensor-core paths are held to.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_MOE_THREADS 256
#define DSV41_MAX_EXPERTS 512
#define DSV41_MAX_TOPK 8
#define DSV41_MAX_K 5120

// E2M1 by nibble: sign, two exponent bits, one mantissa bit. The magnitudes
// are 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ float dsv41_fp4(unsigned int nibble) {
    const unsigned int exponent = (nibble >> 1) & 3u;
    const float mantissa = 0.5f * (float)(nibble & 1u);
    const float v = exponent == 0u ? mantissa : ldexpf(1.0f + mantissa, (int)exponent - 1);
    return (nibble & 8u) ? -v : v;
}

// Routing of one token per block.
// x [T, H] BF16, gate [E, H] BF16, bias [E] FP32 -> chosen [T, top_k] u32 in
// order of selection, weights [T, top_k] FP32.
extern "C" __global__ void __launch_bounds__(DSV41_MOE_THREADS) dsv41_moe_route(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ bias,
    unsigned int* __restrict__ chosen,
    float* __restrict__ weights,
    unsigned int H,
    unsigned int experts,
    unsigned int top_k,
    float route_scale) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float xs[DSV41_MAX_K];
    __shared__ float score[DSV41_MAX_EXPERTS];
    for (unsigned int d = tid; d < H; d += DSV41_MOE_THREADS) xs[d] = __bfloat162float(x[(size_t)t * H + d]);
    __syncthreads();
    for (unsigned int e = tid; e < experts; e += DSV41_MOE_THREADS) {
        const __nv_bfloat16* row = gate + (size_t)e * H;
        float logit = 0.0f;
        for (unsigned int d = 0; d < H; ++d) logit += xs[d] * __bfloat162float(row[d]);
        // softplus as PyTorch computes it: the identity above 20.
        score[e] = sqrtf(logit > 20.0f ? logit : log1pf(expf(logit)));
    }
    __syncthreads();
    if (tid != 0) return;
    unsigned int pick[DSV41_MAX_TOPK];
    float sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) {
        unsigned int best = 0;
        float top = -INFINITY;
        for (unsigned int e = 0; e < experts; ++e) {
            bool taken = false;
            for (unsigned int j = 0; j < k; ++j) taken |= pick[j] == e;
            const float v = score[e] + bias[e];
            if (!taken && v > top) {
                top = v;
                best = e;
            }
        }
        pick[k] = best;
        sum += score[best];
    }
    for (unsigned int k = 0; k < top_k; ++k) {
        chosen[(size_t)t * top_k + k] = pick[k];
        weights[(size_t)t * top_k + k] = score[pick[k]] / (sum + 1e-20f) * route_scale;
    }
}

// Routing in two launches, for when one token must not take a millisecond:
// every (expert, token) is a block here, so a thread adds 20 products where
// `dsv41_moe_route`'s adds 5,120. The scores are the same FP32 sums in
// another order.
//
// score[t, e] = sqrt(softplus(x[t] . gate[e])). Grid (experts, T).
extern "C" __global__ void __launch_bounds__(DSV41_MOE_THREADS) dsv41_moe_scores(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    float* __restrict__ score,
    unsigned int H,
    unsigned int experts) {
    const unsigned int e = blockIdx.x, t = blockIdx.y;
    const float logit = dsv41_dot_wide(x + (size_t)t * H, gate + (size_t)e * H, H);
    if (threadIdx.x != 0) return;
    score[(size_t)t * experts + e] = sqrtf(logit > 20.0f ? logit : log1pf(expf(logit)));
}

// score[i] = sqrt(softplus(score[i])), in place: a batch's scores from the
// logits a GEMM left (x . gate^T in FP32), as `dsv41_moe_scores` leaves them.
extern "C" __global__ void dsv41_moe_soften(float* __restrict__ score, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float logit = score[i];
    score[i] = sqrtf(logit > 20.0f ? logit : log1pf(expf(logit)));
}

// The top_k of score + bias and their weights, as `dsv41_moe_route` picks
// them: best first, the lower expert first among equals. One block per
// token, one thread per expert: a thread counts the experts ahead of its own.
// With `by_expert` the chosen leave in ascending expert order, the order the
// reference adds their outputs in, each with its weight.
extern "C" __global__ void __launch_bounds__(DSV41_MAX_EXPERTS) dsv41_moe_pick(
    const float* __restrict__ score,
    const float* __restrict__ bias,
    unsigned int* __restrict__ chosen,
    float* __restrict__ weights,
    unsigned int experts,
    unsigned int top_k,
    float route_scale,
    unsigned int by_expert) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float value[DSV41_MAX_EXPERTS];
    __shared__ unsigned int pick[DSV41_MAX_TOPK];
    const float* s = score + (size_t)t * experts;
    if (tid < experts) value[tid] = s[tid] + bias[tid];
    __syncthreads();
    if (tid < experts) {
        const float mine = value[tid];
        unsigned int ahead = 0;
        for (unsigned int j = 0; j < experts; ++j)
            ahead += (value[j] > mine) || (value[j] == mine && j < tid);
        if (ahead < top_k) pick[ahead] = tid;
    }
    __syncthreads();
    if (tid != 0) return;
    float sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) sum += s[pick[k]];
    unsigned int order[DSV41_MAX_TOPK];
    for (unsigned int k = 0; k < top_k; ++k) {
        unsigned int at = k;
        for (; by_expert && at > 0 && order[at - 1] > pick[k]; --at) order[at] = order[at - 1];
        order[at] = pick[k];
    }
    for (unsigned int k = 0; k < top_k; ++k) {
        chosen[(size_t)t * top_k + k] = order[k];
        weights[(size_t)t * top_k + k] = s[order[k]] / (sum + 1e-20f) * route_scale;
    }
}

// out[row] = BF16(sum over blocks of 32 columns of (sum_k w4[row, k] * x[k]) * 2^scale),
// one vector x. w4 [n, k / 2] packs two E2M1 values a byte, low nibble first;
// scales [n, k / 32] are E8M0. One thread per row.
extern "C" __global__ void __launch_bounds__(DSV41_MOE_THREADS) dsv41_fp4_gemv(
    const __nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ w4,
    const unsigned char* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    unsigned int n,
    unsigned int k) {
    __shared__ float xs[DSV41_MAX_K];
    for (unsigned int d = threadIdx.x; d < k; d += DSV41_MOE_THREADS) xs[d] = __bfloat162float(x[d]);
    __syncthreads();
    const unsigned int row = blockIdx.x * DSV41_MOE_THREADS + threadIdx.x;
    if (row >= n) return;
    const unsigned char* w = w4 + (size_t)row * (k / 2);
    const unsigned char* s = scales + (size_t)row * (k / 32);
    float acc = 0.0f;
    for (unsigned int block = 0; block < k / 32; ++block) {
        float part = 0.0f;
        for (unsigned int i = 0; i < 16; ++i) {
            const unsigned int byte = w[block * 16 + i];
            part += dsv41_fp4(byte & 15u) * xs[block * 32 + 2 * i];
            part += dsv41_fp4(byte >> 4) * xs[block * 32 + 2 * i + 1];
        }
        acc += part * __uint_as_float((unsigned int)s[block] << 23);
    }
    out[row] = __float2bfloat16(acc);
}

// h = BF16(weight * silu(min(gate, limit)) * clamp(up, -limit, limit)).
extern "C" __global__ void dsv41_swiglu(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ h,
    unsigned int n,
    float weight,
    float limit) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = __float2bfloat16(
        dsv41_swiglu_one(__bfloat162float(gate[i]), __bfloat162float(up[i]), weight, limit));
}

// acc += x, FP32 += BF16; and the rounding back.
extern "C" __global__ void dsv41_accumulate(
    float* __restrict__ acc, const __nv_bfloat16* __restrict__ x, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += __bfloat162float(x[i]);
}

extern "C" __global__ void dsv41_round_bf16(
    const float* __restrict__ acc, __nv_bfloat16* __restrict__ out, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(acc[i]);
}
