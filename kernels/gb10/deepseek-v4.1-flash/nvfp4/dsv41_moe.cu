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
    const float g = fminf(__bfloat162float(gate[i]), limit);
    const float u = fminf(fmaxf(__bfloat162float(up[i]), -limit), limit);
    const float silu = g * (1.0f / (1.0f + expf(-g)));
    h[i] = __float2bfloat16(weight * (silu * u));
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
