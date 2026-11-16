// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash, the two side paths that run in FP32 in the reference
// (`Compressor` and `Engram` in the release's `inference/model.py`).
//
// Compressor, on a layer that produces compressed keys at ratio > 1: a key
// and a gate projection of the attention input, both FP32, pooled over each
// group of `ratio` tokens by a per-value softmax of the gate, rounded to
// BF16 and normed. Engram: an n-gram lookup's value added into every copy of
// the stream through a gate that compares the copy with the lookup's key.
//
// Plain forms: one thread per output value, sums in input order; but for the
// projections, which are a block per value.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_SIDE_THREADS 256

// out[m, row] = sum_k w[row, k] * x[m, k] in FP32, not rounded. x [M, k]
// BF16, w [n, k] BF16, out [M, n] FP32. The reference widens both to FP32
// first, which is exact. A block per (row, m), its 256 threads adding the
// products in a tree: a thread a row, adding them one after another, takes
// a millisecond over 5,120 columns. Grid (n, M), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_DOT_THREADS) dsv41_gemv_f32(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out,
    unsigned int k) {
    const float sum =
        dsv41_dot_wide(x + (size_t)blockIdx.y * k, w + (size_t)blockIdx.x * k, k);
    if (threadIdx.x == 0) out[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = sum;
}

// pooled[g, d] = BF16(sum_r kv[g, r, d] * softmax_r(score[g, r, d])) over the
// `ratio` tokens of each group. kv and score [groups, ratio, dim] FP32.
// Grid (ceil(dim / threads), groups).
extern "C" __global__ void dsv41_compress_pool(
    const float* __restrict__ kv,
    const float* __restrict__ score,
    __nv_bfloat16* __restrict__ pooled,
    unsigned int ratio,
    unsigned int dim) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    kv += (size_t)blockIdx.y * ratio * dim;
    score += (size_t)blockIdx.y * ratio * dim;
    float top = score[d];
    for (unsigned int r = 1; r < ratio; ++r) top = fmaxf(top, score[r * dim + d]);
    float total = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r) total += expf(score[r * dim + d] - top);
    float acc = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r)
        acc += kv[r * dim + d] * (expf(score[r * dim + d] - top) / total);
    pooled[(size_t)blockIdx.y * dim + d] = __float2bfloat16(acc);
}

// Engram table rows as the values a lookup reads. Column c of the lookup is
// row row_slots[c] of `rows` (256 FP8 E4M3 values a row) under row
// scale_slots[c] of `scales` (8 E8M0 scales a row, one per 32 values), as
// the table stores them:
//   out[c * 256 + d] = BF16(E4M3(row[d]) * 2^(scale[d / 32] - 127))
extern "C" __global__ void dsv41_engram_rows(
    const unsigned char* __restrict__ rows,
    const unsigned char* __restrict__ scales,
    const unsigned int* __restrict__ row_slots,
    const unsigned int* __restrict__ scale_slots,
    __nv_bfloat16* __restrict__ out,
    unsigned int columns) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns * 256) return;
    const unsigned int c = i / 256, d = i % 256;
    const unsigned int b = rows[(size_t)row_slots[c] * 256 + d];
    const int e = (b >> 3) & 0xF;
    const float m = (float)(b & 7);
    // Subnormal below exponent 1; the bias is 7 and the mantissa 3 bits.
    float v = e == 0 ? ldexpf(m, -9) : ldexpf(8.0f + m, e - 10);
    if (b & 0x80) v = -v;
    out[i] = __float2bfloat16(ldexpf(v, (int)scales[(size_t)scale_slots[c] * 8 + d / 32] - 127));
}

// Engram on one copy of one token per block. stream [T, hc, H] BF16 in
// place; kv is the lookup's projection, (hc + 1) * H BF16 a token, hc keys
// then one value, in the four parts it was computed in: a part a quarter of
// every token's values, the T tokens' one after another. weight [hc, H]
// FP32 is q_weight * k_weight.
//   gate_j = sigmoid(sign(dot_j) * sqrt(max(|dot_j|, 1e-6)))
//   dot_j  = sum(h_j * weight_j * key_j) * rsqrt(mean(h_j^2) + eps)
//            * rsqrt(mean(key_j^2) + eps) / sqrt(H)
//   h_j   += gate_j * value
// A copy's gate reads nothing of the other copies, so each has its block and
// they run side by side. Grid (T, hc), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_SIDE_THREADS) dsv41_engram_gate(
    __nv_bfloat16* __restrict__ stream,
    const __nv_bfloat16* __restrict__ kv,
    const float* __restrict__ weight,
    unsigned int H,
    unsigned int hc,
    float eps) {
    const unsigned int t = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    __nv_bfloat16* h = stream + ((size_t)t * hc + j) * H;
    // Value e of this token's projection.
    const unsigned int part = (hc + 1) * H / DSV41_PARTS;
    const size_t part_stride = (size_t)gridDim.x * part;
    auto projected = [&](unsigned int e) {
        return __bfloat162float(kv[(e / part) * part_stride + (size_t)t * part + e % part]);
    };
    const float* w = weight + (size_t)j * H;
    __shared__ float red[3][DSV41_SIDE_THREADS];
    __shared__ float gate;
    float hh = 0.0f, kk = 0.0f, hk = 0.0f;
    for (unsigned int d = tid; d < H; d += DSV41_SIDE_THREADS) {
        const float hv = __bfloat162float(h[d]);
        const float kval = projected(j * H + d);
        hh += hv * hv;
        kk += kval * kval;
        hk += hv * w[d] * kval;
    }
    red[0][tid] = hh;
    red[1][tid] = kk;
    red[2][tid] = hk;
    __syncthreads();
    for (unsigned int width = DSV41_SIDE_THREADS / 2; width > 0; width >>= 1) {
        if (tid < width)
            for (int s = 0; s < 3; ++s) red[s][tid] += red[s][tid + width];
        __syncthreads();
    }
    if (tid == 0) {
        const float rstd = rsqrtf(red[0][0] / (float)H + eps) * rsqrtf(red[1][0] / (float)H + eps);
        const float dot = red[2][0] * rstd * rsqrtf((float)H);
        const float root = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
        gate = 1.0f / (1.0f + expf(-root));
    }
    __syncthreads();
    for (unsigned int d = tid; d < H; d += DSV41_SIDE_THREADS)
        h[d] = __float2bfloat16(__bfloat162float(h[d]) + gate * projected(hc * H + d));
}
