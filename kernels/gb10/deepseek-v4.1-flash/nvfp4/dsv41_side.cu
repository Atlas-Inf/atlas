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
// Plain forms: one thread per output value, sums in input order.

#include <cuda_bf16.h>

#define DSV41_SIDE_THREADS 256
#define DSV41_SIDE_MAX_K 5120

// out[row] = sum_k w[row, k] * x[k] in FP32, not rounded. x [k] BF16,
// w [n, k] BF16, out [n] FP32. The reference widens both to FP32 first,
// which is exact.
extern "C" __global__ void __launch_bounds__(DSV41_SIDE_THREADS) dsv41_gemv_f32(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out,
    unsigned int n,
    unsigned int k) {
    __shared__ float xs[DSV41_SIDE_MAX_K];
    for (unsigned int d = threadIdx.x; d < k; d += DSV41_SIDE_THREADS) xs[d] = __bfloat162float(x[d]);
    __syncthreads();
    const unsigned int row = blockIdx.x * DSV41_SIDE_THREADS + threadIdx.x;
    if (row >= n) return;
    const __nv_bfloat16* wr = w + (size_t)row * k;
    float acc = 0.0f;
    for (unsigned int d = 0; d < k; ++d) acc += xs[d] * __bfloat162float(wr[d]);
    out[row] = acc;
}

// pooled[d] = BF16(sum_r kv[r, d] * softmax_r(score[r, d])) over the `ratio`
// tokens of one group. kv and score [ratio, dim] FP32.
extern "C" __global__ void dsv41_compress_pool(
    const float* __restrict__ kv,
    const float* __restrict__ score,
    __nv_bfloat16* __restrict__ pooled,
    unsigned int ratio,
    unsigned int dim) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float top = score[d];
    for (unsigned int r = 1; r < ratio; ++r) top = fmaxf(top, score[r * dim + d]);
    float total = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r) total += expf(score[r * dim + d] - top);
    float acc = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r)
        acc += kv[r * dim + d] * (expf(score[r * dim + d] - top) / total);
    pooled[d] = __float2bfloat16(acc);
}

// Engram table rows as the values a lookup reads:
//   out[i] = BF16(E4M3(rows[i]) * 2^(scales[i / 32] - 127))
// rows [n] FP8 E4M3 and scales [n / 32] E8M0, as the table stores them.
extern "C" __global__ void dsv41_engram_rows(
    const unsigned char* __restrict__ rows,
    const unsigned char* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int b = rows[i];
    const int e = (b >> 3) & 0xF;
    const float m = (float)(b & 7);
    // Subnormal below exponent 1; the bias is 7 and the mantissa 3 bits.
    float v = e == 0 ? ldexpf(m, -9) : ldexpf(8.0f + m, e - 10);
    if (b & 0x80) v = -v;
    out[i] = __float2bfloat16(ldexpf(v, (int)scales[i / 32] - 127));
}

// Engram on one token per block. stream [T, hc, H] BF16 in place;
// kv [T, (hc + 1) * H] BF16 is the lookup's projection, hc keys then one
// value; weight [hc, H] FP32 is q_weight * k_weight.
//   gate_j = sigmoid(sign(dot_j) * sqrt(max(|dot_j|, 1e-6)))
//   dot_j  = sum(h_j * weight_j * key_j) * rsqrt(mean(h_j^2) + eps)
//            * rsqrt(mean(key_j^2) + eps) / sqrt(H)
//   h_j   += gate_j * value
extern "C" __global__ void __launch_bounds__(DSV41_SIDE_THREADS) dsv41_engram_gate(
    __nv_bfloat16* __restrict__ stream,
    const __nv_bfloat16* __restrict__ kv,
    const float* __restrict__ weight,
    unsigned int H,
    unsigned int hc,
    float eps) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x;
    __nv_bfloat16* h = stream + (size_t)t * hc * H;
    const __nv_bfloat16* key = kv + (size_t)t * (hc + 1) * H;
    const __nv_bfloat16* value = key + (size_t)hc * H;
    __shared__ float red[3][DSV41_SIDE_THREADS];
    __shared__ float gate[4];
    for (unsigned int j = 0; j < hc; ++j) {
        float hh = 0.0f, kk = 0.0f, hk = 0.0f;
        for (unsigned int d = tid; d < H; d += DSV41_SIDE_THREADS) {
            const float hv = __bfloat162float(h[j * H + d]);
            const float kval = __bfloat162float(key[j * H + d]);
            hh += hv * hv;
            kk += kval * kval;
            hk += hv * weight[j * H + d] * kval;
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
            gate[j] = 1.0f / (1.0f + expf(-root));
        }
        __syncthreads();
    }
    for (unsigned int d = tid; d < H; d += DSV41_SIDE_THREADS) {
        const float v = __bfloat162float(value[d]);
        for (unsigned int j = 0; j < hc; ++j)
            h[j * H + d] = __float2bfloat16(__bfloat162float(h[j * H + d]) + gate[j] * v);
    }
}
