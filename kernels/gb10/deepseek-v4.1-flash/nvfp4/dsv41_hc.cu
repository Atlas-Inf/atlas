// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash Hyper-Connections: the residual stream is `hc` copies,
// BF16, and every sublayer sits between a collapse to one vector and an
// expansion back. The reference is `Block.hc_mixes / hc_pre / hc_post` in the
// release's `inference/model.py` and `hc_split_sinkhorn` in its `kernel.py`.
//
// V4.1 differs from DeepSeek-V4 in which coefficients a collapse uses: a
// sublayer is collapsed with the `pre` the sublayer BEFORE it computed
// (attention with the previous block's FFN site, the FFN with this block's
// attention site, the head with the last FFN site), and layer 0's attention
// with the one-hot (1, 0, 0, 0). So the mix, the collapse and the expansion
// are three kernels here, and `pre` is handed from site to site.
//
// One block per token, 256 threads. Every sum runs in an order fixed by the
// thread count, so a token's bytes do not depend on the tokens beside it.

#include <cuda_bf16.h>

#define DSV41_HC_THREADS 256
#define DSV41_HC_MAX 4

__device__ __forceinline__ float dsv41_hc_reduce(float* red, unsigned int tid) {
    for (unsigned int width = DSV41_HC_THREADS / 2; width > 0; width >>= 1) {
        if (tid < width) red[tid] += red[tid + width];
        __syncthreads();
    }
    return red[0];
}

__device__ __forceinline__ float dsv41_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// streams [T, hc, H] BF16 -> pre [T, hc], post [T, hc], comb [T, hc, hc].
// hc_fn [(2 + hc) * hc, hc * H], hc_scale [3], hc_base [(2 + hc) * hc], FP32.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_mix(
    const __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    unsigned int H,
    unsigned int hc,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x;
    const unsigned int hc_dim = hc * H, mix_hc = (2 + hc) * hc;
    const __nv_bfloat16* x = streams + (size_t)t * hc_dim;
    __shared__ float red[DSV41_HC_THREADS];
    __shared__ float mix[(2 + DSV41_HC_MAX) * DSV41_HC_MAX];
    __shared__ float rsqrt;

    // One statistic for the whole flattened stream.
    float ss = 0.0f;
    for (unsigned int k = tid; k < hc_dim; k += DSV41_HC_THREADS) {
        const float v = __bfloat162float(x[k]);
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    const float sum = dsv41_hc_reduce(red, tid);
    if (tid == 0) rsqrt = rsqrtf(sum / (float)hc_dim + norm_eps);
    __syncthreads();

    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* row = hc_fn + (size_t)m * hc_dim;
        float acc = 0.0f;
        for (unsigned int k = tid; k < hc_dim; k += DSV41_HC_THREADS) acc += row[k] * __bfloat162float(x[k]);
        red[tid] = acc;
        __syncthreads();
        const float dot = dsv41_hc_reduce(red, tid);
        if (tid == 0) mix[m] = dot * rsqrt;
        __syncthreads();
    }
    if (tid != 0) return;

    float comb[DSV41_HC_MAX * DSV41_HC_MAX];
    for (unsigned int j = 0; j < hc; ++j) {
        pre_out[(size_t)t * hc + j] = dsv41_sigmoid(mix[j] * hc_scale[0] + hc_base[j]) + hc_eps;
        post_out[(size_t)t * hc + j] = 2.0f * dsv41_sigmoid(mix[j + hc] * hc_scale[1] + hc_base[j + hc]);
    }
    for (unsigned int j = 0; j < hc; ++j)
        for (unsigned int k = 0; k < hc; ++k)
            comb[j * hc + k] = mix[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2];
    // comb = softmax over each row, plus eps.
    for (unsigned int j = 0; j < hc; ++j) {
        float top = comb[j * hc];
        for (unsigned int k = 1; k < hc; ++k) top = fmaxf(top, comb[j * hc + k]);
        float row = 0.0f;
        for (unsigned int k = 0; k < hc; ++k) {
            comb[j * hc + k] = expf(comb[j * hc + k] - top);
            row += comb[j * hc + k];
        }
        for (unsigned int k = 0; k < hc; ++k) comb[j * hc + k] = comb[j * hc + k] / row + hc_eps;
    }
    // Columns once, then `sinkhorn_iters - 1` rounds of rows and columns,
    // each dividing by (sum + eps).
    for (unsigned int round = 0; round < sinkhorn_iters; ++round) {
        if (round > 0) {
            for (unsigned int j = 0; j < hc; ++j) {
                float row = 0.0f;
                for (unsigned int k = 0; k < hc; ++k) row += comb[j * hc + k];
                for (unsigned int k = 0; k < hc; ++k) comb[j * hc + k] /= row + hc_eps;
            }
        }
        for (unsigned int k = 0; k < hc; ++k) {
            float col = 0.0f;
            for (unsigned int j = 0; j < hc; ++j) col += comb[j * hc + k];
            for (unsigned int j = 0; j < hc; ++j) comb[j * hc + k] /= col + hc_eps;
        }
    }
    for (unsigned int i = 0; i < hc * hc; ++i) comb_out[(size_t)t * hc * hc + i] = comb[i];
}

// y[t, d] = sum_j pre[t, j] * streams[t, j, d], rounded to BF16.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_collapse(
    const __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ pre,
    __nv_bfloat16* __restrict__ y,
    unsigned int H,
    unsigned int hc) {
    const unsigned int t = blockIdx.x;
    const __nv_bfloat16* x = streams + (size_t)t * hc * H;
    const float* p = pre + (size_t)t * hc;
    for (unsigned int d = threadIdx.x; d < H; d += DSV41_HC_THREADS) {
        float acc = p[0] * __bfloat162float(x[d]);
        for (unsigned int j = 1; j < hc; ++j) acc += p[j] * __bfloat162float(x[j * H + d]);
        y[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}

// out[t, j, d] = post[t, j] * x[t, d] + sum_i comb[t, i, j] * residual[t, i, d],
// rounded to BF16. `out` may be `residual`.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_expand(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* out,
    unsigned int H,
    unsigned int hc) {
    const unsigned int t = blockIdx.x;
    const __nv_bfloat16* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    __nv_bfloat16* o = out + (size_t)t * hc * H;
    for (unsigned int d = threadIdx.x; d < H; d += DSV41_HC_THREADS) {
        const float xd = __bfloat162float(x[(size_t)t * H + d]);
        float rv[DSV41_HC_MAX];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = __bfloat162float(res[i * H + d]);
        for (unsigned int j = 0; j < hc; ++j) {
            float mixed = c[j] * rv[0];
            for (unsigned int i = 1; i < hc; ++i) mixed += c[i * hc + j] * rv[i];
            o[j * H + d] = __float2bfloat16(p[j] * xd + mixed);
        }
    }
}
