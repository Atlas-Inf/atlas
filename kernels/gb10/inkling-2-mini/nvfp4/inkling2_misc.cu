// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 glue for the per-visit layer, with the reference's dtypes (transformers
// PR #51050, Apache-2.0):
//   inkling2_rmsnorm      BF16(w * x * rsqrt(mean(x^2) + eps)), x fp32 or BF16
//   inkling2_gate_mul     BF16(a * sigmoid(g))   (attention output gate)
//   inkling2_residual     r = (restart ? 0 : x) + c   (fp32; x, c BF16)
//   inkling2_layer_out    BF16(r + m)                 (r fp32, m BF16)
//   inkling2_f32_to_bf16

#include "atlas_pdl.cuh"
#include <cuda_bf16.h>

// one CTA (256) per row; exactly one of x_f32 / x_bf16 non-NULL
extern "C" __global__ void inkling2_rmsnorm(const float* __restrict__ x_f32, const __nv_bfloat16* __restrict__ x_bf16,
                                            const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                            unsigned int H, unsigned int in_stride, unsigned int out_stride, float eps) {
    atlas_pdl_enter();
    const unsigned long long r = blockIdx.x;
    __shared__ float red[8];
    __shared__ float s_rstd;
    float ss = 0.0f;
    for (unsigned int e = threadIdx.x; e < H; e += blockDim.x) {
        const float v = x_f32 != nullptr ? x_f32[r * in_stride + e] : __bfloat162float(x_bf16[r * in_stride + e]);
        ss += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned int i = 0; i < blockDim.x / 32; i++) s += red[i];
        s_rstd = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();
    for (unsigned int e = threadIdx.x; e < H; e += blockDim.x) {
        const float v = x_f32 != nullptr ? x_f32[r * in_stride + e] : __bfloat162float(x_bf16[r * in_stride + e]);
        // Inkling2RMSNorm: (w * fp32(x) * rstd), one cast at the end
        y[r * out_stride + e] = __float2bfloat16(__bfloat162float(w[e]) * (v * s_rstd));
    }
}

// elementwise over n values
extern "C" __global__ void inkling2_gate_mul(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ g,
                                             __nv_bfloat16* __restrict__ out, unsigned int n) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = __bfloat162float(g[i]);
    out[i] = __float2bfloat16(__bfloat162float(a[i]) / (1.0f + __expf(-gv)));
}

extern "C" __global__ void inkling2_residual(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ c,
                                             float* __restrict__ r, unsigned int n, unsigned int restart) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r[i] = (restart ? 0.0f : __bfloat162float(x[i])) + __bfloat162float(c[i]);
}

extern "C" __global__ void inkling2_layer_out(const float* __restrict__ r, const __nv_bfloat16* __restrict__ m,
                                              __nv_bfloat16* __restrict__ out, unsigned int n) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(r[i] + __bfloat162float(m[i]));
}

extern "C" __global__ void inkling2_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ out,
                                                unsigned int n) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(x[i]);
}
