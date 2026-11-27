// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 glue for the per-visit layer, with the reference's dtypes (transformers
// PR #51050, Apache-2.0):
//   inkling2_rmsnorm      BF16(w * x * rsqrt(mean(x^2) + eps)), x fp32 or BF16
//   inkling2_gate_mul     BF16(a * sigmoid(g))   (attention output gate)
//   inkling2_f32_to_bf16
//   inkling2_window_meta  the window layers' slots and block tables, through the window map

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

// values a thread holds in registers for one row (rows up to 16 x the CTA width)
#define RMS_MAX_PER_THREAD 16

// rsqrt(mean of squares + eps) over the CTA from each thread's partial sum ss
__device__ __forceinline__ float rms_block_rstd(float ss, float* red, float* s_rstd, unsigned int H, float eps) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned int i = 0; i < blockDim.x / 32; i++) s += red[i];
        *s_rstd = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();
    return *s_rstd;
}

// one CTA (256) per row and input (grid = rows, inputs <= 3); exactly one of x_f32 /
// x_bf16 non-NULL. Input g reads x + g * in_plane with weight w0 / w1 / w2 and writes
// y + g * out_plane: an attention visit's three input norms in one launch.
extern "C" __global__ void inkling2_rmsnorm(const float* __restrict__ x_f32, const __nv_bfloat16* __restrict__ x_bf16,
                                            const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
                                            const __nv_bfloat16* __restrict__ w2, __nv_bfloat16* __restrict__ y,
                                            unsigned int H, unsigned int in_stride, unsigned int out_stride,
                                            unsigned long long in_plane, unsigned long long out_plane, float eps) {
    atlas_pdl_enter();
    const unsigned long long r = blockIdx.x;
    const unsigned int g = blockIdx.y;
    if (x_f32 != nullptr) x_f32 += g * in_plane;
    if (x_bf16 != nullptr) x_bf16 += g * in_plane;
    y += g * out_plane;
    const __nv_bfloat16* __restrict__ w = g == 0 ? w0 : (g == 1 ? w1 : w2);
    __shared__ float red[8];
    __shared__ float s_rstd;
    float ss = 0.0f;
    if (H <= blockDim.x * RMS_MAX_PER_THREAD) {
        // a row's values and weights in registers: every load is issued at once instead of one
        // latency per stride step, and the sums run in the same order as below (bit-identical)
        float v[RMS_MAX_PER_THREAD];
        float wv[RMS_MAX_PER_THREAD];
#pragma unroll
        for (unsigned int i = 0; i < RMS_MAX_PER_THREAD; i++) {
            const unsigned int e = threadIdx.x + i * blockDim.x;
            v[i] = 0.0f;
            wv[i] = 0.0f;
            if (e < H) {
                v[i] = x_f32 != nullptr ? x_f32[r * in_stride + e] : __bfloat162float(x_bf16[r * in_stride + e]);
                wv[i] = __bfloat162float(w[e]);
            }
        }
#pragma unroll
        for (unsigned int i = 0; i < RMS_MAX_PER_THREAD; i++)
            if (threadIdx.x + i * blockDim.x < H) ss += v[i] * v[i];
        const float rstd = rms_block_rstd(ss, red, &s_rstd, H, eps);
#pragma unroll
        for (unsigned int i = 0; i < RMS_MAX_PER_THREAD; i++) {
            const unsigned int e = threadIdx.x + i * blockDim.x;
            // Inkling2RMSNorm: (w * fp32(x) * rstd), one cast at the end
            if (e < H) y[r * out_stride + e] = __float2bfloat16(wv[i] * (v[i] * rstd));
        }
        return;
    }
    for (unsigned int e = threadIdx.x; e < H; e += blockDim.x) {
        const float v = x_f32 != nullptr ? x_f32[r * in_stride + e] : __bfloat162float(x_bf16[r * in_stride + e]);
        ss += v * v;
    }
    const float rstd = rms_block_rstd(ss, red, &s_rstd, H, eps);
    for (unsigned int e = threadIdx.x; e < H; e += blockDim.x) {
        const float v = x_f32 != nullptr ? x_f32[r * in_stride + e] : __bfloat162float(x_bf16[r * in_stride + e]);
        // Inkling2RMSNorm: (w * fp32(x) * rstd), one cast at the end
        y[r * out_stride + e] = __float2bfloat16(__bfloat162float(w[e]) * (v * rstd));
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

extern "C" __global__ void inkling2_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ out,
                                                unsigned int n) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(x[i]);
}

// Window-bounded KV (spark-runtime kv_cache/window.rs): the sliding-window layers keep their
// rows at a local block the primary block maps to. This maps each row's KV slot, and each
// block table's entries from its first row's window (from the 32-key tile holding the first
// key, where the prefill kernels start) through its last row; the attention never reads the
// entries outside. Unmapped blocks (graph padding rows' dummy block) go to the null block 0.
// Grid: one CTA per table (one a row when table_stride > 0, else the rows' shared table);
// CTA 0 also maps the slots.
#define INK_NO_WINDOW 0xFFFFFFFFu
#define INK_KEY_TILE 32u

extern "C" __global__ void inkling2_window_meta(const long long* __restrict__ slots, const int* __restrict__ positions,
                                                const int* __restrict__ tables, const unsigned int* __restrict__ map,
                                                long long* __restrict__ local_slots, int* __restrict__ local_tables,
                                                unsigned int rows, unsigned int table_stride, unsigned int block_size,
                                                unsigned int window) {
    atlas_pdl_enter();
    const unsigned int t = blockIdx.x;
    if (t == 0) {
        for (unsigned int r = threadIdx.x; r < rows; r += blockDim.x) {
            const long long s = slots[r];
            const unsigned int local = map[(unsigned long long)s / block_size];
            local_slots[r] = (long long)(local == INK_NO_WINDOW ? 0u : local) * block_size + s % block_size;
        }
    }
    const unsigned int first = (unsigned int)positions[table_stride == 0 ? 0 : t];
    const unsigned int last = table_stride == 0 ? (unsigned int)positions[rows - 1] : first;
    const unsigned int key = first + 1 > window ? first + 1 - window : 0u;
    const unsigned int lo = key / INK_KEY_TILE * INK_KEY_TILE / block_size, hi = last / block_size;
    const int* table = tables + (unsigned long long)t * table_stride;
    int* out = local_tables + (unsigned long long)t * table_stride;
    for (unsigned int b = lo + threadIdx.x; b <= hi; b += blockDim.x) {
        const unsigned int local = map[(unsigned int)table[b]];
        out[b] = (int)(local == INK_NO_WINDOW ? 0u : local);
    }
}
