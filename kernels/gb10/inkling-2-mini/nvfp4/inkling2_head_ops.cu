// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 per-head epilogues and the loop's cross-pass message gate. Reference:
// transformers PR #51050 `Inkling2(Scaled|LogScaled)L2Clip`, `Inkling2UnweightedRMSNorm`,
// `Inkling2LogScale` and `Inkling2LoopMessageGate` (Apache-2.0). Roundings follow the
// reference: a clipped head is rounded to BF16 before the log scale and again after.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define INK_MAX_HD 128

// One warp per (row, head).
//   mode 0: y = clip(scale * x, threshold)  (scale NULL = 1); then, with positions,
//           y *= 1 + alpha * ln(max((pos + 1) / n_floor, 1))
//   mode 1: y = x * rsqrt(mean(x^2) + eps)
extern "C" __global__ void inkling2_head_norm(
    const __nv_bfloat16* __restrict__ x,   // [rows, in_stride]
    __nv_bfloat16* __restrict__ y,         // [rows, out_stride]
    const __nv_bfloat16* __restrict__ scale, // [heads, head_dim] or NULL
    const int* __restrict__ positions,     // [rows] or NULL
    const unsigned int rows,
    const unsigned int heads,
    const unsigned int head_dim,
    const unsigned int in_stride,
    const unsigned int out_stride,
    const unsigned int mode,
    const float threshold_or_eps,
    const float n_floor,
    const float alpha
) {
    atlas_pdl_enter();
    const unsigned int gw = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gw >= rows * heads) return;
    const unsigned int row = gw / heads, head = gw % heads;
    const __nv_bfloat16* xp = x + (unsigned long long)row * in_stride + head * head_dim;
    float v[INK_MAX_HD / 32];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < INK_MAX_HD / 32; i++) {
        const unsigned int e = lane + 32 * i;
        float t = 0.0f;
        if (e < head_dim) {
            t = __bfloat162float(xp[e]);
            if (mode == 0 && scale != nullptr) t *= __bfloat162float(scale[head * head_dim + e]);
        }
        v[i] = t;
        ss += t * t;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    float f;
    if (mode == 0) f = fminf(threshold_or_eps / (sqrtf(ss) + 1e-6f), 1.0f);
    else f = rsqrtf(ss / (float)head_dim + threshold_or_eps);
    float tau = 1.0f;
    if (mode == 0 && positions != nullptr)
        tau = 1.0f + alpha * logf(fmaxf((float)(positions[row] + 1) / n_floor, 1.0f));
    __nv_bfloat16* yp = y + (unsigned long long)row * out_stride + head * head_dim;
#pragma unroll
    for (int i = 0; i < INK_MAX_HD / 32; i++) {
        const unsigned int e = lane + 32 * i;
        if (e >= head_dim) continue;
        __nv_bfloat16 o = __float2bfloat16(v[i] * f);
        if (positions != nullptr && mode == 0) o = __float2bfloat16(__bfloat162float(o) * tau);
        yp[e] = o;
    }
}

// out = (1 - tanh(F z)) * h + tanh(G z) * m,  z = BF16(D BF16([rms0(h); rms0(m)])).
// One CTA (256 threads) per row; rank <= 64.
#define INK_MAX_RANK 64
extern "C" __global__ void inkling2_msg_gate(
    const float* __restrict__ h,           // [rows, H]
    const float* __restrict__ m,           // [rows, H]
    const __nv_bfloat16* __restrict__ D,   // [rank, 2H]
    const __nv_bfloat16* __restrict__ F,   // [H, rank] forget
    const __nv_bfloat16* __restrict__ G,   // [H, rank] message (gain)
    float* __restrict__ out,               // [rows, H] (may alias h)
    const unsigned int rows,
    const unsigned int H,
    const unsigned int rank,
    const float eps
) {
    atlas_pdl_enter();
    const unsigned long long row = blockIdx.x;
    if (row >= rows) return;
    const float* hp = h + row * H;
    const float* mp = m + row * H;
    __shared__ float red[2][8];
    __shared__ float s_rstd[2];
    __shared__ float z[INK_MAX_RANK];
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nwarps = blockDim.x / 32;

    float sh = 0.0f, sm = 0.0f;
    for (unsigned int e = threadIdx.x; e < H; e += blockDim.x) {
        sh += hp[e] * hp[e];
        sm += mp[e] * mp[e];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sh += __shfl_xor_sync(0xffffffff, sh, off);
        sm += __shfl_xor_sync(0xffffffff, sm, off);
    }
    if (lane == 0) {
        red[0][warp] = sh;
        red[1][warp] = sm;
    }
    __syncthreads();
    if (threadIdx.x < 2) {
        float s = 0.0f;
        for (unsigned int w = 0; w < nwarps; w++) s += red[threadIdx.x][w];
        s_rstd[threadIdx.x] = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();

    // z: each warp takes rank rows in turn, lanes stride the 2H inputs
    for (unsigned int r = warp; r < rank; r += nwarps) {
        const __nv_bfloat16* d = D + (unsigned long long)r * 2 * H;
        float acc = 0.0f;
        for (unsigned int e = lane; e < 2 * H; e += 32) {
            const float in = e < H ? hp[e] * s_rstd[0] : mp[e - H] * s_rstd[1];
            acc += __bfloat162float(d[e]) * __bfloat162float(__float2bfloat16(in));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) z[r] = __bfloat162float(__float2bfloat16(acc));
    }
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < H; e += blockDim.x) {
        float f = 0.0f, g = 0.0f;
        for (unsigned int r = 0; r < rank; r++) {
            f += __bfloat162float(F[(unsigned long long)e * rank + r]) * z[r];
            g += __bfloat162float(G[(unsigned long long)e * rank + r]) * z[r];
        }
        out[row * H + e] = (1.0f - tanhf(f)) * hp[e] + tanhf(g) * mp[e];
    }
}
