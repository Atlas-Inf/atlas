// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 per-head epilogues and the loop's cross-pass message gates. Reference:
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

// The loop's message gates, out = (1 - tanh(F z)) * h + tanh(G z) * m with
// z = BF16(D BF16([rms0(h); rms0(m)])), for up to three gates of one visit at once
// (gate g reads h + g * h_stride, m + g * m_stride and writes out + g * out_stride;
// h_stride 0 fans one input out to several gates). Two launches spread the work over
// the GPU: `_z` one warp per (row, gate, rank element), `_out` one thread per
// (row, channel) looping over the gates, every gate's inputs read before any output
// is written (out may alias h).
#define INK_MAX_RANK 64
#define INK_MAX_GATES 3
#define INK_Z_WARPS 4

__device__ __forceinline__ const __nv_bfloat16* ink_pick(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                                         const __nv_bfloat16* c, unsigned int g) {
    return g == 0 ? a : (g == 1 ? b : c);
}

// Rows per block: each block reads its weight rows once for this many rows (one row a
// block re-read D, F and G from L2 for every prefill row).
#define INK_Z_ROWS 8
#define INK_OUT_ROWS 16

// grid (ceil(rank / INK_Z_WARPS), ceil(rows / INK_Z_ROWS), gates), block 32 * INK_Z_WARPS
extern "C" __global__ void inkling2_msg_gate_z(
    const float* __restrict__ h,                  // [rows, H] per gate
    const float* __restrict__ m,                  // [rows, H] per gate
    const __nv_bfloat16* __restrict__ D0,         // [rank, 2H] per gate
    const __nv_bfloat16* __restrict__ D1,
    const __nv_bfloat16* __restrict__ D2,
    float* __restrict__ z,                        // [gates, rows, rank]
    const unsigned int rows,
    const unsigned int H,
    const unsigned int rank,
    const unsigned long long h_stride,
    const unsigned long long m_stride,
    const float eps
) {
    atlas_pdl_enter();
    const unsigned int row0 = blockIdx.y * INK_Z_ROWS, g = blockIdx.z;
    const unsigned int nr = min(INK_Z_ROWS, rows - row0);
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* hb = h + g * h_stride + (unsigned long long)row0 * H;
    const float* mb = m + g * m_stride + (unsigned long long)row0 * H;
    __shared__ float s_rstd[INK_Z_ROWS][2];
    // one warp per (row, half): the rms of h and of m of each row
    for (unsigned int t = warp; t < 2 * nr; t += INK_Z_WARPS) {
        const float* p = (t % 2 ? mb : hb) + (unsigned long long)(t / 2) * H;
        float s = 0.0f;
        for (unsigned int e = lane; e < H; e += 32) s += p[e] * p[e];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        if (lane == 0) s_rstd[t / 2][t % 2] = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();
    const unsigned int r = blockIdx.x * INK_Z_WARPS + warp;
    if (r >= rank) return;
    const __nv_bfloat16* d = ink_pick(D0, D1, D2, g) + (unsigned long long)r * 2 * H;
    float acc[INK_Z_ROWS];
#pragma unroll
    for (int i = 0; i < INK_Z_ROWS; i++) acc[i] = 0.0f;
    // 8 inputs per lane step; H is a multiple of 8, so a chunk never straddles h and m
    for (unsigned int c = lane; c < 2 * H / 8; c += 32) {
        const unsigned int e = c * 8;
        const uint4 dv = *(const uint4*)(d + e);
        const unsigned int dr[4] = {dv.x, dv.y, dv.z, dv.w};
        float w[8];
#pragma unroll
        for (int i = 0; i < 8; i++)
            w[i] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(i % 2 ? dr[i / 2] >> 16 : dr[i / 2] & 0xFFFF)));
        const bool in_h = e < H;
#pragma unroll
        for (int q = 0; q < INK_Z_ROWS; q++) {
            if (q >= (int)nr) break;
            const float* src = (in_h ? hb + e : mb + (e - H)) + (unsigned long long)q * H;
            const float sc = s_rstd[q][in_h ? 0 : 1];
            const float4 a = *(const float4*)src, b = *(const float4*)(src + 4);
            const float in[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; i++) acc[q] += w[i] * __bfloat162float(__float2bfloat16(in[i] * sc));
        }
    }
#pragma unroll
    for (int q = 0; q < INK_Z_ROWS; q++) {
        if (q >= (int)nr) break;
        float v = acc[q];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) z[((unsigned long long)g * rows + row0 + q) * rank + r] = __bfloat162float(__float2bfloat16(v));
    }
}

// grid (ceil(H / 32), ceil(rows / INK_OUT_ROWS)), block 256: 8 lanes per channel, each
// holding 8 of its rank weights of every gate in registers across the block's rows
// (coalesced loads; the partial dots meet in shuffles).
extern "C" __global__ void inkling2_msg_gate_out(
    const float* __restrict__ h,
    const float* __restrict__ m,
    const __nv_bfloat16* __restrict__ F0,         // [H, rank] forget, per gate
    const __nv_bfloat16* __restrict__ F1,
    const __nv_bfloat16* __restrict__ F2,
    const __nv_bfloat16* __restrict__ G0,         // [H, rank] message, per gate
    const __nv_bfloat16* __restrict__ G1,
    const __nv_bfloat16* __restrict__ G2,
    const float* __restrict__ z,                  // [gates, rows, rank]
    float* out,                                   // may alias h
    const unsigned int rows,
    const unsigned int H,
    const unsigned int rank,
    const unsigned int gates,
    const unsigned long long h_stride,
    const unsigned long long m_stride,
    const unsigned long long out_stride
) {
    atlas_pdl_enter();
    const unsigned int row0 = blockIdx.y * INK_OUT_ROWS;
    const unsigned int nr = min(INK_OUT_ROWS, rows - row0);
    __shared__ float zs[INK_OUT_ROWS * INK_MAX_GATES * INK_MAX_RANK];
    for (unsigned int i = threadIdx.x; i < nr * gates * rank; i += blockDim.x) {
        const unsigned int q = i / (gates * rank), gi = i % (gates * rank);
        zs[i] = z[((unsigned long long)(gi / rank) * rows + row0 + q) * rank + gi % rank];
    }
    __syncthreads();
    const unsigned int sub = threadIdx.x % 8;                  // this lane's 8 of the rank
    const unsigned int e = blockIdx.x * 32 + threadIdx.x / 8;  // the channel
    const bool live = e < H && sub * 8 < rank;
    float fw[INK_MAX_GATES][8], gw[INK_MAX_GATES][8];
#pragma unroll
    for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
#pragma unroll
        for (int i = 0; i < 8; i++) fw[g][i] = gw[g][i] = 0.0f;
        if (g >= gates || !live) continue;
        const uint4 fv = *(const uint4*)(ink_pick(F0, F1, F2, g) + (unsigned long long)e * rank + sub * 8);
        const uint4 gv = *(const uint4*)(ink_pick(G0, G1, G2, g) + (unsigned long long)e * rank + sub * 8);
        const unsigned int fr[4] = {fv.x, fv.y, fv.z, fv.w}, gr[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
        for (int i = 0; i < 8; i++) {
            fw[g][i] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(i % 2 ? fr[i / 2] >> 16 : fr[i / 2] & 0xFFFF)));
            gw[g][i] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(i % 2 ? gr[i / 2] >> 16 : gr[i / 2] & 0xFFFF)));
        }
    }
    // every gate's inputs of a row are read before any of its outputs is written (out
    // may alias h), row by row
    for (unsigned int q = 0; q < nr; q++) {
        float res[INK_MAX_GATES];
        const unsigned long long at = (unsigned long long)(row0 + q) * H + e;
        // compile-time gate indices keep fw/gw/res in registers
#pragma unroll
        for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
            if (g >= gates) break;
            const float* zg = zs + (q * gates + g) * rank + sub * 8;
            float f = 0.0f, gg = 0.0f;
            if (live) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    f += fw[g][i] * zg[i];
                    gg += gw[g][i] * zg[i];
                }
            }
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                f += __shfl_xor_sync(0xffffffff, f, off);
                gg += __shfl_xor_sync(0xffffffff, gg, off);
            }
            res[g] = (sub == 0 && e < H) ? (1.0f - tanhf(f)) * h[g * h_stride + at] + tanhf(gg) * m[g * m_stride + at] : 0.0f;
        }
        if (sub == 0 && e < H) {
#pragma unroll
            for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
                if (g >= gates) break;
                out[g * out_stride + at] = res[g];
            }
        }
    }
}
