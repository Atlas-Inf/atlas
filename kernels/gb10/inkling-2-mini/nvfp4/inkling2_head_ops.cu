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

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip, gfx1151): the fused message gates below on WMMA
// (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32; fragment layout as kernels/strix-hip validates it:
// A lane l -> a[i] = A[l & 15][k + i], B (N x K rows) b[i] = B[n + (l & 15)][k + i],
// C element e -> row 2e + (l >> 4), column l & 15). Same phases, arguments and roundings as the
// gb10 kernel after this block: rms of every gate's h and m rows (their producer's sums of squares); z = BF16(D BF16([rms0(h); rms0(m)]))
// on warps 0-3 (one 16-wide rank tile each); then 64 channels a step, F z on warps 0-3 and G z on
// warps 4-7 staged in fp32, and out = (1 - tanh(F z)) * h + tanh(G z) * m with every gate's inputs
// read before its outputs are written (out may alias h). Rank 64. grid (ceil(rows / 16)), block 256.
#define INK_FG_ROWS 16
#define INK_FG_K 64
#define INK_FG_CH 64
#define INK_FG_PITCH (INK_FG_K + 8)
typedef __bf16 ink_fg_v16bf __attribute__((ext_vector_type(16)));
typedef float ink_fg_v8f __attribute__((ext_vector_type(8)));
// 16 consecutive BF16 of a 16-byte-aligned shared row as one WMMA operand (two ds_read_b128).
__device__ __forceinline__ ink_fg_v16bf ink_fg_ld16(const __nv_bfloat16* p) {
    union { uint4 u[2]; ink_fg_v16bf v; } x;
    x.u[0] = ((const uint4*)p)[0];
    x.u[1] = ((const uint4*)p)[1];
    return x.v;
}

extern "C" __global__ void __launch_bounds__(256) inkling2_msg_gate_fused(
    const float* h, const float* m,
    const float* __restrict__ sq_h, const float* __restrict__ sq_m,   // rows' sums of squares
    const __nv_bfloat16* __restrict__ D0, const __nv_bfloat16* __restrict__ D1, const __nv_bfloat16* __restrict__ D2,
    const __nv_bfloat16* __restrict__ F0, const __nv_bfloat16* __restrict__ F1, const __nv_bfloat16* __restrict__ F2,
    const __nv_bfloat16* __restrict__ G0, const __nv_bfloat16* __restrict__ G1, const __nv_bfloat16* __restrict__ G2,
    float* out,
    const unsigned int rows,
    const unsigned int H,
    const unsigned int gates,
    const unsigned long long h_stride,
    const unsigned long long m_stride,
    const unsigned long long out_stride,
    const float eps
) {
    atlas_pdl_enter();
    constexpr unsigned int RANK = INK_MAX_RANK;
    __shared__ float s_rstd[INK_MAX_GATES][INK_FG_ROWS][2];
    __shared__ __align__(16) __nv_bfloat16 s_a[INK_FG_ROWS][INK_FG_PITCH];
    __shared__ __align__(16) __nv_bfloat16 s_w[RANK][INK_FG_PITCH];      // D (z) or F (out) step
    __shared__ __align__(16) __nv_bfloat16 s_w2[INK_FG_CH][INK_FG_PITCH]; // G (out) step
    __shared__ __align__(16) __nv_bfloat16 s_z[INK_MAX_GATES][INK_FG_ROWS][INK_FG_PITCH];
    __shared__ float s_f[INK_MAX_GATES][INK_FG_ROWS][INK_FG_CH];
    __shared__ float s_g[INK_MAX_GATES][INK_FG_ROWS][INK_FG_CH];
    const unsigned int row0 = blockIdx.x * INK_FG_ROWS, nr = min(INK_FG_ROWS, rows - row0);
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int l16 = lane & 15, lhi = lane >> 4;

    // rms of every gate's h and m rows from the sums of squares their producer wrote
    // (inkling2_block_res_row), planes `stride / H` rows apart
    for (unsigned int t = tid; t < gates * INK_FG_ROWS * 2; t += 256) {
        const unsigned int g = t / (2 * INK_FG_ROWS), q = (t / 2) % INK_FG_ROWS, half = t % 2;
        float s = 0.0f;
        if (q < nr) s = half ? sq_m[g * (m_stride / H) + row0 + q] : sq_h[g * (h_stride / H) + row0 + q];
        s_rstd[g][q][half] = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();

    for (unsigned int g = 0; g < gates; g++) {
        const float* hg = h + g * h_stride;
        const float* mg = m + g * m_stride;
        const __nv_bfloat16* d = ink_pick(D0, D1, D2, g);
        ink_fg_v8f acc = ink_fg_v8f{0, 0, 0, 0, 0, 0, 0, 0};
        const unsigned int ar = tid / 16, ac = (tid % 16) * 4;
        for (unsigned int k0 = 0; k0 < 2 * H; k0 += INK_FG_K) {
            const bool in_h = k0 < H;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (ar < nr) v = *(const float4*)((in_h ? hg + k0 : mg + (k0 - H)) + (unsigned long long)(row0 + ar) * H + ac);
            const float sc = s_rstd[g][ar][in_h ? 0 : 1];
            s_a[ar][ac] = __float2bfloat16(v.x * sc);
            s_a[ar][ac + 1] = __float2bfloat16(v.y * sc);
            s_a[ar][ac + 2] = __float2bfloat16(v.z * sc);
            s_a[ar][ac + 3] = __float2bfloat16(v.w * sc);
            for (unsigned int i = tid; i < RANK * INK_FG_K / 8; i += 256) {
                const unsigned int r = i / (INK_FG_K / 8), c = (i % (INK_FG_K / 8)) * 8;
                *(uint4*)&s_w[r][c] = *(const uint4*)(d + (unsigned long long)r * 2 * H + k0 + c);
            }
            __syncthreads();
            if (warp < RANK / 16) {
#pragma unroll
                for (unsigned int kk = 0; kk < INK_FG_K; kk += 16)
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ink_fg_ld16(&s_a[l16][kk]),
                                                                      ink_fg_ld16(&s_w[16 * warp + l16][kk]), acc);
            }
            __syncthreads();
        }
        if (warp < RANK / 16) {
#pragma unroll
            for (int e = 0; e < 8; e++) s_z[g][2 * e + lhi][16 * warp + l16] = __float2bfloat16(acc[e]);
        }
    }
    __syncthreads();

    for (unsigned int c0 = 0; c0 < H; c0 += INK_FG_CH) {
        for (unsigned int g = 0; g < gates; g++) {
            const __nv_bfloat16* f = ink_pick(F0, F1, F2, g);
            const __nv_bfloat16* gw = ink_pick(G0, G1, G2, g);
            for (unsigned int i = tid; i < INK_FG_CH * RANK / 8; i += 256) {
                const unsigned int r = i / (RANK / 8), c = (i % (RANK / 8)) * 8;
                *(uint4*)&s_w[r][c] = *(const uint4*)(f + (unsigned long long)(c0 + r) * RANK + c);
                *(uint4*)&s_w2[r][c] = *(const uint4*)(gw + (unsigned long long)(c0 + r) * RANK + c);
            }
            __syncthreads();
            // warps 0-3: F tile (warp % 4); warps 4-7: G tile (warp % 4)
            const __nv_bfloat16 (*wt)[INK_FG_PITCH] = warp < 4 ? s_w : s_w2;
            const unsigned int nb = 16 * (warp % 4);
            ink_fg_v8f acc = ink_fg_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (unsigned int kk = 0; kk < RANK; kk += 16)
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ink_fg_ld16(&s_z[g][l16][kk]), ink_fg_ld16(&wt[nb + l16][kk]), acc);
            float (*dst)[INK_FG_CH] = warp < 4 ? s_f[g] : s_g[g];
#pragma unroll
            for (int e = 0; e < 8; e++) dst[2 * e + lhi][nb + l16] = acc[e];
            __syncthreads();
        }
        // 16 rows x 64 channels: 4 elements a thread, every gate's inputs read before any write
        float res[INK_MAX_GATES][4];
#pragma unroll
        for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
            if (g >= gates) break;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int el = tid + 256 * j, q = el / INK_FG_CH, ch = el % INK_FG_CH;
                res[g][j] = 0.0f;
                if (q >= nr) continue;
                const unsigned long long at = (unsigned long long)(row0 + q) * H + c0 + ch;
                res[g][j] = (1.0f - tanhf(s_f[g][q][ch])) * h[g * h_stride + at] + tanhf(s_g[g][q][ch]) * m[g * m_stride + at];
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
            if (g >= gates) break;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int el = tid + 256 * j, q = el / INK_FG_CH, ch = el % INK_FG_CH;
                if (q < nr) out[g * out_stride + (unsigned long long)(row0 + q) * H + c0 + ch] = res[g][j];
            }
        }
    }
}
#else
// The same message gates for prefill rows, fused on the tensor cores: a CTA takes 16 rows
// and every gate of the launch. Per gate, the rms of its h and m rows, then
// z = BF16(D BF16([rms0(h); rms0(m)])) (mma.sync m16n8k16, the BF16 inputs staged 64
// columns a step, each step's loads in flight over the step before); then, 64 channels a
// step, F z and G z for every gate and
// out = (1 - tanh(F z)) * h + tanh(G z) * m, each element's inputs for all gates read
// before any of its outputs is written (out may alias h). The rms come from the sums of
// squares inkling2_block_res_row wrote with the rows, so a row is read twice (z, out); the
// two-kernel path re-reads every row once per four rank outputs.
// Rank 64. grid (ceil(rows / 16)), block 256.
#define INK_FG_ROWS 16
#define INK_FG_K 64
#define INK_FG_CH 64
#define INK_FG_PITCH (INK_FG_K + 8)   // BF16 pitch: conflict-free fragment loads

__device__ __forceinline__ void ink_mma16816(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ink_frag_a(const __nv_bfloat16 (*s)[INK_FG_PITCH], unsigned int kk, unsigned int g8,
                                           unsigned int t2, unsigned int (&a)[4]) {
    a[0] = *(const unsigned int*)&s[g8][kk + t2];
    a[1] = *(const unsigned int*)&s[g8 + 8][kk + t2];
    a[2] = *(const unsigned int*)&s[g8][kk + t2 + 8];
    a[3] = *(const unsigned int*)&s[g8 + 8][kk + t2 + 8];
}

extern "C" __global__ void __launch_bounds__(256) inkling2_msg_gate_fused(
    const float* h, const float* m,
    const float* __restrict__ sq_h, const float* __restrict__ sq_m,   // rows' sums of squares
    const __nv_bfloat16* __restrict__ D0, const __nv_bfloat16* __restrict__ D1, const __nv_bfloat16* __restrict__ D2,
    const __nv_bfloat16* __restrict__ F0, const __nv_bfloat16* __restrict__ F1, const __nv_bfloat16* __restrict__ F2,
    const __nv_bfloat16* __restrict__ G0, const __nv_bfloat16* __restrict__ G1, const __nv_bfloat16* __restrict__ G2,
    float* out,
    const unsigned int rows,
    const unsigned int H,
    const unsigned int gates,
    const unsigned long long h_stride,
    const unsigned long long m_stride,
    const unsigned long long out_stride,
    const float eps
) {
    atlas_pdl_enter();
    constexpr unsigned int RANK = INK_MAX_RANK;
    __shared__ float s_rstd[INK_MAX_GATES][INK_FG_ROWS][2];
    __shared__ __align__(16) __nv_bfloat16 s_a[2][INK_FG_ROWS][INK_FG_PITCH];
    static_assert(RANK * INK_FG_K / 8 == 512 && INK_FG_CH * RANK / 8 == 512, "a weight stage: two 16-byte loads a thread");
    // weight stages, two buffers each: D (z: 0, 1), then F (0, 1) and G (2, 3)
    __shared__ __align__(16) __nv_bfloat16 s_w[4][RANK][INK_FG_PITCH];
    __shared__ __align__(16) __nv_bfloat16 s_z[INK_MAX_GATES][INK_FG_ROWS][INK_FG_PITCH];
    const unsigned int row0 = blockIdx.x * INK_FG_ROWS, nr = min(INK_FG_ROWS, rows - row0);
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int g8 = lane / 4, t2 = (lane % 4) * 2;

    // rms of every gate's h and m rows from the sums of squares their producer wrote
    // (inkling2_block_res_row), planes `stride / H` rows apart
    for (unsigned int t = tid; t < gates * INK_FG_ROWS * 2; t += 256) {
        const unsigned int g = t / (2 * INK_FG_ROWS), q = (t / 2) % INK_FG_ROWS, half = t % 2;
        float s = 0.0f;
        if (q < nr) s = half ? sq_m[g * (m_stride / H) + row0 + q] : sq_h[g * (h_stride / H) + row0 + q];
        s_rstd[g][q][half] = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();

    // z per gate: warp w owns rank columns 8w .. 8w + 7; each step's inputs load into
    // registers while the step before runs its MMAs (one barrier a step)
    const unsigned int ar = tid / 16, ac = (tid % 16) * 4;   // this thread's 4 inputs a step
    for (unsigned int g = 0; g < gates; g++) {
        const float* hg = h + g * h_stride;
        const float* mg = m + g * m_stride;
        const __nv_bfloat16* d = ink_pick(D0, D1, D2, g);
        float4 v;
        uint4 dw[2];
        auto fetch = [&](unsigned int k0) {
            v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (ar < nr) v = *(const float4*)((k0 < H ? hg + k0 : mg + (k0 - H)) + (unsigned long long)(row0 + ar) * H + ac);
#pragma unroll
            for (unsigned int j = 0; j < 2; j++) {
                const unsigned int i = tid + 256 * j, r = i / (INK_FG_K / 8), c = (i % (INK_FG_K / 8)) * 8;
                dw[j] = *(const uint4*)(d + (unsigned long long)r * 2 * H + k0 + c);
            }
        };
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        fetch(0);
        for (unsigned int k0 = 0, buf = 0; k0 < 2 * H; k0 += INK_FG_K, buf ^= 1) {
            const float sc = s_rstd[g][ar][k0 < H ? 0 : 1];
            *(__nv_bfloat162*)&s_a[buf][ar][ac] = __floats2bfloat162_rn(v.x * sc, v.y * sc);
            *(__nv_bfloat162*)&s_a[buf][ar][ac + 2] = __floats2bfloat162_rn(v.z * sc, v.w * sc);
#pragma unroll
            for (unsigned int j = 0; j < 2; j++) {
                const unsigned int i = tid + 256 * j, r = i / (INK_FG_K / 8), c = (i % (INK_FG_K / 8)) * 8;
                *(uint4*)&s_w[buf][r][c] = dw[j];
            }
            if (k0 + INK_FG_K < 2 * H) fetch(k0 + INK_FG_K);
            __syncthreads();
#pragma unroll
            for (unsigned int kk = 0; kk < INK_FG_K; kk += 16) {
                unsigned int a[4];
                ink_frag_a(s_a[buf], kk, g8, t2, a);
                const __nv_bfloat16* b = &s_w[buf][8 * warp + g8][kk + t2];
                ink_mma16816(acc, a, *(const unsigned int*)b, *(const unsigned int*)(b + 8));
            }
        }
        *(__nv_bfloat162*)&s_z[g][g8][8 * warp + t2] = __floats2bfloat162_rn(acc[0], acc[1]);
        *(__nv_bfloat162*)&s_z[g][g8 + 8][8 * warp + t2] = __floats2bfloat162_rn(acc[2], acc[3]);
    }
    __syncthreads();

    // outputs, 64 channels a step: warp w owns channels c0 + 8w .. + 7. Step st takes
    // channels (st / gates) * 64 of gate st % gates; its F and G tiles load while the step
    // before runs, and its h and m elements load before its barrier.
    const unsigned int steps = (H / INK_FG_CH) * gates;
    uint4 fw[2], gw[2];
    auto fetch_fg = [&](unsigned int st) {
        const unsigned int c0 = (st / gates) * INK_FG_CH, g = st % gates;
        const __nv_bfloat16* f = ink_pick(F0, F1, F2, g);
        const __nv_bfloat16* gg = ink_pick(G0, G1, G2, g);
#pragma unroll
        for (unsigned int j = 0; j < 2; j++) {
            const unsigned int i = tid + 256 * j, r = i / (RANK / 8), c = (i % (RANK / 8)) * 8;
            fw[j] = *(const uint4*)(f + (unsigned long long)(c0 + r) * RANK + c);
            gw[j] = *(const uint4*)(gg + (unsigned long long)(c0 + r) * RANK + c);
        }
    };
    fetch_fg(0);
    for (unsigned int c0 = 0, st = 0; c0 < H; c0 += INK_FG_CH) {
        const unsigned int ch = c0 + 8 * warp + t2;
        float res[INK_MAX_GATES][4];
#pragma unroll
        for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
            if (g >= gates) break;
            const unsigned int buf = st & 1;
#pragma unroll
            for (unsigned int j = 0; j < 2; j++) {
                const unsigned int i = tid + 256 * j, r = i / (RANK / 8), c = (i % (RANK / 8)) * 8;
                *(uint4*)&s_w[buf][r][c] = fw[j];
                *(uint4*)&s_w[2 + buf][r][c] = gw[j];
            }
            if (st + 1 < steps) fetch_fg(st + 1);
            float hv[4], mv[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int q = g8 + 8 * (j / 2);
                const unsigned long long at = (unsigned long long)(row0 + q) * H + ch + (j % 2);
                hv[j] = q < nr ? h[g * h_stride + at] : 0.0f;
                mv[j] = q < nr ? m[g * m_stride + at] : 0.0f;
            }
            __syncthreads();
            float fa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, ga[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (unsigned int kk = 0; kk < RANK; kk += 16) {
                unsigned int a[4];
                ink_frag_a(s_z[g], kk, g8, t2, a);
                const __nv_bfloat16* bf = &s_w[buf][8 * warp + g8][kk + t2];
                const __nv_bfloat16* bg = &s_w[2 + buf][8 * warp + g8][kk + t2];
                ink_mma16816(fa, a, *(const unsigned int*)bf, *(const unsigned int*)(bf + 8));
                ink_mma16816(ga, a, *(const unsigned int*)bg, *(const unsigned int*)(bg + 8));
            }
#pragma unroll
            for (int j = 0; j < 4; j++) res[g][j] = (1.0f - tanhf(fa[j])) * hv[j] + tanhf(ga[j]) * mv[j];
            st++;
        }
#pragma unroll
        for (unsigned int g = 0; g < INK_MAX_GATES; g++) {
            if (g >= gates) break;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int q = g8 + 8 * (j / 2);
                if (q < nr) out[g * out_stride + (unsigned long long)(row0 + q) * H + ch + (j % 2)] = res[g][j];
            }
        }
    }
}
#endif  // AMD
