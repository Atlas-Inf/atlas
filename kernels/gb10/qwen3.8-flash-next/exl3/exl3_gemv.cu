// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 batch-1 dense GEMV on the packed trellis (ATLAS_EXL3_DENSE_NATIVE=1).
//
//   y = had_r128(had_r128(x * suh) . W_inner) * svh
//
// read straight from the packed 16x16 trellis tiles, no reconstruct and no
// BF16 copy of W. Structure follows the small-m GEMV of the exllamav3 fork
// (vcruz305/exllamav3 047ce72, exllamav3_ext/quant/exl3_gemv_kernel.cuh,
// MIT, Copyright (c) 2025 Turboderp; see exl3_vendor/NOTICE.md): warps split
// k and never synchronize in the main loop, the trellis streams to registers
// with ld.global.cs behind a prefetch ring, one m16n8k16 MMA pair per tile,
// per-block cross-warp reduction over the k splits. Upstream instantiates
// K = 2..4 only with lane-shuffle extraction; the dense linears of the 3.87
// pack are K8, so every K here uses upstream's smem-staged form
// (SMEM_STAGE = true): the tile words are staged through warp-private shared
// memory and decoded with the vendored dq_dispatch<K, cb = 2> — the same
// decode reconstruct_tile uses, so the fragments are the reconstruct's.
// Accumulation is fp32 (upstream folds fp16 partials every few tiles).
// The cooperative grid.sync of upstream is replaced by two launches:
//
//   exl3_gemv[_raw]_k<K>  block 512 (16 warps), grid <= n / 32 groups of 32
//                         columns (grid-stride over groups). dynamic smem =
//                         k*2 (fp16 x after the input transform)
//                         + 16 warps * 16K words (staged tiles)
//                         + 16 * 32 floats (reduction).
//                         Every block re-derives xh = had_r128(x * suh) into
//                         shared memory (bit-identical to exl3_had_r128_pre on
//                         the bf16 -> fp16 conversion of exl3_bf16_to_f16);
//                         the _raw form takes an already transformed fp16 xh
//                         (tests). Output: fp32 y_inner [n].
//   exl3_gemv_post_bf16   block 128 (4 x 128-wide Hadamard blocks), grid
//                         ceil(n / 512): out = bf16(had_r128(y_inner) * svh),
//                         fp32 until the single bf16 rounding; writes only
//                         the first n_valid columns (lm_head: vocab < padded).
//
// Requirements (checked by the Rust launcher): k % 128 == 0, n % 128 == 0,
// trellis 16-byte aligned, K in 2..8, mul1 codebook.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "exl3_vendor/exl3_dq.cuh"
#include "exl3_vendor/hadamard_inner.cuh"

#define EXL3_GEMV_WARPS 16
#define EXL3_GEMV_WNT 2                    // adjacent 16-column tiles per warp
#define EXL3_GEMV_COLS (EXL3_GEMV_WNT * 16) // columns per group
#define EXL3_GEMV_PF 4                     // prefetch ring depth (k-slices)

// m16n8k16, fp16 x fp16 + fp32 -> fp32. Only row 0 of A is live (batch 1):
// a_lo = x[k0 + 2q, +1], a_hi = x[k0 + 8 + 2q, +1] on lanes 0..3, zero else.
__device__ __forceinline__ void exl3_gemv_mma(uint32_t a_lo, uint32_t a_hi, const FragB& b, float (&c)[4])
{
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    const uint32_t z = 0;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a_lo), "r"(z), "r"(a_hi), "r"(z), "r"(bb[0]), "r"(bb[1]));
}

// xh = had_r128(fp16(x) * suh) into shared memory, one warp per 128-chunk.
// Same arithmetic, element order and roundings as exl3_bf16_to_f16 followed
// by had_hf_r_128_inner<true, false> (exl3_had_r128_pre).
template <bool RAW>
__device__ __forceinline__ void exl3_gemv_stage_x(const void* __restrict__ x, const half* __restrict__ suh,
                                                  half* sh_x, int k)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int c = warp; c < k / 128; c += EXL3_GEMV_WARPS)
    {
        half2* dst = reinterpret_cast<half2*>(sh_x) + c * 64 + lane * 2;
        if constexpr (RAW)
        {
            const half2* src = reinterpret_cast<const half2*>(x) + c * 64 + lane * 2;
            dst[0] = src[0];
            dst[1] = src[1];
        }
        else
        {
            const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(x) + c * 64 + lane * 2;
            const half2* sc = reinterpret_cast<const half2*>(suh) + c * 64 + lane * 2;
            const __nv_bfloat162 b01 = xb[0];
            const __nv_bfloat162 b23 = xb[1];
            half2 vx = __halves2half2(__float2half_rn(__low2float(b01)), __float2half_rn(__high2float(b01)));
            half2 vy = __halves2half2(__float2half_rn(__low2float(b23)), __float2half_rn(__high2float(b23)));
            vx = __hmul2(vx, sc[0]);
            vy = __hmul2(vy, sc[1]);
            float v0 = __half2float(__low2half(vx));
            float v1 = __half2float(__high2half(vx));
            float v2 = __half2float(__low2half(vy));
            float v3 = __half2float(__high2half(vy));
            float s0 = v0 + v1;
            float d0 = v0 - v1;
            float s1 = v2 + v3;
            float d1 = v2 - v3;
            float h0 = s0 + s1;
            float h1 = d0 + d1;
            float h2 = s0 - s1;
            float h3 = d0 - d1;
            shuffle_had_f4x32(h0, h1, h2, h3, lane);
            const float r = 0.088388347648f;
            dst[0] = __floats2half2_rn(h0 * r, h1 * r);
            dst[1] = __floats2half2_rn(h2 * r, h3 * r);
        }
    }
}

template <int K, bool RAW>
__device__ __forceinline__ void exl3_gemv_body(const void* __restrict__ x, const half* __restrict__ suh,
                                               const uint16_t* __restrict__ trellis, float* __restrict__ y,
                                               int k, int n, int n_act)
{
    constexpr int TW = 8 * K;                    // uint32 words per 16x16 tile
    constexpr int SW = EXL3_GEMV_WNT * TW;       // words per k-slice per warp
    constexpr int LANES = SW / 4;                // lanes loading one uint4 each
    static_assert(LANES <= 32, "one uint4 per lane per k-slice");

    extern __shared__ __align__(16) uint8_t exl3_gemv_smem[];
    half* sh_x = reinterpret_cast<half*>(exl3_gemv_smem);
    uint32_t* sh_stage = reinterpret_cast<uint32_t*>(exl3_gemv_smem + (size_t) k * 2);
    float* sh_red = reinterpret_cast<float*>(sh_stage + EXL3_GEMV_WARPS * SW);

    exl3_gemv_stage_x<RAW>(x, suh, sh_x, k);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ntiles = n / 16;
    const int kslices = k / 16;
    const int groups = n_act / EXL3_GEMV_COLS;  // leading n_act columns only
    const int chunk = (kslices + EXL3_GEMV_WARPS - 1) / EXL3_GEMV_WARPS;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_u4 = (size_t) ntiles * TW / 4;
    uint32_t* my_stage = sh_stage + warp * SW;
    const half2* x2 = reinterpret_cast<const half2*>(sh_x);

    for (int group = blockIdx.x; group < groups; group += gridDim.x)
    {
        const uint4* gp = reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint32_t*>(trellis) + ((size_t) ks0 * ntiles + (size_t) group * EXL3_GEMV_WNT) * TW) + lane;

        uint4 pf[EXL3_GEMV_PF];
        #pragma unroll
        for (int d = 0; d < EXL3_GEMV_PF; ++d)
        {
            pf[d] = make_uint4(0, 0, 0, 0);
            if (d < myn && lane < LANES) pf[d] = __ldcs(gp + (size_t) d * slice_u4);
        }

        float acc[EXL3_GEMV_WNT][2][4] = {};

        for (int ib = 0; ib < myn; ib += EXL3_GEMV_PF)
        {
            #pragma unroll
            for (int d = 0; d < EXL3_GEMV_PF; ++d)
            {
                const int i = ib + d;
                if (i >= myn) break;
                __syncwarp();
                if (lane < LANES) reinterpret_cast<uint4*>(my_stage)[lane] = pf[d];
                if (i + EXL3_GEMV_PF < myn && lane < LANES)
                    pf[d] = __ldcs(gp + (size_t) (i + EXL3_GEMV_PF) * slice_u4);
                __syncwarp();

                const int ks = ks0 + i;
                uint32_t a_lo = 0, a_hi = 0;
                if (lane < 4)
                {
                    half2 lo = x2[ks * 8 + lane];
                    half2 hi = x2[ks * 8 + 4 + lane];
                    a_lo = *reinterpret_cast<uint32_t*>(&lo);
                    a_hi = *reinterpret_cast<uint32_t*>(&hi);
                }

                #pragma unroll
                for (int t = 0; t < EXL3_GEMV_WNT; ++t)
                {
                    FragB f0, f1;
                    dq_dispatch<K, 2>(my_stage + t * TW, lane << 3, f0, f1);
                    exl3_gemv_mma(a_lo, a_hi, f0, acc[t][0]);
                    exl3_gemv_mma(a_lo, a_hi, f1, acc[t][1]);
                }
            }
        }

        // Row 0 of C lives in lanes 0..3: cols t*16 + f*8 + 2*lane (+1).
        float* red = sh_red + warp * EXL3_GEMV_COLS;
        if (lane < 4)
        {
            #pragma unroll
            for (int t = 0; t < EXL3_GEMV_WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    red[t * 16 + f * 8 + 2 * lane] = acc[t][f][0];
                    red[t * 16 + f * 8 + 2 * lane + 1] = acc[t][f][1];
                }
        }
        __syncthreads();
        if (threadIdx.x < EXL3_GEMV_COLS)
        {
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < EXL3_GEMV_WARPS; ++j) s += sh_red[j * EXL3_GEMV_COLS + threadIdx.x];
            y[group * EXL3_GEMV_COLS + threadIdx.x] = s;
        }
        __syncthreads();
    }
}

#define EXL3_GEMV_INST(K)                                                                                      \
    extern "C" __global__ __launch_bounds__(512, 1) void exl3_gemv_k##K(                                       \
        const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh, const uint16_t* __restrict__ trellis, \
        float* __restrict__ y, int k, int n, int n_act)                                                        \
    {                                                                                                          \
        exl3_gemv_body<K, false>(x, suh, trellis, y, k, n, n_act);                                                    \
    }                                                                                                          \
    extern "C" __global__ __launch_bounds__(512, 1) void exl3_gemv_raw_k##K(                                   \
        const half* __restrict__ x, const half* __restrict__ suh, const uint16_t* __restrict__ trellis,        \
        float* __restrict__ y, int k, int n, int n_act)                                                        \
    {                                                                                                          \
        exl3_gemv_body<K, true>(x, suh, trellis, y, k, n, n_act);                                                     \
    }

EXL3_GEMV_INST(2)
EXL3_GEMV_INST(3)
EXL3_GEMV_INST(4)
EXL3_GEMV_INST(5)
EXL3_GEMV_INST(6)
EXL3_GEMV_INST(7)
EXL3_GEMV_INST(8)

// out[c] = bf16(had_r128(y)[c] * svh[c]) for c < n_valid. One warp per
// 128-wide block, same butterfly/element order as had_hf_r_128_inner, fp32
// throughout (y_inner is fp32), one rounding at the store.
extern "C" __global__ __launch_bounds__(128) void exl3_gemv_post_bf16(
    const float* __restrict__ y, __nv_bfloat16* __restrict__ out, const half* __restrict__ svh, int n, int n_valid)
{
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= n / 128) return;  // warp-uniform
    const float4 v = reinterpret_cast<const float4*>(y)[c * 32 + lane];
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float r = 0.088388347648f;
    const half2* sv = reinterpret_cast<const half2*>(svh) + c * 64 + lane * 2;
    const half2 sa = sv[0];
    const half2 sb = sv[1];
    const float o[4] = {
        h0 * r * __low2float(sa), h1 * r * __high2float(sa),
        h2 * r * __low2float(sb), h3 * r * __high2float(sb)};
    const int base = c * 128 + lane * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (base + i < n_valid) out[base + i] = __float2bfloat16_rn(o[i]);
}

// ---------------------------------------------------------------------------
// M rows (MTP verify, m = 2..8): the same trellis stream feeds every row.
//
// m16n8k16 A-fragment row g = lane >> 2 carries activation row g (rows 8..15
// stay zero), so up to 8 rows ride one MMA with the trellis read once. Per
// row the k-split, prefetch order, MMA sequence and the cross-warp reduction
// order are those of exl3_gemv_k<K>, and the activation transform is the
// same arithmetic as exl3_gemv_stage_x, so row r matches a batch-1 call on
// row r. The transformed activations do not fit shared memory for m rows at
// k = 6144 under 48 KiB, so they come from a pre-kernel in global memory:
//
//   exl3_gemv_xh_rows     grid (ceil(k/128/4), m), block 128: xh[r] =
//                         had_r128(fp16(x[r]) * suh)  (fp16 [m, k])
//   exl3_gemv_m_k<K>      grid <= n/32, block 512, dynamic smem =
//                         16 warps * SW words + 16 * 8 * 32 floats.
//                         y[r * y_stride + c] (fp32)
//   exl3_gemv_post_rows_bf16  grid (ceil(n/512), m), block 128:
//                         out[r * out_stride + c] = bf16(had_r128(y[r]) * svh)

extern "C" __global__ __launch_bounds__(128) void exl3_gemv_xh_rows(
    const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh, half* __restrict__ xh, int k, int x_stride)
{
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int r = blockIdx.y;
    if (c >= k / 128) return;  // warp-uniform
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(x + (size_t) r * x_stride) + c * 64 + lane * 2;
    const half2* sc = reinterpret_cast<const half2*>(suh) + c * 64 + lane * 2;
    half2* dst = reinterpret_cast<half2*>(xh + (size_t) r * k) + c * 64 + lane * 2;
    const __nv_bfloat162 b01 = xb[0];
    const __nv_bfloat162 b23 = xb[1];
    half2 vx = __halves2half2(__float2half_rn(__low2float(b01)), __float2half_rn(__high2float(b01)));
    half2 vy = __halves2half2(__float2half_rn(__low2float(b23)), __float2half_rn(__high2float(b23)));
    vx = __hmul2(vx, sc[0]);
    vy = __hmul2(vy, sc[1]);
    float v0 = __half2float(__low2half(vx));
    float v1 = __half2float(__high2half(vx));
    float v2 = __half2float(__low2half(vy));
    float v3 = __half2float(__high2half(vy));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float rr = 0.088388347648f;
    dst[0] = __floats2half2_rn(h0 * rr, h1 * rr);
    dst[1] = __floats2half2_rn(h2 * rr, h3 * rr);
}

#define EXL3_GEMV_MROWS 8

template <int K>
__device__ __forceinline__ void exl3_gemv_m_body(const half* __restrict__ xh, const uint16_t* __restrict__ trellis,
                                                 float* __restrict__ y, int k, int n, int m, int y_stride)
{
    constexpr int TW = 8 * K;
    constexpr int SW = EXL3_GEMV_WNT * TW;
    constexpr int LANES = SW / 4;
    static_assert(LANES <= 32, "one uint4 per lane per k-slice");

    extern __shared__ __align__(16) uint8_t exl3_gemv_m_smem[];
    uint32_t* sh_stage = reinterpret_cast<uint32_t*>(exl3_gemv_m_smem);
    float* sh_red = reinterpret_cast<float*>(sh_stage + EXL3_GEMV_WARPS * SW);  // [warp][row][col]

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int q = lane & 3;
    const bool live = g < m;
    const int ntiles = n / 16;
    const int kslices = k / 16;
    const int groups = n / EXL3_GEMV_COLS;
    const int chunk = (kslices + EXL3_GEMV_WARPS - 1) / EXL3_GEMV_WARPS;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_u4 = (size_t) ntiles * TW / 4;
    uint32_t* my_stage = sh_stage + warp * SW;
    const half2* x2 = reinterpret_cast<const half2*>(xh + (size_t) (live ? g : 0) * k);

    for (int group = blockIdx.x; group < groups; group += gridDim.x)
    {
        const uint4* gp = reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint32_t*>(trellis) + ((size_t) ks0 * ntiles + (size_t) group * EXL3_GEMV_WNT) * TW) + lane;

        uint4 pf[EXL3_GEMV_PF];
        #pragma unroll
        for (int d = 0; d < EXL3_GEMV_PF; ++d)
        {
            pf[d] = make_uint4(0, 0, 0, 0);
            if (d < myn && lane < LANES) pf[d] = __ldcs(gp + (size_t) d * slice_u4);
        }

        float acc[EXL3_GEMV_WNT][2][4] = {};

        for (int ib = 0; ib < myn; ib += EXL3_GEMV_PF)
        {
            #pragma unroll
            for (int d = 0; d < EXL3_GEMV_PF; ++d)
            {
                const int i = ib + d;
                if (i >= myn) break;
                __syncwarp();
                if (lane < LANES) reinterpret_cast<uint4*>(my_stage)[lane] = pf[d];
                if (i + EXL3_GEMV_PF < myn && lane < LANES)
                    pf[d] = __ldcs(gp + (size_t) (i + EXL3_GEMV_PF) * slice_u4);
                __syncwarp();

                const int ks = ks0 + i;
                uint32_t a_lo = 0, a_hi = 0;
                if (live)
                {
                    half2 lo = x2[ks * 8 + q];
                    half2 hi = x2[ks * 8 + 4 + q];
                    a_lo = *reinterpret_cast<uint32_t*>(&lo);
                    a_hi = *reinterpret_cast<uint32_t*>(&hi);
                }

                #pragma unroll
                for (int t = 0; t < EXL3_GEMV_WNT; ++t)
                {
                    FragB f0, f1;
                    dq_dispatch<K, 2>(my_stage + t * TW, lane << 3, f0, f1);
                    exl3_gemv_mma(a_lo, a_hi, f0, acc[t][0]);
                    exl3_gemv_mma(a_lo, a_hi, f1, acc[t][1]);
                }
            }
        }

        // Row g of C lives in lanes 4g..4g+3: cols t*16 + f*8 + 2q (+1).
        float* red = sh_red + (warp * EXL3_GEMV_MROWS + g) * EXL3_GEMV_COLS;
        if (live)
        {
            #pragma unroll
            for (int t = 0; t < EXL3_GEMV_WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    red[t * 16 + f * 8 + 2 * q] = acc[t][f][0];
                    red[t * 16 + f * 8 + 2 * q + 1] = acc[t][f][1];
                }
        }
        __syncthreads();
        if ((int) threadIdx.x < m * EXL3_GEMV_COLS)
        {
            const int r = threadIdx.x / EXL3_GEMV_COLS;
            const int c = threadIdx.x % EXL3_GEMV_COLS;
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < EXL3_GEMV_WARPS; ++j)
                s += sh_red[(j * EXL3_GEMV_MROWS + r) * EXL3_GEMV_COLS + c];
            y[(size_t) r * y_stride + group * EXL3_GEMV_COLS + c] = s;
        }
        __syncthreads();
    }
}

#define EXL3_GEMV_M_INST(K)                                                                                   \
    extern "C" __global__ __launch_bounds__(512, 1) void exl3_gemv_m_k##K(                                    \
        const half* __restrict__ xh, const uint16_t* __restrict__ trellis, float* __restrict__ y, int k, int n, \
        int m, int y_stride)                                                                                  \
    {                                                                                                         \
        exl3_gemv_m_body<K>(xh, trellis, y, k, n, m, y_stride);                                               \
    }

EXL3_GEMV_M_INST(2)
EXL3_GEMV_M_INST(3)
EXL3_GEMV_M_INST(4)
EXL3_GEMV_M_INST(5)
EXL3_GEMV_M_INST(6)
EXL3_GEMV_M_INST(7)
EXL3_GEMV_M_INST(8)

// Row r = blockIdx.y of exl3_gemv_post_bf16 (same arithmetic), strided in
// both y and out.
extern "C" __global__ __launch_bounds__(128) void exl3_gemv_post_rows_bf16(
    const float* __restrict__ y, __nv_bfloat16* __restrict__ out, const half* __restrict__ svh, int n, int n_valid,
    int y_stride, int out_stride)
{
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= n / 128) return;  // warp-uniform
    y += (size_t) blockIdx.y * y_stride;
    out += (size_t) blockIdx.y * out_stride;
    const float4 v = reinterpret_cast<const float4*>(y)[c * 32 + lane];
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float r = 0.088388347648f;
    const half2* sv = reinterpret_cast<const half2*>(svh) + c * 64 + lane * 2;
    const half2 sa = sv[0];
    const half2 sb = sv[1];
    const float o[4] = {
        h0 * r * __low2float(sa), h1 * r * __high2float(sa),
        h2 * r * __low2float(sb), h3 * r * __high2float(sb)};
    const int base = c * 128 + lane * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (base + i < n_valid) out[base + i] = __float2bfloat16_rn(o[i]);
}
