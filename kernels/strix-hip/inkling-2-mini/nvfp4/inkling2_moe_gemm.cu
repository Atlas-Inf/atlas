// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 grouped expert GEMMs for prefill rows, strix-hip (gfx1151 WMMA) port of
// the BF16-activation grouped GEMM that kernels/gb10/inkling-2-mini/nvfp4/inkling2_moe_gemm.cu
// carried until #401's cb10558e6 moved gb10 prefill to W4A4 / W8A8 only (same entry points,
// arguments, grids and numerics): C[m, n] = sum_k A[row(m), k] * W_e[n, k] for the rows m
// of each expert e (moe_sort_by_expert order), W_e read from the checkpoint's stacked
// tensors and dequantised per 32-wide K step to BF16 exactly as the gb10 tile does:
//   NVFP4  BF16(e2m1 * (E4M3(scale) * s2[e]))      codes [E, N, K/2], scales [E, N, K/16]
//   MXFP8  BF16(E4M3(code) * 2^(E8M0 - 127))       codes [E, N, K],   scales [E, N, K/32]
// E4M3 decodes by bit math (the AMD compat __nv_fp8_e4m3 decode is not the standard format).
// The product runs on __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 with the fragment layout
// validated in kernels/strix-hip (w4a16_wmma_ref.hip; qwen3.6-27b moe_w4a16_grouped_gemm.cu):
//   A: lane l -> a[i] = A_s[m + (l & 15)][i];  B (N x K rows): b[k] = B_s[n + (l & 15)][k];
//   C: lane l, element e -> C[m + 2e + (l >> 4)][n + (l & 15)].
// Tile TM rows (128 NVFP4, 64 MXFP8, as gb10's GM_FP4 / GM) x 128 outputs, 256 threads:
// 8 warps as (TM / 16) x (8 / (TM / 16)), K step 32, loads synchronous (no cp.async on HIP).
// The W4A4 / W8A8 entries (block-scaled MMA on gb10) and their tile list are trap stubs here:
// spark-model takes the BF16-row path under cfg!(atlas_scale).
// grouped grid (N / 128, ceil(max rows of an expert / TM), E), block 256.

#include <cuda_bf16.h>

typedef __bf16 ink_v16bf __attribute__((ext_vector_type(16)));
typedef float ink_v8f __attribute__((ext_vector_type(8)));

#define INK_GN 128
#define INK_GK 32
#define INK_PITCH (INK_GK + 8)

__device__ __constant__ float INK_HIP_E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// 16 consecutive BF16 of a 16-byte-aligned shared row as one WMMA operand (two ds_read_b128).
__device__ __forceinline__ ink_v16bf ink_ld16(const __nv_bfloat16* p) {
    union { uint4 u[2]; ink_v16bf v; } x;
    x.u[0] = ((const uint4*)p)[0];
    x.u[1] = ((const uint4*)p)[1];
    return x.v;
}

__device__ __forceinline__ float ink_hip_e4m3(unsigned int b) {
    const unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u) v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

template <bool MX, int TM, bool SW>
__device__ __forceinline__ void ink_hip_gemm_tile(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wc,
    const unsigned char* __restrict__ ws, const float g, __nv_bfloat16* __restrict__ C,
    const int m_start, const int m_count, const int m0, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K, const float cg, const float cu
) {
    constexpr int WM = TM / 16;            // warps along M
    constexpr int WN = 8 / WM;             // warps along N
    constexpr int NSUB = INK_GN / WN / 16; // 16-wide output sub-tiles per warp
    __shared__ __align__(16) __nv_bfloat16 As[TM][INK_PITCH];
    __shared__ __align__(16) __nv_bfloat16 Bs[INK_GN][INK_PITCH];
    __shared__ float lut[16];
    __shared__ int src[TM];
    const unsigned int n0 = blockIdx.x * INK_GN;
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    if (tid < 16) lut[tid] = INK_HIP_E2M1[tid];
    for (int r = tid; r < TM; r += 256) src[r] = m0 + r < m_count ? (rows ? rows[m_start + m0 + r] : m_start + m0 + r) : -1;
    __syncthreads();

    const int wm = (int)(warp % WM) * 16, wn = (int)(warp / WM) * (NSUB * 16);
    ink_v8f acc[NSUB];
#pragma unroll
    for (int j = 0; j < NSUB; j++) acc[j] = ink_v8f{0, 0, 0, 0, 0, 0, 0, 0};

    // Per-thread loads fixed across the K loop: A chunk it of A_ITERS 16-byte
    // chunks (row r, column (idx%4)*8), and weight row bn half bh (16 values,
    // one scale per K step). The next K32's global loads are issued into
    // registers under this step's WMMAs (prefetch below), so the dequant and the
    // stores stay the only work between the two barriers.
    constexpr int A_ITERS = TM * 4 / 256;
    const __nv_bfloat16* a_ptr[A_ITERS];
    int a_off[A_ITERS];
    bool a_val[A_ITERS];
#pragma unroll
    for (int it = 0; it < A_ITERS; it++) {
        const int idx = it * 256 + (int)tid, r = idx / 4, col = (idx % 4) * 8;
        const int sr = src[r];
        a_val[it] = sr >= 0;
        a_ptr[it] = A + (unsigned long long)sr * K + col;
        a_off[it] = r * INK_PITCH + col;
    }
    const unsigned int br = tid / 2, bh = tid % 2, bn = n0 + br;
    const bool b_ok = bn < N;
    const unsigned char* ws_row = ws + (unsigned long long)bn * (MX ? K / 32 : K / 16);
    const unsigned char* wc_row = wc + (unsigned long long)bn * (MX ? K : K / 2) + bh * (MX ? 16 : 8);

    uint4 reg_a[A_ITERS];
    uint4 reg_q;
    unsigned int reg_s = 0;
    auto prefetch = [&](const unsigned int k0) {
#pragma unroll
        for (int it = 0; it < A_ITERS; it++)
            reg_a[it] = a_val[it] ? *(const uint4*)(a_ptr[it] + k0) : make_uint4(0, 0, 0, 0);
        if (b_ok) {
            reg_s = MX ? ws_row[k0 / 32] : ws_row[k0 / 16 + bh];
            if (MX) reg_q = *(const uint4*)(wc_row + k0);
            else {
                const uint2 q = *(const uint2*)(wc_row + k0 / 2);
                reg_q.x = q.x; reg_q.y = q.y;
            }
        }
    };

    prefetch(0);
    for (unsigned int k0 = 0; k0 < K; k0 += INK_GK) {
#pragma unroll
        for (int it = 0; it < A_ITERS; it++)
            *(uint4*)((__nv_bfloat16*)As + a_off[it]) = reg_a[it];
        // weights: thread t dequantises row t / 2, half t % 2 (16 values, one NVFP4 scale)
        {
            float v[16];
            if (b_ok) {
                const unsigned int w[4] = {reg_q.x, reg_q.y, reg_q.z, reg_q.w};
                if (MX) {
                    const float s = exp2f((float)reg_s - 127.0f);
#pragma unroll
                    for (int i = 0; i < 16; i++) v[i] = ink_hip_e4m3((w[i / 4] >> (8 * (i % 4))) & 0xFF) * s;
                } else {
                    const float s = ink_hip_e4m3(reg_s) * g;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const unsigned int byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
                        v[2 * i] = lut[byte & 0xF] * s;
                        v[2 * i + 1] = lut[byte >> 4] * s;
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; i++) v[i] = 0.0f;
            }
            unsigned int pk[8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                pk[i] = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v[2 * i])) |
                        ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(v[2 * i + 1])) << 16);
            uint4* dst = (uint4*)&Bs[br][bh * 16];
            dst[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            dst[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        }
        __syncthreads();
        if (k0 + INK_GK < K) prefetch(k0 + INK_GK);
        if (m0 + wm < m_count) {
#pragma unroll
            for (int kk = 0; kk < INK_GK; kk += 16) {
                const ink_v16bf a = ink_ld16(&As[wm + (lane & 15)][kk]);
#pragma unroll
                for (int j = 0; j < NSUB; j++)
                    acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, ink_ld16(&Bs[wn + j * 16 + (lane & 15)][kk]), acc[j]);
            }
        }
        __syncthreads();
    }
    if (!SW) {
#pragma unroll
        for (int j = 0; j < NSUB; j++) {
            const unsigned int c = n0 + wn + j * 16 + (lane & 15);
            if (c >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int r = m0 + wm + 2 * e + (int)(lane >> 4);
                if (r < m_count) C[(unsigned long long)(m_start + r) * N + c] = __float2bfloat16(acc[j][e]);
            }
        }
        return;
    }
    // SW: softcapped SwiGLU on the gate/up column pairs, exactly as
    // inkling2_softcap_swiglu reads them from gu: g = even column's BF16-rounded
    // accumulator, u = the odd neighbour's, the same fp32 expression, the
    // activation written BF16. Adjacent columns sit in adjacent lanes, so the
    // pair is one xor shuffle; two i values pack into one u32 via a down
    // shuffle (CIRU's epilogue shape, adapted).
    const unsigned int I = N / 2;
#pragma unroll
    for (int j = 0; j < NSUB; j++) {
        const unsigned int c = n0 + wn + j * 16 + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int r = m0 + wm + 2 * e + (int)(lane >> 4);
            // the shuffles run converged on every lane; only the write is guarded
            const float g16 = __bfloat162float(__float2bfloat16(acc[j][e]));
            const float u16 = __shfl_xor_sync(0xffffffffu, g16, 1);
            const float v = tanhf(g16 / cg) * cg * (1.0f / (1.0f + __expf(-g16))) * tanhf(u16 / cu) * cu;
            const unsigned int h = ((lane & 15) & 1) == 0
                ? (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v)) : 0u;
            const unsigned int h2 = __shfl_down_sync(0xffffffffu, h, 2);
            if (r < m_count && c < N && ((lane & 15) & 3) == 0)
                *(unsigned int*)(C + (unsigned long long)(m_start + r) * I + (c >> 1)) = h | (h2 << 16);
        }
    }
}

// Same convention as gb10's inkling2_moe_tiles (kernels/gb10/inkling-2-mini/
// nvfp4/inkling2_moe_gemm.cu): tiles[0] = the row tiles' number, tiles[1 + t] =
// expert << 16 | m-tile within the expert (in mtile-row units). grid (1),
// block E rounded up to a warp. Launching only the tiles that exist drops the
// ~94% of CTAs that would otherwise exit on the m0 >= m_count guard when the
// grid spans the full token count per expert.
extern "C" __global__ void __launch_bounds__(1024) inkling2_moe_tiles_bf16(
    const int* __restrict__ offsets, int* __restrict__ tiles, const unsigned int E, const unsigned int mtile
) {
    __shared__ int warp_sums[32];
    const unsigned int e = threadIdx.x, lane = e % 32, warp = e / 32;
    const int n = e < E ? (offsets[e + 1] - offsets[e] + (int)mtile - 1) / (int)mtile : 0;
    int incl = n;   // inclusive scan within the warp, then across warps
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)lane >= o) incl += v;
    }
    if (lane == 31) warp_sums[warp] = incl;
    __syncthreads();
    int before = 0;
    for (unsigned int w = 0; w < warp; w++) before += warp_sums[w];
    const int start = before + incl - n;
    for (int i = 0; i < n; i++) tiles[1 + start + i] = (int)(e << 16) | i;
    if (e == blockDim.x - 1) tiles[0] = start + n;
}

// Routed experts: blockIdx.z is the expert; its rows are offsets[e] .. offsets[e + 1].
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_nvfp4(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float* __restrict__ s2, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const unsigned int N, const unsigned int K
) {
    const unsigned int e = blockIdx.z;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = blockIdx.y * 128;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<false, 128, false>(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K, 0.0f, 0.0f);
}

// Same GEMM with the softcapped SwiGLU folded into the epilogue: C is the
// activation buffer of N / 2 columns (the gate/up pairs reduced in-register).
// Bit-identical to inkling2_moe_gemm_nvfp4 -> inkling2_softcap_swiglu.
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_nvfp4_swiglu(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float* __restrict__ s2, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const unsigned int N, const unsigned int K,
    const float cg, const float cu
) {
    const unsigned int e = blockIdx.z;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = blockIdx.y * 128;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<false, 128, true>(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K, cg, cu);
}

extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_mxfp8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const unsigned int N, const unsigned int K
) {
    const unsigned int e = blockIdx.z;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = blockIdx.y * 64;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<true, 64, false>(A, codes + ne * K, scales + ne * (K / 32), 1.0f, C, m_start, m_count, m0, rows, N, K, 0.0f, 0.0f);
}

extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_mxfp8_swiglu(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const unsigned int N, const unsigned int K,
    const float cg, const float cu
) {
    const unsigned int e = blockIdx.z;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = blockIdx.y * 64;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<true, 64, true>(A, codes + ne * K, scales + ne * (K / 32), 1.0f, C, m_start, m_count, m0, rows, N, K, cg, cu);
}

// Tile-list variants: the same GEMMs over the (expert, m-tile) list
// inkling2_moe_tiles_bf16 writes, blockIdx.y selecting the tile — one grid
// dimension fewer than the blockIdx.z = expert launches above. Same math per
// tile, so the outputs are bit-identical.
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_nvfp4_tl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float* __restrict__ s2, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const int* __restrict__ tiles,
    const unsigned int N, const unsigned int K
) {
    if (blockIdx.y >= (unsigned int)tiles[0]) return;
    const unsigned int t = (unsigned int)tiles[1 + blockIdx.y], e = t >> 16;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = (int)(t & 0xFFFFu) * 128;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<false, 128, false>(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K, 0.0f, 0.0f);
}

extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_nvfp4_swiglu_tl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float* __restrict__ s2, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const int* __restrict__ tiles,
    const unsigned int N, const unsigned int K, const float cg, const float cu
) {
    if (blockIdx.y >= (unsigned int)tiles[0]) return;
    const unsigned int t = (unsigned int)tiles[1 + blockIdx.y], e = t >> 16;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = (int)(t & 0xFFFFu) * 128;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<false, 128, true>(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K, cg, cu);
}

extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_mxfp8_tl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const int* __restrict__ tiles,
    const unsigned int N, const unsigned int K
) {
    if (blockIdx.y >= (unsigned int)tiles[0]) return;
    const unsigned int t = (unsigned int)tiles[1 + blockIdx.y], e = t >> 16;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = (int)(t & 0xFFFFu) * 64;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<true, 64, false>(A, codes + ne * K, scales + ne * (K / 32), 1.0f, C, m_start, m_count, m0, rows, N, K, 0.0f, 0.0f);
}

extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_mxfp8_swiglu_tl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ offsets, const int* __restrict__ rows, const int* __restrict__ tiles,
    const unsigned int N, const unsigned int K, const float cg, const float cu
) {
    if (blockIdx.y >= (unsigned int)tiles[0]) return;
    const unsigned int t = (unsigned int)tiles[1 + blockIdx.y], e = t >> 16;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = (int)(t & 0xFFFFu) * 64;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_hip_gemm_tile<true, 64, true>(A, codes + ne * K, scales + ne * (K / 32), 1.0f, C, m_start, m_count, m0, rows, N, K, cg, cu);
}

// The softcapped SwiGLU between the grouped GEMMs (kept here: gb10 dropped its BF16-row
// prefill path in favour of the block-scaled one): act[r, i] = BF16(tanh(g / cg) * cg * sigmoid(g)
// * tanh(u / cu) * cu), g = gu[r, 2i], u = gu[r, 2i + 1] (the checkpoint's interleaved w13 rows),
// as gb10's ink_situ rounds it. grid (rows, ceil(I / 256)), block 256.
extern "C" __global__ void inkling2_softcap_swiglu(const __nv_bfloat16* __restrict__ gu,
                                                   __nv_bfloat16* __restrict__ act, const unsigned int I,
                                                   const float cg, const float cu) {
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = blockIdx.x;
    if (i >= I) return;
    const __nv_bfloat162 p = ((const __nv_bfloat162*)(gu + r * 2 * I))[i];
    const float g = __low2float(p), u = __high2float(p);
    const float v = tanhf(g / cg) * cg * (1.0f / (1.0f + __expf(-g))) * tanhf(u / cu) * cu;
    act[r * I + i] = __float2bfloat16(v);
}

// Block-scaled W4A4 / W8A8 (gb10 tensor-core formats) and their tile list: not on AMD.
extern "C" __global__ void inkling2_moe_tiles() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w4a4() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w8a8() { __builtin_trap(); }
