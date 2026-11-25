// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 routed experts for prefill rows: grouped GEMMs C[m, n] = sum_k A[row(m), k] * W_e[n, k]
// over the rows m of each expert e (the moe_sort_by_expert order), at the checkpoint's
// declared numerics. The rows are quantized first (inkling2_moe_prefill.cu): NVFP4 rows
// for the NVFP4 experts (W4A4, mma.sync kind::mxf4nvf4) and MXFP8 rows for the MXFP8
// experts (W8A8, kind::mxf8f6f4), both against the experts as the checkpoint stores them
// (codes [E, N, K/2] or [E, N, K], block scales [E, N, K/16] or [E, N, K/32]), on the
// 128 x 128 tile of common/nvfp4_w4a4_tile.cuh. Rows are gathered through `rows` (sorted
// -> source row) or taken in order when it is NULL; inkling2_moe_tiles lists the CTAs'
// (expert, row tile) pairs.

#include "../../common/atlas_pdl.cuh"
#include "../../common/nvfp4_w4a4_tile.cuh"
#include <cuda_bf16.h>

// The row tiles of the experts' sorted rows, 128 rows each, for the grouped W4A4 / W8A8
// GEMMs: tiles[0] = their number, tiles[1 + t] = expert << 16 | tile index within the
// expert. One block of up to 1024 experts. grid (1), block E rounded up to a warp.
extern "C" __global__ void __launch_bounds__(1024) inkling2_moe_tiles(const int* __restrict__ offsets,
                                                                       int* __restrict__ tiles, const unsigned int E) {
    atlas_pdl_enter();
    __shared__ int warp_sums[32];
    const unsigned int e = threadIdx.x, lane = e % 32, warp = e / 32;
    const int n = e < E ? (offsets[e + 1] - offsets[e] + W4_BM - 1) / W4_BM : 0;
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

// The grouped W4A4 / W8A8 tile of one (column tile, row tile, expert): rows quantized
// by this file's quantizers, experts as stored; C = BF16(acc * outer[row] * s2[e]),
// or BF16(acc) for MXFP8 (no outer or global scales).
template <bool MX8, bool PointerTable = false>
__device__ __forceinline__ void ink_moe_lowp(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ As, const float* __restrict__ outer,
    const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
    const float* __restrict__ s2, __nv_bfloat16* __restrict__ C, const int* __restrict__ offsets,
    const int* __restrict__ tiles, const int* __restrict__ rows, const unsigned int N, const unsigned int K
) {
    __shared__ __align__(16) W4Smem sm;
    atlas_pdl_enter();
    // this CTA's (expert, row tile) from inkling2_moe_tiles
    if (blockIdx.y >= (unsigned int)tiles[0]) return;
    const unsigned int t = (unsigned int)tiles[1 + blockIdx.y], e = t >> 16;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const unsigned int m0 = (t & 0xFFFFu) * W4_BM;
    const unsigned int rows_left = (unsigned int)m_count - m0, n0 = blockIdx.x * 128;
    w4_rows(sm, rows, (unsigned int)m_start + m0, rows_left);
    const unsigned long long ne = (unsigned long long)e * N;
    const unsigned char* B = PointerTable ? reinterpret_cast<const unsigned char* const*>(codes)[e] : codes + ne * (MX8 ? K : K / 2);
    const unsigned char* S = PointerTable ? reinterpret_cast<const unsigned char* const*>(scales)[e] : scales + ne * (MX8 ? K / 32 : K / 16);
    float acc[16][4];
    w4_tile<MX8>(sm, acc, A, As, K, rows_left, B, S, n0, B, S, n0 + 64);
    const float g = MX8 ? 1.0f : s2[e];
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int q = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
#pragma unroll
    for (unsigned int h = 0; h < 2; ++h) {
#pragma unroll
        for (unsigned int mt = 0; mt < 4; ++mt) {
            const unsigned int lr = wm * 64 + mt * 16 + q + 8 * h;
            if (lr >= rows_left) continue;
            const float rs = MX8 ? 1.0f : __fmul_rn(outer[sm.tok[lr]], g);
            __nv_bfloat16* out = C + ((unsigned long long)m_start + m0 + lr) * N;
#pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                const unsigned int col = n0 + w4_nt(wn, j) * 8 + 2 * c;
                const float* a = acc[mt * 4 + j];
                *(__nv_bfloat162*)&out[col] = __floats2bfloat162_rn(__fmul_rn(a[2 * h], rs), __fmul_rn(a[2 * h + 1], rs));
            }
        }
    }
}

// The checkpoint's own NVFP4 experts at its declared numerics (W4A4): rows
// quantized by inkling2_quant_fp4 / inkling2_situ_quant_fp4 (codes [*, K/2],
// E4M3 scales [*, K/16], outer scales [*]) against the experts as stored, on
// the E2M1 tensor cores (nvfp4_w4a4_tile.cuh, 128 x 128 tiles);
// C[m, n] = BF16(acc * (outer[row(m)] * s2[e])), as vLLM's Inkling 2 expert
// GEMM scales it. grid (N / 128, ceil(rows / 128) + E), block 256: blockIdx.y indexes
// the row tiles inkling2_moe_tiles lists (the rest return).
extern "C" __global__ void __launch_bounds__(256, 2) inkling2_moe_gemm_w4a4(
    const unsigned char* __restrict__ A,       // [source rows, K/2]
    const unsigned char* __restrict__ As,      // [source rows, K/16]
    const float* __restrict__ outer,           // [source rows]
    const unsigned char* __restrict__ codes,   // [E, N, K/2]
    const unsigned char* __restrict__ scales,  // [E, N, K/16]
    const float* __restrict__ s2,              // [E]
    __nv_bfloat16* __restrict__ C,             // [sorted rows, N]
    const int* __restrict__ offsets,           // [E + 1]
    const int* __restrict__ tiles,             // inkling2_moe_tiles
    const int* __restrict__ rows,              // [sorted rows] -> source row, or NULL
    const unsigned int N,
    const unsigned int K
) {
    ink_moe_lowp<false>(A, As, outer, codes, scales, s2, C, offsets, tiles, rows, N, K);
}

// The checkpoint's MXFP8 experts at its declared numerics (W8A8): rows quantized by
// inkling2_quant_mxfp8 / inkling2_situ_quant_mxfp8 (E4M3 [*, K], UE8M0 scales
// [*, K/32]) against the experts as stored (codes [E, N, K], scales [E, N, K/32]) on
// kind::mxf8f6f4; C[m, n] = BF16(acc). Same grid as inkling2_moe_gemm_w4a4.
extern "C" __global__ void __launch_bounds__(256, 2) inkling2_moe_gemm_w8a8(
    const unsigned char* __restrict__ A,       // [source rows, K]
    const unsigned char* __restrict__ As,      // [source rows, K/32]
    const unsigned char* __restrict__ codes,   // [E, N, K]
    const unsigned char* __restrict__ scales,  // [E, N, K/32]
    __nv_bfloat16* __restrict__ C,             // [sorted rows, N]
    const int* __restrict__ offsets,           // [E + 1]
    const int* __restrict__ tiles,             // inkling2_moe_tiles
    const int* __restrict__ rows,              // [sorted rows] -> source row, or NULL
    const unsigned int N,
    const unsigned int K
) {
    ink_moe_lowp<true>(A, As, nullptr, codes, scales, nullptr, C, offsets, tiles, rows, N, K);
}
