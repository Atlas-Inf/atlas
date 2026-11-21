// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 NVFP4 GEMMs for prefill rows: the grouped expert GEMM C[m, n] = sum_k A[row(m), k] * W_e[n, k]
// for the rows m of each expert e (the moe_sort_by_expert order), W_e read straight from
// the checkpoint's stacked NVFP4 tensor (codes [E, N, K/2], even element in the low nibble;
// E4M3 scales [E, N, K/16]; per-expert global s2[e]). Each 32-wide K step of W is
// dequantised to BF16 in shared memory (e2m1 * scale * s2 rounded to BF16, as the
// reference's dequantised weights) and multiplied on the tensor cores
// (mma.sync m16n8k16 BF16, fp32 accumulation) against the BF16 activations.
//
// Tile 128 rows (NVFP4; 64 for MXFP8) x 128 outputs, 256 threads (8 warps as 4 x 2 of
// 32 x 64, or 2 x 4 of 32 x 32), K step 32,
// cp.async double buffering of the activation rows and the raw codes. Rows are gathered
// through `rows` (sorted -> source row) or taken in order when it is NULL.
// The dense entry runs the same tile for one NVFP4 projection (Spark mode's attention).
// The MXFP8 entry reads the coda layers' experts (E4M3 codes [E, N, K], E8M0 scales per
// 32 inputs [E, N, K/32]): an E4M3 value times a power of two is exact in BF16.
// grouped grid (N / 128, ceil(max rows of an expert / tile rows), E), block 256.

#include "../../common/atlas_pdl.cuh"
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip): these kernels are mma.sync / cp.async / block-scaled MMA PTX. The
// symbols exist so the kernel table resolves; spark-model never launches them under
// cfg!(atlas_scale) (inkling2/moe_ops.rs routes prefill rows to the grouped GEMVs).
extern "C" __global__ void inkling2_moe_gemm_nvfp4() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_tiles() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w4a4() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w8a8() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_mxfp8() { __builtin_trap(); }
extern "C" __global__ void inkling2_gemm_nvfp4() { __builtin_trap(); }
#else
#include "../../common/nvfp4_w4a4_tile.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define GM 64     // MXFP8 tile rows (the 128-row tile's shared memory does not fit 32-byte codes)
#define GM_FP4 128 // NVFP4 tile rows: an expert's ~128 rows a prefill chunk dequantize its weights once
#define GN 128
#define GK 32
#define GPAD 8
#define GSTRIDE (GK + GPAD)   // 40 BF16 = 80 bytes: conflict-free 32-bit fragment loads

__device__ __constant__ float INK_GEMM_E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                                   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ void ink_cp16(void* smem, const void* gmem, bool pred) {
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;   // zero-fill when out of range
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void ink_mma(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One TM x 128 tile of rows m_start + m0 .. of `C` against one weight: NVFP4 [N, K/2]
// (codes wc, E4M3 scales ws per 16, global g) or MXFP8 [N, K] (E8M0 scales per 32).
template <bool MX, int TM>
__device__ __forceinline__ void ink_gemm_tile(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wc,
    const unsigned char* __restrict__ ws, const float g, __nv_bfloat16* __restrict__ C,
    const int m_start, const int m_count, const int m0, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K
) {
    __shared__ __align__(16) __nv_bfloat16 As[2][TM][GSTRIDE];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][GN][GSTRIDE];
    __shared__ __align__(16) unsigned char Braw[2][GN][MX ? 32 : 16];
    __shared__ float lut[16];
    const unsigned int n0 = blockIdx.x * GN;
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    if (tid < 16) lut[tid] = INK_GEMM_E2M1[tid];

    // this thread's activation chunks: rows tid / 4 (+ 64 ...), 16 bytes (8 values) at (tid % 4) * 8
    constexpr int A_PASSES = TM / 64;
    const int a_row = tid / 4, a_col = (tid % 4) * 8;
    bool a_ok[A_PASSES];
    const __nv_bfloat16* a_ptr[A_PASSES];
#pragma unroll
    for (int p = 0; p < A_PASSES; p++) {
        const int r = a_row + 64 * p;
        a_ok[p] = m0 + r < m_count;
        const long long src = a_ok[p] ? (rows ? rows[m_start + m0 + r] : m_start + m0 + r) : 0;
        a_ptr[p] = A + (unsigned long long)src * K + a_col;
    }
    // this thread's raw code chunk: row n0 + tid % 128, 16 bytes (NVFP4: threads 0..127,
    // 32 values; MXFP8: both halves of the row's 32 bytes)
    const unsigned int c_row = tid & 127, c_half = tid >> 7;
    const unsigned char* c_ptr = wc + (unsigned long long)(n0 + c_row) * (MX ? K : K / 2);

    auto issue = [&](int buf, unsigned int k0) {
#pragma unroll
        for (int p = 0; p < A_PASSES; p++) ink_cp16(&As[buf][a_row + 64 * p][a_col], a_ptr[p] + k0, a_ok[p]);
        if (MX) ink_cp16(&Braw[buf][c_row][c_half * 16], c_ptr + k0 + c_half * 16, n0 + c_row < N);
        else if (tid < GN) ink_cp16(&Braw[buf][tid][0], c_ptr + k0 / 2, n0 + tid < N);
        asm volatile("cp.async.commit_group;\n" ::);
    };
    // dequantise this step's codes: thread t takes row t / 2, half t % 2 (16 values, one scale)
    auto dequant = [&](int buf, unsigned int k0) {
        const unsigned int r = tid / 2, h = tid % 2;
        const unsigned int n = n0 + r;
        unsigned int out[8];
        if (MX) {
            const float s = n < N ? exp2f((float)ws[(unsigned long long)n * (K / 32) + k0 / 32] - 127.0f) : 0.0f;
            const uint4 q = *(const uint4*)&Braw[buf][r][h * 16];
            const unsigned int w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned int pair = (w[i / 2] >> (16 * (i % 2))) & 0xFFFF;
                __nv_fp8_e4m3 a, b;
                *(unsigned char*)&a = pair & 0xFF;
                *(unsigned char*)&b = pair >> 8;
                const __nv_bfloat16 lo = __float2bfloat16((float)a * s), hi = __float2bfloat16((float)b * s);
                out[i] = (unsigned int)__bfloat16_as_ushort(lo) | ((unsigned int)__bfloat16_as_ushort(hi) << 16);
            }
        } else {
            const float s = n < N ? (float)(*(const __nv_fp8_e4m3*)&ws[(unsigned long long)n * (K / 16) + k0 / 16 + h]) * g : 0.0f;
            const uint2 q = *(const uint2*)&Braw[buf][r][h * 8];
            const unsigned int w[2] = {q.x, q.y};
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned int byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
                const __nv_bfloat16 lo = __float2bfloat16(lut[byte & 0xF] * s), hi = __float2bfloat16(lut[byte >> 4] * s);
                out[i] = (unsigned int)__bfloat16_as_ushort(lo) | ((unsigned int)__bfloat16_as_ushort(hi) << 16);
            }
        }
        uint4* dst = (uint4*)&Bs[buf][r][h * 16];
        dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
        dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
    };

    // 8 warps as WARPS_M x WARPS_N, each 32 rows x NJ * 8 columns
    constexpr int WARPS_M = TM / 32, WARPS_N = 8 / WARPS_M, NJ = GN / WARPS_N / 8;
    float acc[2][NJ][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;

    const unsigned int wm = (warp / WARPS_N) * 32, wn = (warp % WARPS_N) * (NJ * 8);
    const unsigned int g8 = lane / 4, t2 = (lane % 4) * 2;
    const unsigned int steps = K / GK;

    issue(0, 0);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    dequant(0, 0);
    __syncthreads();

    for (unsigned int s = 0; s < steps; s++) {
        const int buf = s & 1;
        if (s + 1 < steps) issue(buf ^ 1, (s + 1) * GK);
#pragma unroll
        for (int kk = 0; kk < GK; kk += 16) {
            unsigned int af[2][4], bf[NJ][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const unsigned int r = wm + i * 16 + g8;
                af[i][0] = *(const unsigned int*)&As[buf][r][kk + t2];
                af[i][1] = *(const unsigned int*)&As[buf][r + 8][kk + t2];
                af[i][2] = *(const unsigned int*)&As[buf][r][kk + t2 + 8];
                af[i][3] = *(const unsigned int*)&As[buf][r + 8][kk + t2 + 8];
            }
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                const unsigned int c = wn + j * 8 + g8;
                bf[j][0] = *(const unsigned int*)&Bs[buf][c][kk + t2];
                bf[j][1] = *(const unsigned int*)&Bs[buf][c][kk + t2 + 8];
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++) ink_mma(acc[i][j], af[i], bf[j]);
        }
        if (s + 1 < steps) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            dequant(buf ^ 1, (s + 1) * GK);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r0 = m0 + wm + i * 16 + g8;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            const unsigned int c = n0 + wn + j * 8 + t2;
            if (c >= N) continue;
            if (r0 < m_count)
                *(__nv_bfloat162*)&C[(unsigned long long)(m_start + r0) * N + c] = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (r0 + 8 < m_count)
                *(__nv_bfloat162*)&C[(unsigned long long)(m_start + r0 + 8) * N + c] = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// Routed experts: blockIdx.z is the expert; its rows are offsets[e] .. offsets[e + 1].
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_nvfp4(
    const __nv_bfloat16* __restrict__ A,       // [source rows, K]
    const unsigned char* __restrict__ codes,   // [E, N, K/2]
    const unsigned char* __restrict__ scales,  // [E, N, K/16]
    const float* __restrict__ s2,              // [E]
    __nv_bfloat16* __restrict__ C,             // [sorted rows, N]
    const int* __restrict__ offsets,           // [E + 1]
    const int* __restrict__ rows,              // [sorted rows] -> source row, or NULL
    const unsigned int N,
    const unsigned int K
) {
    atlas_pdl_enter();
    const unsigned int e = blockIdx.z;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = blockIdx.y * GM_FP4;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_gemm_tile<false, GM_FP4>(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K);
}

// The checkpoint's own NVFP4 experts at its declared numerics (W4A4): rows
// quantized by inkling2_quant_fp4 / inkling2_situ_quant_fp4 (codes [*, K/2],
// E4M3 scales [*, K/16], outer scales [*]) against the experts as stored, on
// the E2M1 tensor cores (nvfp4_w4a4_tile.cuh, 128 x 128 tiles);
// C[m, n] = BF16(acc * (outer[row(m)] * s2[e])), as vLLM's Inkling 2 expert
// GEMM scales it. grid (N / 128, ceil(rows / 128) + E), block 256: blockIdx.y indexes
// the row tiles inkling2_moe_tiles lists (the rest return).
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
template <bool MX8>
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
    const unsigned char* B = codes + ne * (MX8 ? K : K / 2);
    const unsigned char* S = scales + ne * (MX8 ? K / 32 : K / 16);
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

// The same over MXFP8 experts (codes [E, N, K], E8M0 scales [E, N, K/32], no global).
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_gemm_mxfp8(
    const __nv_bfloat16* __restrict__ A,       // [source rows, K]
    const unsigned char* __restrict__ codes,   // [E, N, K]
    const unsigned char* __restrict__ scales,  // [E, N, K/32]
    __nv_bfloat16* __restrict__ C,             // [sorted rows, N]
    const int* __restrict__ offsets,           // [E + 1]
    const int* __restrict__ rows,              // [sorted rows] -> source row, or NULL
    const unsigned int N,
    const unsigned int K
) {
    atlas_pdl_enter();
    const unsigned int e = blockIdx.z;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const int m0 = blockIdx.y * GM;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_gemm_tile<true, GM>(A, codes + ne * K, scales + ne * (K / 32), 1.0f, C, m_start, m_count, m0, rows, N, K);
}

// A dense NVFP4 projection (ModelOpt codes [N, K/2], E4M3 scales [N, K/16], scalar
// global) of M BF16 rows: C [M, N] = A [M, K] W^T. grid (N / 128, ceil(M / 128)), block 256.
extern "C" __global__ void __launch_bounds__(256) inkling2_gemm_nvfp4(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float s2, __nv_bfloat16* __restrict__ C,
    const unsigned int M, const unsigned int N, const unsigned int K
) {
    atlas_pdl_enter();
    const int m0 = blockIdx.y * GM_FP4;
    if (m0 >= (int)M) return;
    ink_gemm_tile<false, GM_FP4>(A, codes, scales, s2, C, 0, (int)M, m0, nullptr, N, K);
}
#endif  // AMD
