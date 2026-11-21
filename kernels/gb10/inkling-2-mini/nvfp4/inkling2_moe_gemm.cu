// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 NVFP4 GEMMs for prefill rows: the grouped expert GEMM C[m, n] = sum_k A[row(m), k] * W_e[n, k]
// for the rows m of each expert e (the moe_sort_by_expert order), W_e read straight from
// the checkpoint's stacked NVFP4 tensor (codes [E, N, K/2], even element in the low nibble;
// E4M3 scales [E, N, K/16]; per-expert global s2[e]). Each 32-wide K step of W is
// dequantised to BF16 in shared memory (e2m1 * scale * s2 rounded to BF16, as the
// reference's dequantised weights) and multiplied on the tensor cores
// (mma.sync m16n8k16 BF16, fp32 accumulation) against the BF16 activations.
//
// Tile 64 rows x 128 outputs, 256 threads (8 warps as 2 x 4, 32 x 32 each), K step 32,
// cp.async double buffering of the activation rows and the raw codes. Rows are gathered
// through `rows` (sorted -> source row) or taken in order when it is NULL.
// The dense entry runs the same tile for one NVFP4 projection (Spark mode's attention).
// grouped grid (N / 128, ceil(max rows of an expert / 64), E), block 256.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define GM 64
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

// One 64 x 128 tile of rows m_start + m0 .. of `C` against one NVFP4 weight [N, K/2]
// (codes wc, scales ws, global g).
__device__ __forceinline__ void ink_gemm_tile(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wc,
    const unsigned char* __restrict__ ws, const float g, __nv_bfloat16* __restrict__ C,
    const int m_start, const int m_count, const int m0, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K
) {
    __shared__ __align__(16) __nv_bfloat16 As[2][GM][GSTRIDE];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][GN][GSTRIDE];
    __shared__ __align__(16) unsigned char Braw[2][GN][16];
    __shared__ float lut[16];
    const unsigned int n0 = blockIdx.x * GN;
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    if (tid < 16) lut[tid] = INK_GEMM_E2M1[tid];

    // this thread's activation chunk: row tid / 4, 16 bytes (8 values) at (tid % 4) * 8
    const int a_row = tid / 4, a_col = (tid % 4) * 8;
    const bool a_ok = m0 + a_row < m_count;
    const long long src = a_ok ? (rows ? rows[m_start + m0 + a_row] : m_start + m0 + a_row) : 0;
    const __nv_bfloat16* a_ptr = A + (unsigned long long)src * K + a_col;
    // this thread's raw code chunk (threads 0..127): row n0 + tid, 16 bytes = 32 values
    const unsigned char* c_ptr = wc + (unsigned long long)(n0 + (tid & 127)) * (K / 2);

    auto issue = [&](int buf, unsigned int k0) {
        ink_cp16(&As[buf][a_row][a_col], a_ptr + k0, a_ok);
        if (tid < GN) ink_cp16(&Braw[buf][tid][0], c_ptr + k0 / 2, n0 + tid < N);
        asm volatile("cp.async.commit_group;\n" ::);
    };
    // dequantise this step's codes: thread t takes row t / 2, half t % 2 (16 values, one scale)
    auto dequant = [&](int buf, unsigned int k0) {
        const unsigned int r = tid / 2, h = tid % 2;
        const unsigned int n = n0 + r;
        const float s = n < N ? (float)(*(const __nv_fp8_e4m3*)&ws[(unsigned long long)n * (K / 16) + k0 / 16 + h]) * g : 0.0f;
        const uint2 q = *(const uint2*)&Braw[buf][r][h * 8];
        const unsigned int w[2] = {q.x, q.y};
        unsigned int out[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
            const __nv_bfloat16 lo = __float2bfloat16(lut[byte & 0xF] * s), hi = __float2bfloat16(lut[byte >> 4] * s);
            out[i] = (unsigned int)__bfloat16_as_ushort(lo) | ((unsigned int)__bfloat16_as_ushort(hi) << 16);
        }
        uint4* dst = (uint4*)&Bs[buf][r][h * 16];
        dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
        dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
    };

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;

    const unsigned int wm = (warp / 4) * 32, wn = (warp % 4) * 32;
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
            unsigned int af[2][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const unsigned int r = wm + i * 16 + g8;
                af[i][0] = *(const unsigned int*)&As[buf][r][kk + t2];
                af[i][1] = *(const unsigned int*)&As[buf][r + 8][kk + t2];
                af[i][2] = *(const unsigned int*)&As[buf][r][kk + t2 + 8];
                af[i][3] = *(const unsigned int*)&As[buf][r + 8][kk + t2 + 8];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int c = wn + j * 8 + g8;
                bf[j][0] = *(const unsigned int*)&Bs[buf][c][kk + t2];
                bf[j][1] = *(const unsigned int*)&Bs[buf][c][kk + t2 + 8];
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) ink_mma(acc[i][j], af[i], bf[j]);
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
        for (int j = 0; j < 4; j++) {
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
    const int m0 = blockIdx.y * GM;
    if (m0 >= m_count) return;
    const unsigned long long ne = (unsigned long long)e * N;
    ink_gemm_tile(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K);
}

// A dense NVFP4 projection (ModelOpt codes [N, K/2], E4M3 scales [N, K/16], scalar
// global) of M BF16 rows: C [M, N] = A [M, K] W^T. grid (N / 128, ceil(M / 64)), block 256.
extern "C" __global__ void __launch_bounds__(256) inkling2_gemm_nvfp4(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float s2, __nv_bfloat16* __restrict__ C,
    const unsigned int M, const unsigned int N, const unsigned int K
) {
    atlas_pdl_enter();
    const int m0 = blockIdx.y * GM;
    if (m0 >= (int)M) return;
    ink_gemm_tile(A, codes, scales, s2, C, 0, (int)M, m0, nullptr, N, K);
}
