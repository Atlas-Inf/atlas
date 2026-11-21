// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 grouped expert GEMMs and the dense NVFP4 GEMM, strix-hip (gfx1151 WMMA) port of
// kernels/gb10/inkling-2-mini/nvfp4/inkling2_moe_gemm.cu (same entry points, arguments, grids
// and numerics for the BF16-activation path): C[m, n] = sum_k A[row(m), k] * W_e[n, k] for the
// rows m of each expert e (moe_sort_by_expert order), W_e read from the checkpoint's stacked
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

__device__ __forceinline__ float ink_hip_e4m3(unsigned int b) {
    const unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u) v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

template <bool MX, int TM>
__device__ __forceinline__ void ink_hip_gemm_tile(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wc,
    const unsigned char* __restrict__ ws, const float g, __nv_bfloat16* __restrict__ C,
    const int m_start, const int m_count, const int m0, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K
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

    for (unsigned int k0 = 0; k0 < K; k0 += INK_GK) {
        // activations: TM rows x 32 BF16 = TM * 4 chunks of 16 bytes
        for (int c = tid; c < TM * 4; c += 256) {
            const int r = c / 4, col = (c % 4) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (src[r] >= 0) v = *(const uint4*)(A + (unsigned long long)src[r] * K + k0 + col);
            *(uint4*)&As[r][col] = v;
        }
        // weights: thread t dequantises row t / 2, half t % 2 (16 values, one NVFP4 scale)
        {
            const unsigned int r = tid / 2, h = tid % 2, n = n0 + r;
            float v[16];
            if (n < N) {
                if (MX) {
                    const float s = exp2f((float)ws[(unsigned long long)n * (K / 32) + k0 / 32] - 127.0f);
                    const uint4 q = *(const uint4*)(wc + (unsigned long long)n * K + k0 + h * 16);
                    const unsigned int w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                    for (int i = 0; i < 16; i++) v[i] = ink_hip_e4m3((w[i / 4] >> (8 * (i % 4))) & 0xFF) * s;
                } else {
                    const float s = ink_hip_e4m3(ws[(unsigned long long)n * (K / 16) + k0 / 16 + h]) * g;
                    const uint2 q = *(const uint2*)(wc + (unsigned long long)n * (K / 2) + k0 / 2 + h * 8);
                    const unsigned int w[2] = {q.x, q.y};
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
#pragma unroll
            for (int i = 0; i < 16; i++) Bs[r][h * 16 + i] = __float2bfloat16(v[i]);
        }
        __syncthreads();
        if (m0 + wm < m_count) {
#pragma unroll
            for (int kk = 0; kk < INK_GK; kk += 16) {
                ink_v16bf a;
#pragma unroll
                for (int i = 0; i < 16; i++) a[i] = (__bf16)__bfloat162float(As[wm + (lane & 15)][kk + i]);
#pragma unroll
                for (int j = 0; j < NSUB; j++) {
                    ink_v16bf b;
#pragma unroll
                    for (int i = 0; i < 16; i++) b[i] = (__bf16)__bfloat162float(Bs[wn + j * 16 + (lane & 15)][kk + i]);
                    acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[j]);
                }
            }
        }
        __syncthreads();
    }
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
    ink_hip_gemm_tile<false, 128>(A, codes + ne * (K / 2), scales + ne * (K / 16), s2[e], C, m_start, m_count, m0, rows, N, K);
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
    ink_hip_gemm_tile<true, 64>(A, codes + ne * K, scales + ne * (K / 32), 1.0f, C, m_start, m_count, m0, rows, N, K);
}

// A dense NVFP4 projection (Spark mode's attention): C [M, N] = A [M, K] W^T.
// grid (N / 128, ceil(M / 128)), block 256.
extern "C" __global__ void __launch_bounds__(256) inkling2_gemm_nvfp4(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales, const float s2, __nv_bfloat16* __restrict__ C,
    const unsigned int M, const unsigned int N, const unsigned int K
) {
    const int m0 = blockIdx.y * 128;
    if (m0 >= (int)M) return;
    ink_hip_gemm_tile<false, 128>(A, codes, scales, s2, C, 0, (int)M, m0, nullptr, N, K);
}

// Block-scaled W4A4 / W8A8 (gb10 tensor-core formats) and their tile list: not on AMD.
extern "C" __global__ void inkling2_moe_tiles() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w4a4() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w8a8() { __builtin_trap(); }
