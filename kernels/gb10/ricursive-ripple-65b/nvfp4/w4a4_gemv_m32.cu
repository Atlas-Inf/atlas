// SPDX-License-Identifier: AGPL-3.0-only
//
// Small-M W4A4 NVFP4 GEMV for M <= 32 activation rows (DFlash verify width)
// on sm_121a: C[M,N] = (A_fp4 * sfa * scaleA2) . (B_fp4 * sfb * scaleB2)^T.
//
// Same operands, fragment layout and block-scaled MMA as `w4a4_gemm_mfast`
// (`mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`), but the
// prefill GEMM's 128x128 tiles waste 3/4 of each MMA at 32 rows and give the
// down projection (N = 5120) only 40 CTAs. A CTA here owns BN weight rows for
// all M rows. Each stage stages 256 K values of the BN-row weight panel and of
// the activations into shared memory with 16-byte cp.async (3 stages in
// flight); warp w computes the stage's w-th 64-value step, and the four warps'
// partial sums are reduced through shared memory at the end.
//
// A: packed [M, K/2], scales [M, K/16] (E4M3, groups of 16) — Atlas's native
// NVFP4 layout, as `quantize_bf16_to_nvfp4[_scaled]` writes it. B: the same
// layout [N, K/2] / [N, K/16]. K % (256 * W4G_SPW) == 0. Grid: (ceil(N/BN)). Block: 128.

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>
#include <stdint.h>

// PDL entry: the first W4G_TOUCH_CTAS CTAs pull their weight panels' first
// W4G_STAGES stages into L2 while the predecessor finishes (weights only).
#define W4G_TOUCH_CTAS 96u

#define W4G_WARPS 4
#ifndef W4G_STAGES
#define W4G_STAGES 3
#endif
#ifndef W4G_SPW
#define W4G_SPW 2     // 64-value K steps per warp per stage (GB10 sweep: 3 x 2 best)
#endif
#define W4G_ROWB (W4G_WARPS * W4G_SPW * 32)  // packed bytes per row per stage
#define W4G_ROWS (W4G_WARPS * W4G_SPW * 4)   // scale bytes per row per stage
// Shared rows padded by 16 B: fragment reads of 8 rows x 4 lanes hit 32 banks.
#define W4G_PITCH (W4G_ROWB + 16)

__device__ __forceinline__ void w4g_mma(
    float* acc, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

__device__ __forceinline__ void w4g_cp16(void* dst, const void* src) {
    unsigned d = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" : : "r"(d), "l"(src));
}

template <int BN>
__device__ __forceinline__ void w4a4_gemv_m32_impl(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const uint8_t* __restrict__ B_packed, const uint8_t* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K) {
    constexpr int NF = BN / 8;
    __shared__ __align__(16) uint8_t sB[W4G_STAGES][BN][W4G_PITCH];
    __shared__ __align__(16) uint8_t sBs[W4G_STAGES][BN][W4G_ROWS];
    __shared__ __align__(16) uint8_t sA[W4G_STAGES][32][W4G_PITCH];
    __shared__ __align__(16) uint8_t sAs[W4G_STAGES][32][W4G_ROWS];

    const unsigned t = threadIdx.x;
    const unsigned warp = t / 32, lane = t % 32;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    const unsigned n0 = blockIdx.x * BN;
    const size_t KB = (size_t)K / 2, KS = (size_t)K / 16;
    const unsigned Mm1 = (unsigned)(M - 1), Nm1 = (unsigned)(N - 1);
    const unsigned stages = (unsigned)K / (W4G_WARPS * W4G_SPW * 64u);

    auto load = [&](unsigned buf, unsigned st) {
        const size_t kb = (size_t)st * W4G_ROWB, ks = (size_t)st * W4G_ROWS;
        constexpr unsigned CPR = W4G_ROWB / 16, SPR = W4G_ROWS / 16;
#pragma unroll
        for (unsigned c = t; c < BN * CPR; c += 128u) {
            const unsigned r = c / CPR, off = (c % CPR) << 4;
            w4g_cp16(&sB[buf][r][off], B_packed + (size_t)min(n0 + r, Nm1) * KB + kb + off);
        }
        for (unsigned c = t; c < BN * SPR; c += 128u) {
            const unsigned r = c / SPR, off = (c % SPR) << 4;
            w4g_cp16(&sBs[buf][r][off], B_sf + (size_t)min(n0 + r, Nm1) * KS + ks + off);
        }
#pragma unroll
        for (unsigned c = t; c < 32u * CPR; c += 128u) {
            const unsigned r = c / CPR, off = (c % CPR) << 4;
            w4g_cp16(&sA[buf][r][off], A_packed + (size_t)min(r, Mm1) * KB + kb + off);
        }
        for (unsigned c = t; c < 32u * SPR; c += 128u) {
            const unsigned r = c / SPR, off = (c % SPR) << 4;
            w4g_cp16(&sAs[buf][r][off], A_sf + (size_t)min(r, Mm1) * KS + ks + off);
        }
    };

    float acc[2][NF][4];
#pragma unroll
    for (int mf = 0; mf < 2; mf++)
#pragma unroll
        for (int nf = 0; nf < NF; nf++)
            acc[mf][nf][0] = acc[mf][nf][1] = acc[mf][nf][2] = acc[mf][nf][3] = 0.0f;

#pragma unroll
    for (unsigned s = 0; s < W4G_STAGES - 1; s++) {
        if (s < stages) load(s, s);
        asm volatile("cp.async.commit_group;");
    }
    for (unsigned st = 0; st < stages; st++) {
        asm volatile("cp.async.wait_group %0;" : : "n"(W4G_STAGES - 2));
        __syncthreads();
        const unsigned buf = st % W4G_STAGES;
#pragma unroll
        for (unsigned sp = 0; sp < W4G_SPW; sp++) {
        const unsigned kb_w = (warp * W4G_SPW + sp) * 32u, ks_w = (warp * W4G_SPW + sp) * 4u;
        uint32_t a[2][4], sfa[2];
#pragma unroll
        for (int mf = 0; mf < 2; mf++) {
            const unsigned r0 = mf * 16u + gid;
            a[mf][0] = *(const uint32_t*)&sA[buf][r0][kb_w + 4u * tid];
            a[mf][1] = *(const uint32_t*)&sA[buf][r0 + 8u][kb_w + 4u * tid];
            a[mf][2] = *(const uint32_t*)&sA[buf][r0][kb_w + 16u + 4u * tid];
            a[mf][3] = *(const uint32_t*)&sA[buf][r0 + 8u][kb_w + 16u + 4u * tid];
            sfa[mf] = *(const uint32_t*)&sAs[buf][mf * 16u + ((tid & 1u) << 3) + gid][ks_w];
        }
#pragma unroll
        for (int nf = 0; nf < NF; nf++) {
            const unsigned nr = nf * 8u + gid;
            const uint32_t b0 = *(const uint32_t*)&sB[buf][nr][kb_w + 4u * tid];
            const uint32_t b1 = *(const uint32_t*)&sB[buf][nr][kb_w + 16u + 4u * tid];
            const uint32_t sfb = *(const uint32_t*)&sBs[buf][nr][ks_w];
#pragma unroll
            for (int mf = 0; mf < 2; mf++)
                w4g_mma(acc[mf][nf], a[mf][0], a[mf][1], a[mf][2], a[mf][3], b0, b1, sfa[mf], sfb);
        }
        }
        const unsigned nx = st + W4G_STAGES - 1;
        if (nx < stages) load(nx % W4G_STAGES, nx);
        asm volatile("cp.async.commit_group;");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // Two-pass cross-warp reduction through the (now idle) weight panel.
    constexpr int PER_LANE = 2 * NF * 4;
    static_assert(2 * PER_LANE * 32 * sizeof(float) <= sizeof(sB), "reduction scratch");
    float (*red)[PER_LANE][32] = reinterpret_cast<float (*)[PER_LANE][32]>(&sB[0][0][0]);
    auto put = [&](unsigned slot) {
#pragma unroll
        for (int mf = 0; mf < 2; mf++)
#pragma unroll
            for (int nf = 0; nf < NF; nf++)
#pragma unroll
                for (int j = 0; j < 4; j++) red[slot][(mf * NF + nf) * 4 + j][lane] = acc[mf][nf][j];
    };
    auto add = [&](unsigned slot) {
#pragma unroll
        for (int mf = 0; mf < 2; mf++)
#pragma unroll
            for (int nf = 0; nf < NF; nf++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[mf][nf][j] += red[slot][(mf * NF + nf) * 4 + j][lane];
    };
    if (warp >= 2) put(warp - 2);
    __syncthreads();
    if (warp < 2) add(warp);
    __syncthreads();
    if (warp == 1) put(0);
    __syncthreads();
    if (warp != 0) return;
    add(0);

    const float scale = scaleA2 * scaleB2;
#pragma unroll
    for (int mf = 0; mf < 2; mf++)
#pragma unroll
        for (int nf = 0; nf < NF; nf++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned m = mf * 16u + gid + ((j & 2) ? 8u : 0u);
                const unsigned n = n0 + nf * 8u + tid * 2u + (j & 1);
                if (m < (unsigned)M && n < (unsigned)N)
                    C[(size_t)m * N + n] = __float2bfloat16(acc[mf][nf][j] * scale);
            }
}

// GB10 sweep (M = 32, cold weights): wide N (gate/up, 40960 x 5120) 198 GB/s
// at 32-row panels; narrow N (down, 5120 x 40960) 181 GB/s at 16-row panels.
extern "C" __global__ void __launch_bounds__(128) w4a4_gemv_m32_bn32(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const unsigned char* __restrict__ B_packed, const unsigned char* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K) {
    atlas_pdl_enter_touch({B_packed, W4G_STAGES * W4G_ROWB, (unsigned long long)K / 2u},
                          {B_sf, W4G_STAGES * W4G_ROWS, (unsigned long long)K / 16u},
                          min((unsigned)N, W4G_TOUCH_CTAS * 32u), blockIdx.x, W4G_TOUCH_CTAS);
    w4a4_gemv_m32_impl<32>(A_packed, A_sf, B_packed, B_sf, C, scaleA2, scaleB2, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128) w4a4_gemv_m32_bn16(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const unsigned char* __restrict__ B_packed, const unsigned char* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K) {
    atlas_pdl_enter_touch({B_packed, W4G_STAGES * W4G_ROWB, (unsigned long long)K / 2u},
                          {B_sf, W4G_STAGES * W4G_ROWS, (unsigned long long)K / 16u},
                          min((unsigned)N, W4G_TOUCH_CTAS * 16u), blockIdx.x, W4G_TOUCH_CTAS);
    w4a4_gemv_m32_impl<16>(A_packed, A_sf, B_packed, B_sf, C, scaleA2, scaleB2, M, N, K);
}
