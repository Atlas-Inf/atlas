// SPDX-License-Identifier: AGPL-3.0-only
//
// Small-M FP8-weight GEMV for M <= 32 rows (the DFlash2 drafter's block) on
// sm_121: C[m, n] = row_scale[n] * sum_k A[m, k] * W8[n, k], W8 E4M3 with a
// per-row f32 scale (`DenseWeight::quantize_to_fp8`), A BF16, FP32 accumulate.
//
// The drafter only proposes tokens and verification is exact, so its weight
// precision can move acceptance, never output. Same schedule as
// bf16_gemv_m32 (32-row-or-fewer A from L2, the weight panel through 3
// cp.async stages, 4 warps split each stage's K and reduce through shared
// memory), at half the weight bytes: each FP8 pair is widened to BF16 in
// registers before the BF16 MMA.
//
// A: [M, K] row-major BF16. W8: [N, K] row-major E4M3. K % 512 == 0.
// Grid: (ceil(N/16)). Block: 128.

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define F8G_STAGES 3
#define F8G_BN 16
#define F8G_KK 8                    // k16 MMA steps per warp per stage
#define F8G_KS (4 * 16 * F8G_KK)    // K values (= bytes) per row per stage
#define F8G_PITCH (F8G_KS + 16)     // pad: 8 rows x 4 lanes -> distinct banks
#define F8G_TOUCH_CTAS 96u

__device__ __forceinline__ void f8g_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// L1-allocating copies (.ca): on GB10 the .cg (L2-only) form of this
// pipeline streams at ~192 GB/s, .ca at ~238 (pure-read roofline ~240).
__device__ __forceinline__ void f8g_cp16(void* dst, const void* src) {
    unsigned d = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" : : "r"(d), "l"(src));
}

// Two E4M3 values (low byte first) -> packed BF16x2 (exact: E4M3 fits BF16).
__device__ __forceinline__ uint32_t f8g_widen(uint16_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ void fp8_gemv_m32_impl(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ W8,
    const float* __restrict__ row_scale, __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    constexpr int NF = F8G_BN / 8;
    __shared__ __align__(16) uint8_t sW[F8G_STAGES][F8G_BN][F8G_PITCH];

    const unsigned t = threadIdx.x;
    const unsigned warp = t / 32, lane = t % 32;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    const unsigned n0 = blockIdx.x * F8G_BN;
    const unsigned Mm1 = (unsigned)(M - 1), Nm1 = (unsigned)(N - 1);
    const unsigned stages = (unsigned)K / F8G_KS;

    auto load = [&](unsigned buf, unsigned st) {
        const size_t kb = (size_t)st * F8G_KS;
        constexpr unsigned CPR = F8G_KS / 16;
#pragma unroll
        for (unsigned c = t; c < F8G_BN * CPR; c += 128u) {
            const unsigned r = c / CPR, off = (c % CPR) << 4;
            f8g_cp16(&sW[buf][r][off], W8 + (size_t)min(n0 + r, Nm1) * K + kb + off);
        }
    };

    const uint32_t* a_lo[2];
    const uint32_t* a_hi[2];
#pragma unroll
    for (int mf = 0; mf < 2; mf++) {
        a_lo[mf] = reinterpret_cast<const uint32_t*>(A + (size_t)min(mf * 16u + gid, Mm1) * K);
        a_hi[mf] = reinterpret_cast<const uint32_t*>(A + (size_t)min(mf * 16u + gid + 8u, Mm1) * K);
    }

    float acc[2][NF][4];
#pragma unroll
    for (int mf = 0; mf < 2; mf++)
#pragma unroll
        for (int nf = 0; nf < NF; nf++)
            acc[mf][nf][0] = acc[mf][nf][1] = acc[mf][nf][2] = acc[mf][nf][3] = 0.0f;

#pragma unroll
    for (unsigned s = 0; s < F8G_STAGES - 1; s++) {
        if (s < stages) load(s, s);
        asm volatile("cp.async.commit_group;");
    }
    for (unsigned st = 0; st < stages; st++) {
        asm volatile("cp.async.wait_group %0;" : : "n"(F8G_STAGES - 2));
        __syncthreads();
        const unsigned buf = st % F8G_STAGES;
#pragma unroll
        for (int kk = 0; kk < F8G_KK; kk++) {
            const unsigned kl = (warp * F8G_KK + kk) * 16u;
            const unsigned kg = st * F8G_KS + kl;
            uint32_t a[2][4];
#pragma unroll
            for (int mf = 0; mf < 2; mf++) {
                a[mf][0] = __ldg(a_lo[mf] + (kg + 2u * tid) / 2u);
                a[mf][1] = __ldg(a_hi[mf] + (kg + 2u * tid) / 2u);
                a[mf][2] = __ldg(a_lo[mf] + (kg + 8u + 2u * tid) / 2u);
                a[mf][3] = __ldg(a_hi[mf] + (kg + 8u + 2u * tid) / 2u);
            }
#pragma unroll
            for (int nf = 0; nf < NF; nf++) {
                const uint8_t* row = &sW[buf][nf * 8u + gid][0];
                const uint32_t b0 = f8g_widen(*reinterpret_cast<const uint16_t*>(row + kl + 2u * tid));
                const uint32_t b1 = f8g_widen(*reinterpret_cast<const uint16_t*>(row + kl + 8u + 2u * tid));
#pragma unroll
                for (int mf = 0; mf < 2; mf++)
                    f8g_mma(acc[mf][nf], a[mf][0], a[mf][1], a[mf][2], a[mf][3], b0, b1);
            }
        }
        const unsigned nx = st + F8G_STAGES - 1;
        if (nx < stages) load(nx % F8G_STAGES, nx);
        asm volatile("cp.async.commit_group;");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    constexpr int PER_LANE = 2 * NF * 4;
    static_assert(2 * PER_LANE * 32 * sizeof(float) <= sizeof(sW), "reduction scratch");
    float (*red)[PER_LANE][32] = reinterpret_cast<float (*)[PER_LANE][32]>(&sW[0][0][0]);
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

#pragma unroll
    for (int mf = 0; mf < 2; mf++)
#pragma unroll
        for (int nf = 0; nf < NF; nf++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned m = mf * 16u + gid + ((j & 2) ? 8u : 0u);
                const unsigned n = n0 + nf * 8u + tid * 2u + (j & 1);
                if (m < (unsigned)M && n < (unsigned)N)
                    C[(size_t)m * N + n] = __float2bfloat16(acc[mf][nf][j] * __ldg(row_scale + n));
            }
}

// PDL entry: the first F8G_TOUCH_CTAS CTAs pull their panels' first stages
// into L2 while the predecessor finishes (weights and scales only).
extern "C" __global__ void __launch_bounds__(128) fp8_gemv_m32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W8,
    const unsigned char* __restrict__ row_scale, __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    atlas_pdl_enter_touch({W8, F8G_STAGES * F8G_KS, (unsigned long long)K}, {row_scale, 0u, 0ull},
                          min((unsigned)N, F8G_TOUCH_CTAS * F8G_BN), blockIdx.x, F8G_TOUCH_CTAS);
    fp8_gemv_m32_impl(A, W8, reinterpret_cast<const float*>(row_scale), C, M, N, K);
}
