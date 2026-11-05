// SPDX-License-Identifier: AGPL-3.0-only
//
// Small-M BF16 GEMV for M <= 32 rows (DFlash verify width) on sm_121:
// C[m, n] (row stride ldc) = sum_k A[m, k] * W[n, k], FP32 accumulation.
//
// Ripple keeps attention in BF16 (Strict), and its q|gate (20480 x 5120) and
// o (5120 x 10240) projections dominate the verify step. cuBLASLt's 64x64
// tiles at M = 32 stream them at ~175 GB/s. Here a CTA owns 32 weight rows
// for all M rows: the weight panel streams through shared memory with 16-byte
// cp.async (3 stages of 128 K values in flight), activations are read from
// L2 (every CTA reads the same small A), and the 4 warps split each stage's
// K and reduce through shared memory. `mma.sync.m16n8k16` BF16. The tile
// macros are tuned on GB10 (examples: kbench sweep); the host op's grid uses
// the same B16G_BN.
//
// A: [M, K] row-major BF16. W: [N, K] row-major BF16. K % 512 == 0.
// Grid: (ceil(N/16)). Block: 128.

#include <cuda_bf16.h>
#include <stdint.h>

#ifndef B16G_STAGES
#define B16G_STAGES 3
#endif
#ifndef B16G_BN
#define B16G_BN 16                       // GB10 sweep: 3 stages x 16 rows x KK 8 best
#endif
#ifndef B16G_KK
#define B16G_KK 8                        // k16 MMA steps per warp per stage
#endif
#define B16G_KS (4 * 16 * B16G_KK)       // K values per stage (4 warps)
#define B16G_PITCH (B16G_KS * 2 + 16)    // bytes; pad: 8 rows x 4 lanes -> 32 banks

__device__ __forceinline__ void b16g_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void b16g_cp16(void* dst, const void* src) {
    unsigned d = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" : : "r"(d), "l"(src));
}

extern "C" __global__ void __launch_bounds__(128) bf16_gemv_m32(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C, int M, int N, int K, int ldc) {
    constexpr int NF = B16G_BN / 8;
    __shared__ __align__(16) uint8_t sW[B16G_STAGES][B16G_BN][B16G_PITCH];

    const unsigned t = threadIdx.x;
    const unsigned warp = t / 32, lane = t % 32;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    const unsigned n0 = blockIdx.x * B16G_BN;
    const unsigned Mm1 = (unsigned)(M - 1), Nm1 = (unsigned)(N - 1);
    const unsigned stages = (unsigned)K / B16G_KS;
    const uint8_t* Wb = reinterpret_cast<const uint8_t*>(W);
    const size_t row_bytes = (size_t)K * 2;

    auto load = [&](unsigned buf, unsigned st) {
        const size_t kb = (size_t)st * B16G_KS * 2;
        constexpr unsigned CPR = B16G_KS * 2 / 16;  // 16-byte chunks per row
#pragma unroll
        for (unsigned c = t; c < B16G_BN * CPR; c += 128u) {
            const unsigned r = c / CPR, off = (c % CPR) << 4;
            b16g_cp16(&sW[buf][r][off], Wb + (size_t)min(n0 + r, Nm1) * row_bytes + kb + off);
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
    for (unsigned s = 0; s < B16G_STAGES - 1; s++) {
        if (s < stages) load(s, s);
        asm volatile("cp.async.commit_group;");
    }
    for (unsigned st = 0; st < stages; st++) {
        asm volatile("cp.async.wait_group %0;" : : "n"(B16G_STAGES - 2));
        __syncthreads();
        const unsigned buf = st % B16G_STAGES;
#pragma unroll
        for (int kk = 0; kk < B16G_KK; kk++) {
            const unsigned kl = (warp * B16G_KK + kk) * 16u;      // K within the stage
            const unsigned kg = st * B16G_KS + kl;                // global K
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
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (kl + 2u * tid) * 2u);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (kl + 8u + 2u * tid) * 2u);
#pragma unroll
                for (int mf = 0; mf < 2; mf++)
                    b16g_mma(acc[mf][nf], a[mf][0], a[mf][1], a[mf][2], a[mf][3], b0, b1);
            }
        }
        const unsigned nx = st + B16G_STAGES - 1;
        if (nx < stages) load(nx % B16G_STAGES, nx);
        asm volatile("cp.async.commit_group;");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    constexpr int PER_LANE = 2 * NF * 4;
    static_assert(2 * PER_LANE * 32 * sizeof(float) <= sizeof(sW), "reduction scratch");
    static_assert(B16G_BN % 8 == 0, "panel rows");
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
                    C[(size_t)m * ldc + n] = __float2bfloat16(acc[mf][nf][j]);
            }
}
