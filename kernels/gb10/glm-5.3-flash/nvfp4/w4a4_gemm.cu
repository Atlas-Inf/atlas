// SPDX-License-Identifier: AGPL-3.0-only
//
// M-fast W4A4 NVFP4 prefill GEMM for dense FFNs on sm_121a.
//
// A and B use Atlas's native NVFP4 layout: nibble-packed E2M1 values and
// one E4M3 scale per group of 16. The 128-thread schedule computes two
// 64-row chunks from each B fragment, pipelines three K stages, and puts M
// on grid.x so CTAs sharing a B panel are co-resident and reuse it from L2.
// This is the production schedule proven by the Nemotron target; keeping a
// target-local copy lets the DeepSeek/GLM closure expose only the measured
// entry point without changing every GB10 model's kernel surface.

#include <cuda_bf16.h>
#include <cstdint>

#define W4A4_THREADS 128

__device__ __forceinline__ void w4a4_cpa16(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" : : "r"(x), "l"(s));
}

__device__ __forceinline__ void w4a4_cpa4(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n" : : "r"(x), "l"(s));
}

__device__ __forceinline__ void w4a4_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void w4a4_wait1() {
    asm volatile("cp.async.wait_group 1;");
}

__device__ __forceinline__ void w4a4_wait() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void w4a4_mma(
    float* acc,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    uint32_t sfa,
    uint32_t sfb) {
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

extern "C" __global__ __launch_bounds__(W4A4_THREADS, 3) void w4a4_gemm_mfast(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ A_sf,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C,
    float scaleA2,
    float scaleB2,
    int M,
    int N,
    int K) {
    const unsigned cta_m = blockIdx.x * 128u;
    const unsigned cta_n = blockIdx.y * 128u;
    if (cta_m >= (unsigned)M) return;

    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    const unsigned wm = warp * 16;
    const unsigned gid = lane >> 2;
    const unsigned tid = lane & 3;

    __shared__ uint8_t sAf[3][128][32];
    __shared__ uint8_t sBf[3][128][32];
    __shared__ uint8_t sSA[3][128][4];
    __shared__ uint8_t sSB[3][128][4];

    float acc0[16][4], acc1[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = acc0[i][1] = acc0[i][2] = acc0[i][3] = 0;
        acc1[i][0] = acc1[i][1] = acc1[i][2] = acc1[i][3] = 0;
    }

    const int KB = K / 2;
    const int KS = K / 16;
    const unsigned Mm1 = (unsigned)(M - 1);
    const unsigned Nm1 = (unsigned)(N - 1);

#define W4A4_LOAD(buf, kb) do {                                                   \
    _Pragma("unroll")                                                            \
    for (unsigned ch = 0; ch < 2u; ch++) {                                       \
        unsigned idx = threadIdx.x + ch * 128u;                                  \
        unsigned r = idx >> 1, c = (idx & 1u) << 4;                              \
        unsigned ga = min(cta_m + r, Mm1);                                       \
        w4a4_cpa16(&sAf[buf][r][c], A_packed + (size_t)ga * KB + (kb)/2 + c);    \
        unsigned gb = min(cta_n + r, Nm1);                                       \
        w4a4_cpa16(&sBf[buf][r][c], B_packed + (size_t)gb * KB + (kb)/2 + c);    \
    }                                                                             \
    {                                                                             \
        unsigned r = threadIdx.x;                                                 \
        unsigned ga = min(cta_m + r, Mm1);                                       \
        w4a4_cpa4(&sSA[buf][r][0], A_sf + (size_t)ga * KS + (kb)/16);            \
        unsigned gb = min(cta_n + r, Nm1);                                       \
        w4a4_cpa4(&sSB[buf][r][0], B_sf + (size_t)gb * KS + (kb)/16);            \
    }                                                                             \
} while (0)

#define W4A4_COMPUTE(buf) do {                                                    \
    const unsigned fr0 = wm + gid, fr1 = fr0 + 8;                                \
    uint32_t a0 = *(uint32_t*)&sAf[buf][fr0][4*tid];                             \
    uint32_t a1 = *(uint32_t*)&sAf[buf][fr1][4*tid];                             \
    uint32_t a2 = *(uint32_t*)&sAf[buf][fr0][16 + 4*tid];                        \
    uint32_t a3 = *(uint32_t*)&sAf[buf][fr1][16 + 4*tid];                        \
    uint32_t e0 = *(uint32_t*)&sAf[buf][64 + fr0][4*tid];                        \
    uint32_t e1 = *(uint32_t*)&sAf[buf][64 + fr1][4*tid];                        \
    uint32_t e2 = *(uint32_t*)&sAf[buf][64 + fr0][16 + 4*tid];                   \
    uint32_t e3 = *(uint32_t*)&sAf[buf][64 + fr1][16 + 4*tid];                   \
    uint32_t sfa0 = *(uint32_t*)&sSA[buf][wm + ((tid & 1u) << 3) + gid][0];      \
    uint32_t sfa1 = *(uint32_t*)&sSA[buf][64 + wm + ((tid & 1u) << 3) + gid][0]; \
    _Pragma("unroll")                                                            \
    for (int nt = 0; nt < 16; nt++) {                                            \
        unsigned nc = nt * 8 + gid;                                              \
        uint32_t b0 = *(uint32_t*)&sBf[buf][nc][4*tid];                          \
        uint32_t b1 = *(uint32_t*)&sBf[buf][nc][16 + 4*tid];                     \
        uint32_t sfb = *(uint32_t*)&sSB[buf][nc][0];                             \
        w4a4_mma(acc0[nt], a0, a1, a2, a3, b0, b1, sfa0, sfb);                  \
        w4a4_mma(acc1[nt], e0, e1, e2, e3, b0, b1, sfa1, sfb);                  \
    }                                                                             \
} while (0)

    W4A4_LOAD(0, 0);
    w4a4_commit();
    if (K > 64) {
        W4A4_LOAD(1, 64);
        w4a4_commit();
    }
    w4a4_wait1();
    __syncthreads();

    int buf = 0;
    for (int kb = 128; kb < K; kb += 64) {
        int nb = (buf + 2) % 3;
        W4A4_LOAD(nb, kb);
        w4a4_commit();
        W4A4_COMPUTE(buf);
        w4a4_wait1();
        __syncthreads();
        buf = (buf + 1) % 3;
    }
    if (K > 64) {
        w4a4_wait();
        __syncthreads();
        W4A4_COMPUTE(buf);
        buf = (buf + 1) % 3;
    }
    w4a4_wait();
    __syncthreads();
    W4A4_COMPUTE(buf);

#undef W4A4_LOAD
#undef W4A4_COMPUTE

    const float scale = scaleA2 * scaleB2;
    const unsigned fr0 = wm + gid, fr1 = fr0 + 8;
#pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        const unsigned c0 = cta_n + nt * 8 + tid * 2;
        const unsigned c1 = c0 + 1;
        const unsigned r0 = cta_m + fr0;
        const unsigned r1 = cta_m + fr1;
        if (r0 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r0*N+c0] = __float2bfloat16(acc0[nt][0] * scale);
        if (r0 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r0*N+c1] = __float2bfloat16(acc0[nt][1] * scale);
        if (r1 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r1*N+c0] = __float2bfloat16(acc0[nt][2] * scale);
        if (r1 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r1*N+c1] = __float2bfloat16(acc0[nt][3] * scale);
        const unsigned r2 = r0 + 64;
        const unsigned r3 = r1 + 64;
        if (r2 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r2*N+c0] = __float2bfloat16(acc1[nt][0] * scale);
        if (r2 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r2*N+c1] = __float2bfloat16(acc1[nt][1] * scale);
        if (r3 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r3*N+c0] = __float2bfloat16(acc1[nt][2] * scale);
        if (r3 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r3*N+c1] = __float2bfloat16(acc1[nt][3] * scale);
    }
}
