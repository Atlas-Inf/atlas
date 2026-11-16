// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-V4.1-Flash: a routed expert's projection on tensor cores.
//
// A routed expert's weights are FP4 as the release stores them: W[N, K] as
// E2M1 values, two a byte, low nibble first, plus one E8M0 power-of-two
// scale per 32 consecutive K values (S[N, K/32]). Every E2M1 value times a
// power of two is exact in BF16, and the activations are FP8-rounded BF16,
// so the BF16 MMA below multiplies the checkpoint's weights unrounded and
// accumulates in FP32, which is the arithmetic of `dsv41_fp4_gemv` (the
// plain form) in another summation order.
//
//   dsv41_fp4_gemv_tc{8,16,32}: C[M, N] = A[M, K] . dequant(W)[N, K]^T for up
//     to 8/16/32 rows. The layout is `mxfp8_gemv_tc`'s (glm-5.3-flash
//     mxfp8_gemv.cu): CTA = 16 weight rows; 8 warps split K in 64-wide
//     chunks; lane (g = lane/4, c = lane%4) loads 16 consecutive K values of
//     rows g and g+8 (8 packed bytes each) and of each activation tile row
//     8t+g, prefetching its next chunk. MMA step j (0..3) feeds k-slots
//     {2c,2c+1} <- values 4j+{0,1} and {2c+8,2c+9} <- 4j+{2,3} of the lane's
//     run in both A and B. A row's output bits do not depend on how many
//     rows share the launch.
//
// A: BF16 [M, K] (row stride K). C rows at C + m*out_stride. K % 32 == 0.
// Grid: (ceil(N/16),1,1) Block: (256,1,1).

#include <cuda_bf16.h>

#define F4_WARPS 8
#define F4_WARP 32
#define F4_BLOCK 32

template <int NT>
struct F4Chunk {
    unsigned long long w0, w1;  // 16 E2M1 values of rows g and g+8
    uint4 x[NT][2];             // 16 BF16 activations of row 8t+g per tile t
    unsigned int s0, s1;        // E8M0 block scales of rows g, g+8
};

// Chunk `ch` of the lane's rows: packed values at w{0,1}p + ch * 32 (8 bytes
// each) and scales at s{0,1}p[ch * 2]; activations at k = kb = ch * 64 + 16c.
template <int NT>
__device__ __forceinline__ void f4_load(
    F4Chunk<NT>& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p,
    const __nv_bfloat16* A, unsigned int g, unsigned int M, unsigned int K,
    unsigned int ch, unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = 0ull; t.w1 = 0ull; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    const unsigned long long wo = (unsigned long long)ch * 32ull;
    const unsigned long long so = (unsigned long long)ch * 2ull;
    if (v0) { t.w0 = *(const unsigned long long*)(w0p + wo); t.s0 = s0p[so]; }
    if (v1) { t.w1 = *(const unsigned long long*)(w1p + wo); t.s1 = s1p[so]; }
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * K + kb);
            t.x[i][0] = xp[0];
            t.x[i][1] = xp[1];
        }
    }
}

// BF16 2^(s-127) (E8M0 scale; 0 decodes to 0, as does the NaN code 255).
__device__ __forceinline__ __nv_bfloat162 f4_scale(unsigned int s) {
    const unsigned short bits = (s == 0u || s == 255u) ? 0 : (unsigned short)(s << 7);
    const __nv_bfloat16 v = __ushort_as_bfloat16(bits);
    return __halves2bfloat162(v, v);
}

// One m16n8k16 BF16 MMA step into `acc` (FP32).
__device__ __forceinline__ void f4_mma(float (&acc)[4], const unsigned int (&a)[4],
                                       unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int NT>
__device__ __forceinline__ void dsv41_fp4_gemv_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    // E2M1 nibble -> BF16 bits (exact): 0, 0.5, 1, 1.5, 2, 3, 4, 6, then the
    // same negated.
    __shared__ unsigned short s_e2m1[16];
    __shared__ float s_red[F4_WARPS][F4_WARP][NT * 4];
    if (threadIdx.x < 16) {
        const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        const float v = mags[threadIdx.x & 7u];
        s_e2m1[threadIdx.x] =
            __bfloat16_as_ushort(__float2bfloat16((threadIdx.x & 8u) ? -v : v));
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / F4_WARP;
    const unsigned int lane = threadIdx.x % F4_WARP;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = blockIdx.x * 16u + g;
    const unsigned int r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N;
    const unsigned long long blocks = K / F4_BLOCK;
    const unsigned int chunks = (K + 63u) / 64u;
    // A row is K/2 packed bytes; the lane's 16 values are 8 of them.
    const unsigned char* w0p = W + (unsigned long long)(v0 ? r0 : 0u) * (K / 2u) + 8u * c;
    const unsigned char* w1p = W + (unsigned long long)(v1 ? r1 : 0u) * (K / 2u) + 8u * c;
    const unsigned char* s0p = S + (unsigned long long)(v0 ? r0 : 0u) * blocks + (c >> 1);
    const unsigned char* s1p = S + (unsigned long long)(v1 ? r1 : 0u) * blocks + (c >> 1);

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    F4Chunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks)
        f4_load<NT>(cur, w0p, w1p, s0p, s1p, A, g, M, K, ch, ch * 64u + 16u * c, v0, v1);
    for (; ch < chunks; ch += F4_WARPS) {
        const unsigned int next = ch + F4_WARPS;
        if (next < chunks)
            f4_load<NT>(nxt, w0p, w1p, s0p, s1p, A, g, M, K, next, next * 64u + 16u * c, v0, v1);
        const __nv_bfloat162 sc0 = f4_scale(cur.s0), sc1 = f4_scale(cur.s1);
        // Two nibbles -> a BF16 pair, times the block's scale.
        auto pair = [&](unsigned int lo, unsigned int hi, __nv_bfloat162 sc) {
            __nv_bfloat162 v = __halves2bfloat162(__ushort_as_bfloat16(s_e2m1[lo]),
                                                  __ushort_as_bfloat16(s_e2m1[hi]));
            v = __hmul2(v, sc);
            return *reinterpret_cast<unsigned int*>(&v);
        };
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Values 4j..4j+3 of each row's 16-value run = nibbles 4j..4j+3 =
            // bytes 2j and 2j+1 = 16 bits at 16j.
            const unsigned int x0 = (unsigned int)(cur.w0 >> (16 * j)) & 0xFFFFu;
            const unsigned int x1 = (unsigned int)(cur.w1 >> (16 * j)) & 0xFFFFu;
            const unsigned int a[4] = {
                pair(x0 & 0xFu, (x0 >> 4) & 0xFu, sc0),
                pair(x1 & 0xFu, (x1 >> 4) & 0xFu, sc1),
                pair((x0 >> 8) & 0xFu, x0 >> 12, sc0),
                pair((x1 >> 8) & 0xFu, x1 >> 12, sc1),
            };
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                const uint4 lo = cur.x[i][0], hi = cur.x[i][1];
                const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                f4_mma(acc[i], a, xw[2 * j], xw[2 * j + 1]);
            }
        }
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) s_red[warp][lane][i * 4 + q] = acc[i][q];
    }
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            v[q] = 0.0f;
            #pragma unroll
            for (int w = 0; w < F4_WARPS; w++) v[q] += s_red[w][lane][i * 4 + q];
        }
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            const unsigned int n = (q < 2) ? r0 : r1;
            const unsigned int m = 8u * i + 2u * c + (q & 1u);
            if (n < N && m < M)
                C[(unsigned long long)m * out_stride + n] = __float2bfloat16(v[q]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp4_gemv_tc8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    dsv41_fp4_gemv_tc_impl<1>(A, W, S, C, M, N, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp4_gemv_tc16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    dsv41_fp4_gemv_tc_impl<2>(A, W, S, C, M, N, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp4_gemv_tc32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    dsv41_fp4_gemv_tc_impl<4>(A, W, S, C, M, N, K, out_stride);
}
