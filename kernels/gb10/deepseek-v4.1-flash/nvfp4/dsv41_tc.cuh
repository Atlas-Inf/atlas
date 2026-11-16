// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the tensor-core projection every `dsv41_*gemv_tc*`
// entry and the fused expert kernels (`dsv41_expert.cu`) instantiate. See
// `dsv41_tc.cu` for the arithmetic and the layout. `cta` is which 16 weight
// rows the block computes: `blockIdx.x` when a launch is one projection.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define F4_WARPS 8
#define F4_WARP 32
#define F4_BLOCK 32

template <int NT>
struct F4Chunk {
    uint4 w0, w1;               // 16 values of rows g and g+8 (FP4: 8 bytes each)
    uint4 x[NT][2];             // 16 BF16 activations of row 8t+g per tile t
    unsigned int s0, s1;        // E8M0 block scales of rows g, g+8
};

// Chunk `ch` of the lane's rows: values at w{0,1}p + ch * (FP4 ? 32 : 64)
// (8 or 16 bytes each) and scales at s{0,1}p[ch * 2]; activations at k = kb
// = ch * 64 + 16c.
template <int NT, bool FP4>
__device__ __forceinline__ void f4_load(
    F4Chunk<NT>& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p,
    const __nv_bfloat16* A, unsigned int g, unsigned int M, unsigned int K,
    unsigned int ch, unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    const unsigned long long wo = (unsigned long long)ch * (FP4 ? 32ull : 64ull);
    const unsigned long long so = (unsigned long long)ch * 2ull;
    if constexpr (FP4) {
        if (v0) { const uint2 w = *(const uint2*)(w0p + wo); t.w0.x = w.x; t.w0.y = w.y; t.s0 = s0p[so]; }
        if (v1) { const uint2 w = *(const uint2*)(w1p + wo); t.w1.x = w.x; t.w1.y = w.y; t.s1 = s1p[so]; }
    } else {
        if (v0) { t.w0 = *(const uint4*)(w0p + wo); t.s0 = s0p[so]; }
        if (v1) { t.w1 = *(const uint4*)(w1p + wo); t.s1 = s1p[so]; }
    }
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

template <int NT, bool FP4, bool F32OUT>
__device__ __forceinline__ void dsv41_gemv_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    void* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    unsigned int cta
) {
    // Weight code -> BF16 bits (exact). FP4: an E2M1 nibble, 0, 0.5, 1, 1.5,
    // 2, 3, 4, 6, then the same negated. FP8: an E4M3 byte.
    __shared__ unsigned short s_code[256];
    __shared__ float s_red[F4_WARPS][F4_WARP][NT * 4];
    if constexpr (FP4) {
        if (threadIdx.x < 16) {
            const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            const float v = mags[threadIdx.x & 7u];
            s_code[threadIdx.x] =
                __bfloat16_as_ushort(__float2bfloat16((threadIdx.x & 8u) ? -v : v));
        }
    } else {
        for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
            __nv_fp8_e4m3 f;
            *reinterpret_cast<unsigned char*>(&f) = (unsigned char)i;
            s_code[i] = __bfloat16_as_ushort(__float2bfloat16((float)f));
        }
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / F4_WARP;
    const unsigned int lane = threadIdx.x % F4_WARP;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = cta * 16u + g;
    const unsigned int r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N;
    const unsigned long long blocks = K / F4_BLOCK;
    const unsigned int chunks = (K + 63u) / 64u;
    // A row is K bytes, or K/2 packed; the lane's 16 values are 16 or 8 of them.
    const unsigned int row_bytes = FP4 ? K / 2u : K, lane_bytes = FP4 ? 8u : 16u;
    const unsigned char* w0p = W + (unsigned long long)(v0 ? r0 : 0u) * row_bytes + lane_bytes * c;
    const unsigned char* w1p = W + (unsigned long long)(v1 ? r1 : 0u) * row_bytes + lane_bytes * c;
    const unsigned char* s0p = S + (unsigned long long)(v0 ? r0 : 0u) * blocks + (c >> 1);
    const unsigned char* s1p = S + (unsigned long long)(v1 ? r1 : 0u) * blocks + (c >> 1);

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    F4Chunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks)
        f4_load<NT, FP4>(cur, w0p, w1p, s0p, s1p, A, g, M, K, ch, ch * 64u + 16u * c, v0, v1);
    for (; ch < chunks; ch += F4_WARPS) {
        const unsigned int next = ch + F4_WARPS;
        if (next < chunks)
            f4_load<NT, FP4>(nxt, w0p, w1p, s0p, s1p, A, g, M, K, next, next * 64u + 16u * c, v0,
                             v1);
        const __nv_bfloat162 sc0 = f4_scale(cur.s0), sc1 = f4_scale(cur.s1);
        // Two weight codes -> a BF16 pair, times the block's scale.
        auto pair = [&](unsigned int lo, unsigned int hi, __nv_bfloat162 sc) {
            __nv_bfloat162 v = __halves2bfloat162(__ushort_as_bfloat16(s_code[lo]),
                                                  __ushort_as_bfloat16(s_code[hi]));
            v = __hmul2(v, sc);
            return *reinterpret_cast<unsigned int*>(&v);
        };
        const unsigned int wb0[4] = {cur.w0.x, cur.w0.y, cur.w0.z, cur.w0.w};
        const unsigned int wb1[4] = {cur.w1.x, cur.w1.y, cur.w1.z, cur.w1.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int a[4];
            if constexpr (FP4) {
                // Values 4j..4j+3 of each row's 16-value run = nibbles 4j..4j+3
                // = bytes 2j and 2j+1 = 16 bits at 16j of the run's 64.
                const unsigned int x0 = (wb0[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
                const unsigned int x1 = (wb1[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
                a[0] = pair(x0 & 0xFu, (x0 >> 4) & 0xFu, sc0);
                a[1] = pair(x1 & 0xFu, (x1 >> 4) & 0xFu, sc1);
                a[2] = pair((x0 >> 8) & 0xFu, x0 >> 12, sc0);
                a[3] = pair((x1 >> 8) & 0xFu, x1 >> 12, sc1);
            } else {
                // Values 4j..4j+3 = bytes 4j..4j+3 = word j.
                const unsigned int x0 = wb0[j], x1 = wb1[j];
                a[0] = pair(x0 & 0xFFu, (x0 >> 8) & 0xFFu, sc0);
                a[1] = pair(x1 & 0xFFu, (x1 >> 8) & 0xFFu, sc1);
                a[2] = pair((x0 >> 16) & 0xFFu, x0 >> 24, sc0);
                a[3] = pair((x1 >> 16) & 0xFFu, x1 >> 24, sc1);
            }
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
            if (n < N && m < M) {
                const unsigned long long at = (unsigned long long)m * out_stride + n;
                if constexpr (F32OUT) {
                    static_cast<float*>(C)[at] = v[q];
                } else {
                    static_cast<__nv_bfloat16*>(C)[at] = __float2bfloat16(v[q]);
                }
            }
        }
    }
}
