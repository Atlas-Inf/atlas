// SPDX-License-Identifier: AGPL-3.0-only

// MXFP8 weights for small-row projections (speculative drafter blocks).
//
// Format: W[N, K] as E4M3 bytes (row-major) plus one E8M0 power-of-two scale
// per 32 consecutive K values (S[N, K/32]), the OCP MX layout. E4M3 values and
// power-of-two scales are both exactly representable in BF16, so the tensor-
// core GEMV below dequantizes into BF16 MMA operands without rounding.
//
//   mxfp8_quantize_bf16: one thread per 32-value block; the scale is the
//     smallest power of two with amax / scale <= 448, values round-to-nearest
//     with saturation.
//   mxfp8_gemv_tc{8,16,32}: C[M, N] = A[M, K] · dequant(W)[N, K]^T for up to
//     8/16/32 rows. CTA = 16 weight rows; 8 warps split K in 64-wide chunks;
//     lane (g = lane/4, c = lane%4) loads 16 consecutive K values of rows g
//     and g+8 and of each activation tile row 8t+g, prefetching its next
//     chunk. MMA step j (0..3) feeds k-slots {2c,2c+1} <- values 4j+{0,1} and
//     {2c+8,2c+9} <- 4j+{2,3} of the lane's run in both A and B.
//
//   mxfp8_gemv_tc{8,16,32}_tiled: the same on the chunk-tiled twin of
//     mxfp8_tile16, where each warp's chunk is one contiguous KiB instead of
//     sixteen 64-byte row pieces K bytes apart (ATLAS_GLM_MXFP8_TILED).
//
//   fp8b_gemv_tc{8,16,32}[_tiled]: the same on FP8 block weights (E4M3 +
//     f32 per 128x128 block, ModelOpt FP8_PB_WO): unscaled E4M3 operands,
//     each chunk's FP32 partial times its block scale (exact weights).
//
//   *_multi[_tiled]: up to three projections of the same rows in one launch.
//
//   mxfp8_gemv_tc{8,16}_grouped: the same per head h = blockIdx.z for
//     per-head weights (GLM absorbed W_uk / W_uv): A[:, h*K..] (row stride
//     lda) times head h of W [G, N, K] / S [G, N, K/32] into C[:, h*N..].
//     No 32-row tier: at K = 256 half its K-split warps idle and it lost to
//     BF16 cuBLAS (scripts/dev/mxfp8_grouped_bench.cu).
//
// A: BF16 [M, K] (row stride K). C rows at C + m*out_stride. K % 32 == 0.
// Grid: (ceil(N/16),1,G) Block: (256,1,1); G = 1 for the plain tiers.

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define MX_WARPS 8
#define MX_WARP 32
#define MX_BLOCK 32

extern "C" __global__ void mxfp8_quantize_bf16(
    const __nv_bfloat16* __restrict__ W,
    unsigned char* __restrict__ Q,
    unsigned char* __restrict__ S,
    unsigned long long blocks
) {
    const unsigned long long b = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const __nv_bfloat16* w = W + b * MX_BLOCK;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < MX_BLOCK; i++) amax = fmaxf(amax, fabsf(__bfloat162float(w[i])));
    // Smallest e with amax * 2^-e <= 448 (E4M3 max), clamped to E8M0's range.
    int e = -127;
    if (amax > 0.0f) {
        e = (int)ceilf(log2f(amax / 448.0f));
        e = max(-127, min(127, e));
        if (ldexpf(amax, -e) > 448.0f && e < 127) e++;
    }
    S[b] = (unsigned char)(e + 127);
    #pragma unroll
    for (int i = 0; i < MX_BLOCK; i++) {
        const __nv_fp8_e4m3 q(ldexpf(__bfloat162float(w[i]), -e));
        Q[b * MX_BLOCK + i] = *reinterpret_cast<const unsigned char*>(&q);
    }
}

template <int NT>
struct MxChunk {
    uint4 w0, w1;          // 16 E4M3 values of rows g and g+8
    uint4 x[NT][2];        // 16 BF16 activations of row 8t+g per tile t
    unsigned int s0, s1;   // E8M0 block scales of rows g, g+8
};

// Chunk `ch` of the lane's rows: values at w{0,1}p + ch * wstep (16 bytes each)
// and scales at s{0,1}p[ch * sstep]; activations at k = kb = ch * 64 + 16c.
// B128: s0p is the f32 scale of the CTA's 128-row block at K block 0 (one
// row of the [N/128, K/128] grid); chunk ch's scale is its (ch / 2)-th word,
// read by every lane into t.s0 (as bits).
template <int NT, bool B128 = false>
__device__ __forceinline__ void mx_load(
    MxChunk<NT>& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p, unsigned int wstep, unsigned int sstep,
    const __nv_bfloat16* A, unsigned int g, unsigned int M, unsigned int K, unsigned int lda,
    unsigned int ch, unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    const unsigned long long wo = (unsigned long long)ch * wstep;
    if constexpr (B128) {
        (void)sstep;
        (void)s1p;
        if (v0) t.w0 = *(const uint4*)(w0p + wo);
        if (v1) t.w1 = *(const uint4*)(w1p + wo);
        t.s0 = reinterpret_cast<const unsigned int*>(s0p)[ch >> 1];
    } else {
        const unsigned long long so = (unsigned long long)ch * sstep;
        if (v0) { t.w0 = *(const uint4*)(w0p + wo); t.s0 = s0p[so]; }
        if (v1) { t.w1 = *(const uint4*)(w1p + wo); t.s1 = s1p[so]; }
    }
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * lda + kb);
            t.x[i][0] = xp[0];
            t.x[i][1] = xp[1];
        }
    }
}

// SiLU(gate) * up of two projection outputs (FP32 sums), rounded as vLLM's
// `silu_and_mul` on the BF16 GEMM outputs rounds: gate and up to BF16, SiLU
// in FP32 to BF16, the product to BF16. The `_swiglu` tiers' epilogue (the
// qwen3.8-flash-next target's MXFP8 shared expert); prefill's twin of it
// reads the same BF16 values, so both write the same bytes per element.
__device__ __forceinline__ __nv_bfloat16 mx_swiglu(float gate, float up) {
    const float g = __bfloat162float(__float2bfloat16(gate));
    const float u = __bfloat162float(__float2bfloat16(up));
    const float s = __bfloat162float(__float2bfloat16(g / (1.0f + expf(-g))));
    return __float2bfloat16(s * u);
}

// BF16 2^(s-127) (E8M0 scale; 0 decodes to 0, as does the NaN code 255).
__device__ __forceinline__ __nv_bfloat162 mx_scale(unsigned int s) {
    const unsigned short bits = (s == 0u || s == 255u) ? 0 : (unsigned short)(s << 7);
    const __nv_bfloat16 v = __ushort_as_bfloat16(bits);
    return __halves2bfloat162(v, v);
}

// One m16n8k16 BF16 MMA step into `acc` (FP32).
__device__ __forceinline__ void mx_mma(float (&acc)[4], const unsigned int (&a)[4],
                                       unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The body for weight-row tile `cta` (rows 16 * cta ..): the plain tiers'
// CTA `blockIdx.x`, or a CTA of a launch that runs other work beside it
// (glm_mla_fuse.cu, the multi-projection launches below). TILED reads W / S
// in the chunk-tiled layout of `mxfp8_tile16` (same values into the same
// MMAs: bit-identical).
//
// B128 (FP8 block weights, ModelOpt FP8_PB_WO / DeepSeek block FP8): W is
// E4M3 [N, K] as before (or its `mxfp8_tile16` value tiling), S an f32
// [N/128, K/128] grid (row-major, never tiled), N % 16 == 0 and K % 128 ==
// 0. The E4M3 values enter the MMA unscaled (exact in BF16); each warp's
// 64-value chunk accumulates into its own FP32 partial, which is multiplied
// by the chunk's block scale (both of a CTA's 16 rows share one 128-row
// block) and added to the running sum. No weight is ever rounded; the same
// warp / tile / reduction order as the MX tiers keeps each row's bits
// independent of how many rows share the launch.
template <int NT, bool TILED = false, bool B128 = false, bool SWIGLU = false>
__device__ __forceinline__ void mxfp8_gemv_tc_cta(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int lda,
    unsigned int out_stride,
    unsigned int cta,
    unsigned int ldw = 0u
) {
    // E4M3 byte -> BF16 bits (exact).
    __shared__ unsigned short s_e4m3[256];
    __shared__ float s_red[MX_WARPS][MX_WARP][NT * 4];
    for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
        __nv_fp8_e4m3 f;
        *reinterpret_cast<unsigned char*>(&f) = (unsigned char)i;
        s_e4m3[i] = __bfloat16_as_ushort(__float2bfloat16((float)f));
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / MX_WARP;
    const unsigned int lane = threadIdx.x % MX_WARP;
    const unsigned int g = lane >> 2, c = lane & 3u;
    static_assert(!(TILED && SWIGLU), "the SWIGLU tiers read the row-major twin");
    static_assert(!(B128 && SWIGLU), "the SWIGLU tiers require MXFP8 weights");
    const unsigned int r0 = cta * (SWIGLU ? 8u : 16u) + g;
    const unsigned int r1 = r0 + (SWIGLU ? N : 8u);
    const bool v0 = r0 < N, v1 = SWIGLU ? v0 : r1 < N;
    const unsigned long long blocks = K / MX_BLOCK;
    const unsigned int chunks = (K + 63u) / 64u;
    const unsigned char *w0p, *w1p, *s0p, *s1p;
    unsigned int wstep, sstep;
    if (TILED) {
        // Tile `cta` is 16 * K value bytes and 16 * K/32 scale bytes from
        // its row-major offset; chunk ch of lane l at ((ch * 32 + l) * 32)
        // (row g's 16 values, then row g+8's), its scales at ch * 32 + 2g +
        // c/2 (row g) and + 16 (row g+8).
        const unsigned long long t = cta;
        w0p = W + t * 16ull * K + lane * 32u;
        w1p = w0p + 16;
        s0p = S + t * 16ull * blocks + 2u * g + (c >> 1);
        s1p = s0p + 16;
        wstep = 1024u;
        sstep = 32u;
    } else {
        // `ldw` (B128 row-major only): values between weight rows when W is a
        // K-slice of a wider weight (0: K).
        const unsigned long long rs = (B128 && ldw != 0u) ? ldw : K;
        w0p = W + (unsigned long long)(v0 ? r0 : 0u) * rs + 16u * c;
        w1p = W + (unsigned long long)(v1 ? r1 : 0u) * rs + 16u * c;
        s0p = S + (unsigned long long)(v0 ? r0 : 0u) * blocks + (c >> 1);
        s1p = S + (unsigned long long)(v1 ? r1 : 0u) * blocks + (c >> 1);
        wstep = 64u;
        sstep = 2u;
    }
    if constexpr (B128) {
        // The grid row of this CTA's 128-row block (rows 16 cta .. 16 cta + 15),
        // `ldw / 128` scales apart for a K-slice.
        const unsigned long long gs = ((!TILED && ldw != 0u) ? ldw : K) / 128u;
        s0p = S + (unsigned long long)(cta / 8u) * gs * 4u;
        s1p = s0p;
    }

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    MxChunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks)
        mx_load<NT, B128>(cur, w0p, w1p, s0p, s1p, wstep, sstep, A, g, M, K, lda, ch,
                          ch * 64u + 16u * c, v0, v1);
    for (; ch < chunks; ch += MX_WARPS) {
        const unsigned int next = ch + MX_WARPS;
        if (next < chunks)
            mx_load<NT, B128>(nxt, w0p, w1p, s0p, s1p, wstep, sstep, A, g, M, K, lda, next,
                              next * 64u + 16u * c, v0, v1);
        const unsigned int wb0[4] = {cur.w0.x, cur.w0.y, cur.w0.z, cur.w0.w};
        const unsigned int wb1[4] = {cur.w1.x, cur.w1.y, cur.w1.z, cur.w1.w};
        // E4M3 bytes -> a BF16 pair, times the E8M0 scale (MX) or as is (B128).
        const __nv_bfloat162 sc0 = mx_scale(cur.s0), sc1 = mx_scale(cur.s1);
        auto pair = [&](unsigned int lo, unsigned int hi, __nv_bfloat162 sc) {
            __nv_bfloat162 v = __halves2bfloat162(__ushort_as_bfloat16(s_e4m3[lo]),
                                                  __ushort_as_bfloat16(s_e4m3[hi]));
            if constexpr (!B128) {
                v = __hmul2(v, sc);
            } else {
                (void)sc;
            }
            return *reinterpret_cast<unsigned int*>(&v);
        };
        // The B128 chunk partials (unused by the MX tiers).
        [[maybe_unused]] float part[NT][4];
        if constexpr (B128) {
            #pragma unroll
            for (int i = 0; i < NT; i++) part[i][0] = part[i][1] = part[i][2] = part[i][3] = 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Values 4j..4j+3 of each row's 16-value run = bytes 4j..4j+3 = word j.
            const unsigned int x0 = wb0[j], x1 = wb1[j];
            const unsigned int a[4] = {
                pair(x0 & 0xFFu, (x0 >> 8) & 0xFFu, sc0),
                pair(x1 & 0xFFu, (x1 >> 8) & 0xFFu, sc1),
                pair((x0 >> 16) & 0xFFu, x0 >> 24, sc0),
                pair((x1 >> 16) & 0xFFu, x1 >> 24, sc1),
            };
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                const uint4 lo = cur.x[i][0], hi = cur.x[i][1];
                const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                if constexpr (B128) {
                    mx_mma(part[i], a, xw[2 * j], xw[2 * j + 1]);
                } else {
                    mx_mma(acc[i], a, xw[2 * j], xw[2 * j + 1]);
                }
            }
        }
        if constexpr (B128) {
            const float sc = __uint_as_float(cur.s0);
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                #pragma unroll
                for (int q = 0; q < 4; q++) acc[i][q] += part[i][q] * sc;
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
            for (int w = 0; w < MX_WARPS; w++) v[q] += s_red[w][lane][i * 4 + q];
        }
        #pragma unroll
        for (int q = 0; q < (SWIGLU ? 2 : 4); q++) {
            const unsigned int n = (q < 2) ? r0 : r1;
            const unsigned int m = 8u * i + 2u * c + (q & 1u);
            if (n < N && m < M)
                C[(unsigned long long)m * out_stride + n] =
                    SWIGLU ? mx_swiglu(v[q], v[q + 2]) : __float2bfloat16(v[q]);
        }
    }
}

template <int NT, bool TILED = false, bool B128 = false>
__device__ __forceinline__ void mxfp8_gemv_tc_impl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C, unsigned int M,
    unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    mxfp8_gemv_tc_cta<NT, TILED, B128>(A, W, S, C, M, N, K, lda, out_stride, blockIdx.x);
}

// Up to three projections of the same activation rows in one launch (KDA
// q/k/v, MLA q_a + kv_a, an FFN's gate + up): CTAs [0, t0) run projection 0's
// tiles, then projection 1's, then 2's (t_i = ceil(N_i / 16); N_i = 0 skips
// one). Each CTA runs the single-projection body for its tile, so every
// output word is the one its own `_tc{8,16,32}` launch writes. C_i rows are
// N_i apart.
template <int NT, bool TILED, bool B128>
__device__ __forceinline__ void mx_multi_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* W0, const unsigned char* S0, __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* W1, const unsigned char* S1, __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* W2, const unsigned char* S2, __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K
) {
    const unsigned int t0 = (N0 + 15u) / 16u, t1 = (N1 + 15u) / 16u;
    unsigned int b = blockIdx.x;
    const unsigned char *W = W0, *S = S0;
    __nv_bfloat16* C = C0;
    unsigned int N = N0;
    if (b >= t0) {
        b -= t0;
        W = W1; S = S1; C = C1; N = N1;
        if (b >= t1) {
            b -= t1;
            W = W2; S = S2; C = C2; N = N2;
        }
    }
    mxfp8_gemv_tc_cta<NT, TILED, B128>(A, W, S, C, M, N, K, K, N, b);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    atlas_pdl_enter();
    mxfp8_gemv_tc_impl<1>(A, W, S, C, M, N, K, K, out_stride);
}

// `mxfp8_gemv_tc8` whose first `touch_ctas` CTAs pull the first `touch_rows`
// weight rows (values and scales) into L2 while the kernel waits on its PDL
// predecessor (atlas_pdl_touch.cuh). Same body: bit-identical.
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / MX_BLOCK, K / MX_BLOCK}, touch_rows, blockIdx.x,
                          touch_ctas);
    mxfp8_gemv_tc_impl<1>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_impl<2>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_impl<4>(A, W, S, C, M, N, K, K, out_stride);
}

// Touch twins of the 16/32-row tiers (see `mxfp8_gemv_tc8_touch`).
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / MX_BLOCK, K / MX_BLOCK}, touch_rows, blockIdx.x,
                          touch_ctas);
    mxfp8_gemv_tc_impl<2>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / MX_BLOCK, K / MX_BLOCK}, touch_rows, blockIdx.x,
                          touch_ctas);
    mxfp8_gemv_tc_impl<4>(A, W, S, C, M, N, K, K, out_stride);
}

// The plain tiers on a chunk-tiled twin (`mxfp8_tile16`; N % 16 == 0,
// K % 64 == 0): ATLAS_GLM_MXFP8_TILED.
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_tiled(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    atlas_pdl_enter();
    mxfp8_gemv_tc_impl<1, true>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_tiled(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_impl<2, true>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32_tiled(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_impl<4, true>(A, W, S, C, M, N, K, K, out_stride);
}

// Row-major MXFP8 W [N, K] / S [N, K/32] -> the chunk-tiled twin the
// `_tiled` tiers read (N % 16 == 0, K % 64 == 0; S and St null for an FP8
// block twin's values alone): within each 16-row tile,
// chunk ch (K values 64ch..64ch+63) is 1 KiB, lane l = 4g + c's 32 bytes at
// (ch * 32 + l) * 32: values 64ch + 16c.. of row g then of row g + 8; its
// scales are 32 bytes at ch * 32 (row r's two blocks at 2r, 2r + 1). Tiles
// keep their row-major byte ranges. One thread per (tile, chunk, lane).
extern "C" __global__ void mxfp8_tile16(
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
    unsigned char* __restrict__ Wt, unsigned char* __restrict__ St, unsigned int N,
    unsigned int K
) {
    const unsigned int chunks = K / 64u;
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)(N / 16u) * chunks * MX_WARP) return;
    const unsigned int lane = (unsigned int)(i % MX_WARP);
    const unsigned long long tc = i / MX_WARP;  // tile * chunks + ch
    const unsigned int ch = (unsigned int)(tc % chunks);
    const unsigned long long tile = tc / chunks;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned long long r0 = tile * 16u + g, kb = (unsigned long long)ch * 64u + 16u * c;
    uint4* out = (uint4*)(Wt + i * 32u);
    out[0] = *(const uint4*)(W + r0 * K + kb);
    out[1] = *(const uint4*)(W + (r0 + 8u) * K + kb);
    // No scales (S null): an FP8 block twin, whose f32 grid stays row-major.
    if (S == nullptr) return;
    const unsigned long long blocks = K / MX_BLOCK, row = tile * 16u + lane / 2u;
    St[tc * 32u + lane] = S[row * blocks + 2u * ch + (lane & 1u)];
}

// FP8 block weights (B128 above): E4M3 [N, K] + f32 [N/128, K/128], up to
// 8/16/32 rows; `_tiled` on `mxfp8_tile16`'s value tiling (S row-major).
// PDL-entered (ATLAS_PDL); the first `touch_ctas` CTAs ask the first
// `touch_rows` weight rows into L2 during the wait (atlas_pdl_touch.cuh; the
// f32 grid is a few KB and not touched). `touch_rows` 0: plain PDL entry.
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_impl<1, false, true>(A, W, S, C, M, N, K, K, out_stride);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_impl<2, false, true>(A, W, S, C, M, N, K, K, out_stride);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_impl<4, false, true>(A, W, S, C, M, N, K, K, out_stride);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc8_tiled(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_impl<1, true, true>(A, W, S, C, M, N, K, K, out_stride);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc16_tiled(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_impl<2, true, true>(A, W, S, C, M, N, K, K, out_stride);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc32_tiled(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_impl<4, true, true>(A, W, S, C, M, N, K, K, out_stride);
}


// A K-slice of a wider FP8 block weight (row-major): W at the slice's first
// value, S at its first grid column, rows `ldw` values apart (grid rows
// ldw / 128 scales apart; the slice starts and ends on the 128 grid). The
// shared expert's down projection under ATLAS_GLM_SHARED_TP_SPLIT. PDL entry
// and touch as the plain FP8 block tiers.
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc8_ld(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int ldw, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, ldw}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_cta<1, false, true>(A, W, S, C, M, N, K, K, out_stride, blockIdx.x, ldw);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc16_ld(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int ldw, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, ldw}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_cta<2, false, true>(A, W, S, C, M, N, K, K, out_stride, blockIdx.x, ldw);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc32_ld(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int ldw, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, ldw}, {S, 0u, 0u}, min(touch_rows, N), blockIdx.x, touch_ctas);
    mxfp8_gemv_tc_cta<4, false, true>(A, W, S, C, M, N, K, K, out_stride, blockIdx.x, ldw);
}


// Exact MXFP8 -> BF16 expansion of a row-major W [N, K] / S [N, K/32]:
// every E4M3 value times a power of two in range is a BF16 value, so this is
// the checkpoint's math (a transient prefill operand). Eight values a
// thread; `count8` = N * K / 8.
extern "C" __global__ void mxfp8_to_bf16(
    const uint2* __restrict__ W, const unsigned char* __restrict__ S,
    uint4* __restrict__ out, unsigned long long count8
) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count8) return;
    const uint2 v = W[i];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    const __nv_bfloat162 sc = mx_scale(S[i / 4u]);
    unsigned int o[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        __nv_fp8_e4m3 lo, hi;
        lo.__x = b[2 * j];
        hi.__x = b[2 * j + 1];
        __nv_bfloat162 p = __halves2bfloat162(__float2bfloat16((float)lo), __float2bfloat16((float)hi));
        p = __hmul2(p, sc);
        o[j] = *reinterpret_cast<unsigned int*>(&p);
    }
    out[i] = make_uint4(o[0], o[1], o[2], o[3]);
}

// The multi-projection launches (mx_multi_impl) for MXFP8 and FP8 block
// weights, row-major and tiled. Grid: (sum ceil(N_i / 16), 1, 1).
// PDL-entered; each projection's first `touch_ctas` CTAs ask its first
// `touch_rows` rows (values, and MX scales) into L2 during the wait.

// This CTA's (projection, CTA within it, touching CTAs of it) for
// atlas_pdl_enter_touch_tri.
struct MxMultiRole {
    unsigned int sel, cta, ctas;
};

__device__ __forceinline__ MxMultiRole mx_multi_role(
    unsigned int N0, unsigned int N1, unsigned int N2, unsigned int touch_ctas) {
    const unsigned int t0 = (N0 + 15u) / 16u, t1 = (N1 + 15u) / 16u, t2 = (N2 + 15u) / 16u;
    const unsigned int b = blockIdx.x;
    if (b < t0) return {0u, b, min(touch_ctas, t0)};
    if (b < t0 + t1) return {1u, b - t0, min(touch_ctas, t1)};
    return {2u, b - t0 - t1, min(touch_ctas, t2)};
}

// Touch rows no projection lacks (absent projections, N = 0, have no CTAs).
__device__ __forceinline__ unsigned int mx_multi_touch_rows(
    unsigned int rows, unsigned int N0, unsigned int N1, unsigned int N2) {
    if (N0 != 0u) rows = min(rows, N0);
    if (N1 != 0u) rows = min(rows, N1);
    if (N2 != 0u) rows = min(rows, N2);
    return rows;
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_multi(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, K / 32u, K / 32u}, {W1, K, K}, {S1, K / 32u, K / 32u},
                              {W2, K, K}, {S2, K / 32u, K / 32u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<1, false, false>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_multi(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, K / 32u, K / 32u}, {W1, K, K}, {S1, K / 32u, K / 32u},
                              {W2, K, K}, {S2, K / 32u, K / 32u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<2, false, false>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32_multi(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, K / 32u, K / 32u}, {W1, K, K}, {S1, K / 32u, K / 32u},
                              {W2, K, K}, {S2, K / 32u, K / 32u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<4, false, false>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_multi_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, K / 32u, K / 32u}, {W1, K, K}, {S1, K / 32u, K / 32u},
                              {W2, K, K}, {S2, K / 32u, K / 32u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<1, true, false>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_multi_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, K / 32u, K / 32u}, {W1, K, K}, {S1, K / 32u, K / 32u},
                              {W2, K, K}, {S2, K / 32u, K / 32u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<2, true, false>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32_multi_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, K / 32u, K / 32u}, {W1, K, K}, {S1, K / 32u, K / 32u},
                              {W2, K, K}, {S2, K / 32u, K / 32u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<4, true, false>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc8_multi(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, 0u, 0u}, {W1, K, K}, {S1, 0u, 0u},
                              {W2, K, K}, {S2, 0u, 0u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<1, false, true>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc16_multi(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, 0u, 0u}, {W1, K, K}, {S1, 0u, 0u},
                              {W2, K, K}, {S2, 0u, 0u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<2, false, true>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc32_multi(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, 0u, 0u}, {W1, K, K}, {S1, 0u, 0u},
                              {W2, K, K}, {S2, 0u, 0u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<4, false, true>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc8_multi_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, 0u, 0u}, {W1, K, K}, {S1, 0u, 0u},
                              {W2, K, K}, {S2, 0u, 0u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<1, true, true>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc16_multi_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, 0u, 0u}, {W1, K, K}, {S1, 0u, 0u},
                              {W2, K, K}, {S2, 0u, 0u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<2, true, true>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) fp8b_gemv_tc32_multi_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0,
    __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1,
    __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ S2,
    __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch_tri({W0, K, K}, {S0, 0u, 0u}, {W1, K, K}, {S1, 0u, 0u},
                              {W2, K, K}, {S2, 0u, 0u}, mx_multi_touch_rows(touch_rows, N0, N1, N2),
                              mx_multi_role(N0, N1, N2, touch_ctas));
    mx_multi_impl<4, true, true>(A, W0, S0, C0, N0, W1, S1, C1, N1, W2, S2, C2, N2, M, K);
}


template <int NT>
__device__ __forceinline__ void mxfp8_gemv_tc_grouped(
    const __nv_bfloat16* A, const unsigned char* W, const unsigned char* S, __nv_bfloat16* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    atlas_pdl_enter();
    const unsigned long long h = blockIdx.z, nk = (unsigned long long)N * K;
    mxfp8_gemv_tc_impl<NT>(A + h * K, W + h * nk, S + h * (nk / MX_BLOCK), C + h * N,
                           M, N, K, lda, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_grouped(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C, unsigned int M,
    unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    mxfp8_gemv_tc_grouped<1>(A, W, S, C, M, N, K, lda, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_grouped(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C, unsigned int M,
    unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    mxfp8_gemv_tc_grouped<2>(A, W, S, C, M, N, K, lda, out_stride);
}
