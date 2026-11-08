// SPDX-License-Identifier: AGPL-3.0-only
//
// Row-invariant tensor-core skinny GEMMs for the Qwen3.6/3.8-27B decode and
// verify (ATLAS_SKINNY_MMA=1, `crates/spark-model/src/layers/ops/skinny_mma.rs`):
// C[M, N] = A[M, K] · W[N, K]ᵀ for M = 1..16 BF16 rows, one kernel for serial
// decode (M = 1) and every verify width, so a row's bits never depend on M.
//
//   skinny_w4a16_t{1,2}   NVFP4 weight (E2M1 + E4M3 group scale per 16 + FP32
//                         scale2), up to two weights of one input a launch
//                         (gate | up, k | v)
//   skinny_qg_t{1,2}      the interleaved NVFP4 Q+gate weight, output
//                         deinterleaved [Q | G] (as `w4a16_gemv_qg`)
//   skinny_w8a16_t{1,2}   block-scaled FP8 E4M3 weight (FP32 scale per 128 x
//                         128 block), up to two weights a launch
//
// `t1` serves M <= 8, `t2` M <= 16: one or two 8-row token tiles.
//
// MMA MAPPING. `mma.sync.m16n8k16` (BF16 in, FP32 accumulate) takes 16 weight
// rows (outputs) as A and 8 token rows as B, so a CTA owns 16 outputs and a
// token tile is one MMA a k16 step; the second tile (M > 8) is a second MMA
// on the same A fragment. The weight streams once from DRAM whatever M is.
// Lane (g = lane / 4, c = lane % 4) of warp w owns outputs g and g + 8 and,
// in K chunk ch (ch = w, w + 8, ...), a run of RUN consecutive values
// kb = 4 RUN ch + RUN c ..+RUN of those two weight rows and of token rows g
// (and 8 + g): RUN = 64 for NVFP4, 32 for FP8, 32 weight bytes a row either
// way. MMA step j gives k-slots {2c, 2c+1} the run's values 4j + {0, 1} and
// {2c+8, 2c+9} its values 4j + {2, 3}, in A and in B alike, so every product
// pairs the weight and activation of one k. One packed byte
// (two E2M1 codes, low nibble the even k) is one A register; one BF16x2
// activation word is one B register. Nothing is staged in shared memory but
// the final cross-warp combine.
//
// EXACT OPERANDS. Activations are the BF16 rows as they are (no conversion).
// An E2M1 value times its E4M3 group scale has at most 6 significant bits and
// lies in [2^-10, 2688], so the BF16 A operand is exactly that product (times
// 2^-8, `bm_dequant4` / `bm_scales` in common/bf16mma_dequant.cuh); an E4M3
// value is exact in BF16 too (`bm_fp8_pair`, times 2^-8). Accumulation is
// FP32: each MMA starts from zero and its result is added into the running
// FP32 sum with an IEEE add, so the tensor core's internal rounding stays
// within one k16 step. The 2^8, the NVFP4 scale2 and the FP8 block scale are
// applied in FP32 (scale2 once an output, a block scale once a 128-wide FP8
// chunk: the chunk's 8 MMAs summed, then fmaf into the accumulator).
//
// ROW INVARIANCE. An MMA output column depends only on its own B column and
// its A rows, and every other step of an output's sum is fixed by (N, K)
// alone: the chunk -> warp assignment, the MMA order inside a chunk, and the
// warp-order combine (warps 0..7, one fixed `__fadd_rn` chain) — never by M,
// by the row's tile or slot, or by the rows beside it. No atomics, no
// split-K across CTAs. Rows past M load zeros (B columns that touch no other
// column) and are never written. `examples/skinny_mma_microtest.rs` gates
// M-row launches against M one-row launches byte for byte.
//
// Limits (host-checked, `skinny_mma.rs`): K a multiple of the chunk (NVFP4
// 256, FP8 128; it also aligns every load), M in 1..=16, rows A[t] at A + t*K.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "../../common/atlas_pdl.cuh"
#include "../../common/bf16mma_dequant.cuh"

#define SK_WARPS 8u
#define SK_ROWS 16u  // outputs a CTA: the MMA's M

// Per family: a lane's run of consecutive K values a chunk (both weight rows
// get 32 bytes: NVFP4 64 values, FP8 32), the chunk (four lanes' runs), and
// the run's activation vectors a token row. Activations of an NVFP4 chunk are
// read when its MMAs start (from L1/L2: every CTA reads the same rows); the
// FP8 chunk's ride with its weights.
template <bool FP8>
struct SkFam {
    static constexpr unsigned RUN = FP8 ? 32u : 64u;
    static constexpr unsigned KC = 4u * RUN;
    static constexpr int XV = RUN / 8u;                   // uint4 of BF16 a token row
    static constexpr int WV = (FP8 ? RUN : RUN / 2u) / 16u;  // uint4 of a weight row
    static constexpr bool PREFETCH_A = FP8;
};

// Weight bytes stream once: no L1 allocation. Activations take the default
// read-only path. Volatile keeps each load ahead of the previous chunk's
// MMAs, so the next chunk is in flight while they run.
__device__ __forceinline__ uint4 sk_ldw(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned sk_ldw32(const void* p) {
    unsigned v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float sk_ldf(const float* p) {
    float v;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 sk_lda(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// B registers of MMA step j: activation words 2j, 2j+1 of the lane's run.
template <int XV>
__device__ __forceinline__ unsigned sk_b0(const uint4 (&x)[XV], int j) { return (j & 1) ? x[j >> 1].z : x[j >> 1].x; }
template <int XV>
__device__ __forceinline__ unsigned sk_b1(const uint4 (&x)[XV], int j) { return (j & 1) ? x[j >> 1].w : x[j >> 1].y; }

// One chunk's weights of a lane: 32 bytes of rows g and g + 8, and their scales.
struct SkW {
    uint4 w0[2], w1[2];
    unsigned s0, s1;  // NVFP4: the run's four E4M3 group scales of row g / g + 8
    float bs;         // FP8: the chunk's block scale x 2^8
};

// Where a CTA's 16 weight rows live.
struct SkTile {
    const unsigned char* w0p;  // weight row g (NVFP4: packed, K/2 bytes a row; FP8: K bytes)
    const unsigned char* w1p;  // row g + 8
    const unsigned char* s0p;  // NVFP4 group scales of row g (K/16 a row)
    const unsigned char* s1p;
    const float* bsp;          // FP8: the tile's row of block scales (K/128 a row)
    bool v0, v1;               // rows g, g + 8 exist
};

template <bool FP8>
__device__ __forceinline__ void sk_load_w(SkW& t, const SkTile& w, unsigned ch, unsigned c) {
    using F = SkFam<FP8>;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    const unsigned kb = ch * F::KC + F::RUN * c;  // first K value of the lane's run
    const unsigned off = FP8 ? kb : kb / 2u;      // its byte in a weight row
    t.w0[0] = t.w0[1] = t.w1[0] = t.w1[1] = z;
    t.s0 = t.s1 = 0u;
    t.bs = 0.0f;
    if (w.v0) {
#pragma unroll
        for (int i = 0; i < F::WV; i++) t.w0[i] = sk_ldw(w.w0p + off + 16u * i);
        if (!FP8) t.s0 = sk_ldw32(w.s0p + kb / 16u);  // the run's four group scales
    }
    if (w.v1) {
#pragma unroll
        for (int i = 0; i < F::WV; i++) t.w1[i] = sk_ldw(w.w1p + off + 16u * i);
        if (!FP8) t.s1 = sk_ldw32(w.s1p + kb / 16u);
    }
    if (FP8) t.bs = __fmul_rn(sk_ldf(w.bsp + ch), 256.0f);
}

// The lane's activation runs of chunk ch for token rows 8t + g (zeros past M).
template <bool FP8, int NT>
__device__ __forceinline__ void sk_load_a(uint4 (&x)[NT][SkFam<FP8>::XV], const __nv_bfloat16* __restrict__ A,
                                          unsigned K, unsigned M, unsigned g, unsigned ch, unsigned c) {
    using F = SkFam<FP8>;
    const unsigned kb = ch * F::KC + F::RUN * c;
#pragma unroll
    for (int tt = 0; tt < NT; tt++) {
        const unsigned row = 8u * tt + g;
#pragma unroll
        for (int i = 0; i < F::XV; i++) x[tt][i] = make_uint4(0u, 0u, 0u, 0u);
        if (row < M) {
            const __nv_bfloat16* p = A + (unsigned long long)row * K + kb;
#pragma unroll
            for (int i = 0; i < F::XV; i++) x[tt][i] = sk_lda(p + 8 * i);
        }
    }
}

// One MMA from a zero accumulator, its four results then added into acc in
// FP32 (round to nearest): the tensor core's own accumulation rounding stays
// inside one k16 step and never compounds along K.
__device__ __forceinline__ void sk_mma_add(float (&acc)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                           unsigned b0, unsigned b1) {
    float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bm_mma(p, a0, a1, a2, a3, b0, b1);
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = __fadd_rn(acc[i], p[i]);
}

// NVFP4: the chunk's 16 MMAs a token tile, each added into acc.
template <int NT>
__device__ __forceinline__ void sk_mma_w4(float (&acc)[NT][4], const SkW& t, const uint4 (&x)[NT][8]) {
    unsigned sc0[4], sc1[4];  // bf16x2 of scale x 2^118, the run's groups 0..3, rows g / g + 8
    bm_scales(t.s0, sc0);
    bm_scales(t.s1, sc1);
    const unsigned w0[8] = {t.w0[0].x, t.w0[0].y, t.w0[0].z, t.w0[0].w, t.w0[1].x, t.w0[1].y, t.w0[1].z, t.w0[1].w};
    const unsigned w1[8] = {t.w1[0].x, t.w1[0].y, t.w1[0].z, t.w1[0].w, t.w1[1].x, t.w1[1].y, t.w1[1].z, t.w1[1].w};
#pragma unroll
    for (int q = 0; q < 8; q++) {  // word q: bytes 4q..4q+3 (values 8q..8q+7), group q / 2
        const unsigned s0[4] = {sc0[q >> 1], sc0[q >> 1], sc0[q >> 1], sc0[q >> 1]};
        const unsigned s1[4] = {sc1[q >> 1], sc1[q >> 1], sc1[q >> 1], sc1[q >> 1]};
        unsigned d0[4], d1[4];
        bm_dequant4(w0[q], s0, d0);
        bm_dequant4(w1[q], s1, d1);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int j = 2 * q + h;  // bytes 2j (values 4j, 4j+1) and 2j+1 (4j+2, 4j+3)
#pragma unroll
            for (int tt = 0; tt < NT; tt++)
                sk_mma_add(acc[tt], d0[2 * h], d1[2 * h], d0[2 * h + 1], d1[2 * h + 1], sk_b0<8>(x[tt], j),
                           sk_b1<8>(x[tt], j));
        }
    }
}

// FP8: the chunk's 8 MMAs added into a zeroed partial, then acc = fmaf(scale, partial, acc).
template <int NT>
__device__ __forceinline__ void sk_mma_w8(float (&acc)[NT][4], const SkW& t, const uint4 (&x)[NT][4]) {
    const unsigned w0[8] = {t.w0[0].x, t.w0[0].y, t.w0[0].z, t.w0[0].w, t.w0[1].x, t.w0[1].y, t.w0[1].z, t.w0[1].w};
    const unsigned w1[8] = {t.w1[0].x, t.w1[0].y, t.w1[0].z, t.w1[0].w, t.w1[1].x, t.w1[1].y, t.w1[1].z, t.w1[1].w};
    float part[NT][4];
#pragma unroll
    for (int tt = 0; tt < NT; tt++) part[tt][0] = part[tt][1] = part[tt][2] = part[tt][3] = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) {  // word j: values 4j..4j+3 of each row
        const unsigned a0 = bm_fp8_pair<false>(w0[j]), a2 = bm_fp8_pair<true>(w0[j]);
        const unsigned a1 = bm_fp8_pair<false>(w1[j]), a3 = bm_fp8_pair<true>(w1[j]);
#pragma unroll
        for (int tt = 0; tt < NT; tt++)
            sk_mma_add(part[tt], a0, a1, a2, a3, sk_b0<4>(x[tt], j), sk_b1<4>(x[tt], j));
    }
#pragma unroll
    for (int tt = 0; tt < NT; tt++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[tt][i] = fmaf(t.bs, part[tt][i], acc[tt][i]);
}

// Output column of weight row n: identity, or the Q+gate deinterleave
// (interleaved [Q_h(hd) | G_h(hd)] rows -> [Q | G]).
struct SkCol {
    unsigned heads, head_dim;  // 0: identity
    __device__ __forceinline__ unsigned operator()(unsigned n) const {
        if (head_dim == 0u) return n;
        const unsigned group_dim = 2u * head_dim, h = n / group_dim, idx = n % group_dim;
        return idx < head_dim ? h * head_dim + idx : heads * head_dim + h * head_dim + (idx - head_dim);
    }
};

// One weight of a launch, and its output.
struct SkPart {
    const unsigned char* W;  // NVFP4 packed [N, K/2] or FP8 [N, K]
    const unsigned char* S;  // NVFP4 group scales [N, K/16]
    const float* BS;         // FP8 block scales [ceil(N/128), K/128]
    float fscale;            // FP32 factor an output (NVFP4: scale2 x 2^8; FP8: 1)
    __nv_bfloat16* C;        // output row t at C + t * c_stride
    unsigned N;
    SkCol col;
};

// The weight rows of global tile `tile` (part 0's tiles first).
template <bool FP8>
__device__ __forceinline__ SkTile sk_tile_at(const SkPart& p0, const SkPart& p1, unsigned tiles0, unsigned K,
                                             unsigned tile, unsigned g, unsigned& part, unsigned& r0) {
    part = tile < tiles0 ? 0u : 1u;
    const SkPart& p = part ? p1 : p0;
    r0 = (tile - (part ? tiles0 : 0u)) * SK_ROWS + g;
    const unsigned r1 = r0 + 8u;
    SkTile w;
    w.v0 = r0 < p.N;
    w.v1 = r1 < p.N;
    const unsigned long long row_bytes = FP8 ? K : K / 2u;
    w.w0p = p.W + (unsigned long long)(w.v0 ? r0 : 0u) * row_bytes;
    w.w1p = p.W + (unsigned long long)(w.v1 ? r1 : 0u) * row_bytes;
    w.s0p = FP8 ? nullptr : p.S + (unsigned long long)(w.v0 ? r0 : 0u) * (K / 16u);
    w.s1p = FP8 ? nullptr : p.S + (unsigned long long)(w.v1 ? r1 : 0u) * (K / 16u);
    // A tile's 16 rows sit in one 128-row block (16 divides 128).
    w.bsp = FP8 ? p.BS + (unsigned long long)((r0 - g) / 128u) * (K / 128u) : nullptr;
    return w;
}

// CTA blockIdx.x: its tile's 16 outputs over all K for all M rows. Warp w
// takes chunks w, w + 8, ... with its next chunk in flight; the 8 warps'
// sums combine in warp order through shared memory.
template <bool FP8, int NT>
__device__ __forceinline__ void sk_run(const __nv_bfloat16* __restrict__ A, unsigned M, unsigned K,
                                       unsigned c_stride, const SkPart& p0, const SkPart& p1) {
    using F = SkFam<FP8>;
    __shared__ float s_red[SK_WARPS][4 * NT][32];
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const unsigned g = lane >> 2, c = lane & 3u;
    const unsigned chunks = K / F::KC;
    unsigned part, r0;
    const SkTile w = sk_tile_at<FP8>(p0, p1, (p0.N + SK_ROWS - 1u) / SK_ROWS, K, blockIdx.x, g, part, r0);
    float acc[NT][4];
#pragma unroll
    for (int tt = 0; tt < NT; tt++) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.0f;
    SkW cur, nxt;
    uint4 xc[NT][F::XV], xn[NT][F::XV];
    if (warp < chunks) {
        sk_load_w<FP8>(cur, w, warp, c);
        if (F::PREFETCH_A) sk_load_a<FP8, NT>(xc, A, K, M, g, warp, c);
    }
    nxt = cur;
    for (unsigned ch = warp; ch < chunks; ch += SK_WARPS) {
        const bool more = ch + SK_WARPS < chunks;
        if (more) {
            sk_load_w<FP8>(nxt, w, ch + SK_WARPS, c);
            if (F::PREFETCH_A) sk_load_a<FP8, NT>(xn, A, K, M, g, ch + SK_WARPS, c);
        }
        if (!F::PREFETCH_A) sk_load_a<FP8, NT>(xc, A, K, M, g, ch, c);
        if constexpr (FP8) sk_mma_w8<NT>(acc, cur, xc);
        else sk_mma_w4<NT>(acc, cur, xc);
        cur = nxt;
        if (F::PREFETCH_A && more) {
#pragma unroll
            for (int tt = 0; tt < NT; tt++)
#pragma unroll
                for (int i = 0; i < F::XV; i++) xc[tt][i] = xn[tt][i];
        }
    }
    // Fixed-order combine of the 8 warps' K slices.
#pragma unroll
    for (int tt = 0; tt < NT; tt++)
#pragma unroll
        for (int i = 0; i < 4; i++) s_red[warp][4 * tt + i][lane] = acc[tt][i];
    __syncthreads();
    if (warp != 0u) return;
    const SkPart& p = part ? p1 : p0;
#pragma unroll
    for (int tt = 0; tt < NT; tt++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            // C fragment: acc[0..1] -> output g, acc[2..3] -> g + 8; token
            // columns 2c, 2c + 1 of tile tt.
            const unsigned n = i < 2 ? r0 : r0 + 8u;
            const unsigned m = 8u * tt + 2u * c + (i & 1u);
            float v = 0.0f;
#pragma unroll
            for (unsigned ww = 0; ww < SK_WARPS; ww++) v = __fadd_rn(v, s_red[ww][4 * tt + i][lane]);
            if (n < p.N && m < M)
                p.C[(unsigned long long)m * c_stride + p.col(n)] = __float2bfloat16_rn(__fmul_rn(v, p.fscale));
        }
}

// ── Entry points ────────────────────────────────────────────────────────────
// Grid: (ceil(N0/16) + ceil(N1/16), 1, 1), block 256; part 1 (N1 = 0: none)
// owns the CTAs past part 0's, and each output is the word a launch of its
// own weight alone writes. Output row t of part p at C_p + t * c_stride.
// One CTA an SM: the kernels are latency-bound streams, and the register
// budget of `__launch_bounds__(256, 1)` lets the compiler keep more of each
// warp's loads in flight than two smaller CTAs would (measured on GB10).

template <int NT>
__device__ __forceinline__ void sk_w4a16(const __nv_bfloat16* __restrict__ A, unsigned M, unsigned K,
                                         unsigned c_stride, const unsigned char* __restrict__ P0,
                                         const unsigned char* __restrict__ S0, float s0,
                                         __nv_bfloat16* __restrict__ C0, unsigned N0,
                                         const unsigned char* __restrict__ P1,
                                         const unsigned char* __restrict__ S1, float s1,
                                         __nv_bfloat16* __restrict__ C1, unsigned N1) {
    atlas_pdl_enter();
    const SkPart p0{P0, S0, nullptr, __fmul_rn(s0, 256.0f), C0, N0, SkCol{0u, 0u}};
    const SkPart p1{P1, S1, nullptr, __fmul_rn(s1, 256.0f), C1, N1, SkCol{0u, 0u}};
    sk_run<false, NT>(A, M, K, c_stride, p0, p1);
}

#define SK_W4_PARAMS                                                                                 \
    const __nv_bfloat16 *__restrict__ A, unsigned M, unsigned K, unsigned c_stride,                  \
        const unsigned char *__restrict__ P0, const unsigned char *__restrict__ S0, float s0,        \
        __nv_bfloat16 *__restrict__ C0, unsigned N0, const unsigned char *__restrict__ P1,           \
        const unsigned char *__restrict__ S1, float s1, __nv_bfloat16 *__restrict__ C1, unsigned N1
#define SK_W4_ARGS A, M, K, c_stride, P0, S0, s0, C0, N0, P1, S1, s1, C1, N1

extern "C" __global__ void __launch_bounds__(SK_WARPS * 32, 1) skinny_w4a16_t1(SK_W4_PARAMS) {
    sk_w4a16<1>(SK_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(SK_WARPS * 32, 1) skinny_w4a16_t2(SK_W4_PARAMS) {
    sk_w4a16<2>(SK_W4_ARGS);
}

// Q+gate: N = heads x head_dim x 2 interleaved rows, C row t at C + t * N, [Q | G].
#define SK_QG_PARAMS                                                                                 \
    const __nv_bfloat16 *__restrict__ A, unsigned M, unsigned K, const unsigned char *__restrict__ P, \
        const unsigned char *__restrict__ S, float s2, __nv_bfloat16 *__restrict__ C, unsigned N,     \
        unsigned heads, unsigned head_dim

template <int NT>
__device__ __forceinline__ void sk_qg(SK_QG_PARAMS) {
    atlas_pdl_enter();
    const SkPart p{P, S, nullptr, __fmul_rn(s2, 256.0f), C, N, SkCol{heads, head_dim}};
    sk_run<false, NT>(A, M, K, N, p, p);
}

extern "C" __global__ void __launch_bounds__(SK_WARPS * 32, 1) skinny_qg_t1(SK_QG_PARAMS) {
    sk_qg<1>(A, M, K, P, S, s2, C, N, heads, head_dim);
}
extern "C" __global__ void __launch_bounds__(SK_WARPS * 32, 1) skinny_qg_t2(SK_QG_PARAMS) {
    sk_qg<2>(A, M, K, P, S, s2, C, N, heads, head_dim);
}

// FP8: block_scale [ceil(N/128), K/128] FP32 a weight.
#define SK_W8_PARAMS                                                                                 \
    const __nv_bfloat16 *__restrict__ A, unsigned M, unsigned K, unsigned c_stride,                  \
        const unsigned char *__restrict__ B0, const float *__restrict__ BS0,                         \
        __nv_bfloat16 *__restrict__ C0, unsigned N0, const unsigned char *__restrict__ B1,           \
        const float *__restrict__ BS1, __nv_bfloat16 *__restrict__ C1, unsigned N1

template <int NT>
__device__ __forceinline__ void sk_w8a16(SK_W8_PARAMS) {
    atlas_pdl_enter();
    const SkPart p0{B0, nullptr, BS0, 1.0f, C0, N0, SkCol{0u, 0u}};
    const SkPart p1{B1, nullptr, BS1, 1.0f, C1, N1, SkCol{0u, 0u}};
    sk_run<true, NT>(A, M, K, c_stride, p0, p1);
}

extern "C" __global__ void __launch_bounds__(SK_WARPS * 32, 1) skinny_w8a16_t1(SK_W8_PARAMS) {
    sk_w8a16<1>(A, M, K, c_stride, B0, BS0, C0, N0, B1, BS1, C1, N1);
}
extern "C" __global__ void __launch_bounds__(SK_WARPS * 32, 1) skinny_w8a16_t2(SK_W8_PARAMS) {
    sk_w8a16<2>(A, M, K, c_stride, B0, BS0, C0, N0, B1, BS1, C1, N1);
}
