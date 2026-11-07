// SPDX-License-Identifier: AGPL-3.0-only
//
// Dense prefill GEMMs on BF16 tensor cores with register dequant: the NVFP4
// MLP (gate+up with SiLU*up in the epilogue, down) and the per-tensor FP8
// projections (GDN in_proj_qkvz / out_proj) of the Qwen3.6/3.8-27B hybrids.
// The dense sibling of moe_bf16mma_prefill.cu; see that file for the design.
//
//  * Weights are the MMA A operand (16 output features x k16), tokens the B
//    operand (k16 x 8 tokens). A CTA is two 128-thread halves of four warps;
//    each warp owns 32 features and every token tile of a <= 64-row chunk,
//    so each weight value is dequantized once per CTA. The halves share the
//    staged token tile.
//  * K order inside a 64-wide stage is permuted so that thread t of a quad
//    owns k = 16t..16t+15 (one NVFP4 scale group); A and B use the same
//    permutation, so the dot products are unchanged.
//  * NVFP4 weights are the transposed prefill tables ([K/2, N] e2m1,
//    [K/16, N] e4m3 block scales, FP32 tensor scale), as the W4A16 prefill
//    kernels read. FP8 weights are the checkpoint [N, K] e4m3 rows with a
//    per-128-row FP32 scale (the block-scale grid's first column; the host
//    only routes weights whose grid is constant along K).
//  * The grid is 1D and rasterized in groups of DB_GROUP row chunks: a
//    group's CTAs walk the features with the chunk index fastest, so a
//    weight tile is read from DRAM once per group and the group's tokens
//    stay in L2.
//
// Activations stay BF16 (the FP8-MMA W4A16 kernels round them to E4M3, the
// block-scaled FP8 GEMM quantizes them per 128), so this path is at least
// as precise. Every output element is computed by one CTA with a fixed K
// order: results do not depend on the grid.
//
// Low-precision activations (lever ATLAS_DENSE_PREFILL_LOWP, same CTA tiling,
// raster and weight staging; the activations are quantized per token by
// dl_quant_* below):
//  * W4A4: NVFP4 weights x NVFP4 tokens on mma.sync kind::mxf4nvf4
//    m16n8k64, both block scales (E4M3 per 16 K) applied by the MMA. The
//    weights are the checkpoint's bytes; tokens carry a per-token power-of-
//    two global scale applied in the epilogue with the weight tensor scale.
//  * W8A8: per-tensor FP8 weights x E4M3 tokens (per-token power-of-two
//    scale) on m16n8k32 e4m3.

#include "../../common/atlas_pdl.cuh"
#include "../../common/bf16mma_dequant.cuh"
#include "../../common/nvfp4_block_quant.cuh"

#define DB_ROWS 64
#define DB_KS 64
#define DB_GROUP 16     // row chunks per raster group

// DB_FP8T: the transposed [K, N] copy of a per-tensor FP8 weight (the
// w8a16_gemm_t prefill tables): each stage streams 64 rows of 256
// contiguous features instead of 256 rows of 64 bytes.
enum { DB_NVFP4 = 0, DB_FP8 = 1, DB_FP8T = 2 };

// One K stage of staged weights: NVFP4 weight and scale rows swizzled as
// bm_swz_w / bm_swz_s; FP8 64-byte k rows per feature, feature f at row f
// with bits 0 and 2 swapped, so that in each quarter warp the four slot-s
// rows of threads g = 2q, 2q+1 fall in different 64-byte halves of a
// 128-byte line (the LDS.128 of a quarter warp covers all 32 banks).
template <int WF, int HV> struct DbW;
template <int HV> struct __align__(16) DbW<DB_NVFP4, HV> {
    unsigned char w[HV][DB_KS / 2][128];            // 4096 B each: [k pair][feature]
    unsigned char s[HV][DB_KS / 16][128];           //  512 B each: [group][feature]
};
// FP8T: [k][feature] 128-byte rows per half, 16-byte chunk c of row r at
// chunk c ^ dl_swz_f8t(r): the four quad lanes read rows 16 apart.
template <int HV> struct __align__(16) DbW<DB_FP8T, HV> {
    unsigned char w[HV][DB_KS][128];                // 8192 B each: [k][feature]
};
__device__ __forceinline__ unsigned int dl_swz_f8t(unsigned int r) { return ((r >> 4) & 3) << 1; }
template <int HV> struct __align__(16) DbW<DB_FP8, HV> {
    unsigned char w[HV][128][DB_KS];                // 8192 B each: [feature][k]
};
// One K stage of tokens, 16-byte chunks swizzled (db_tok).
struct __align__(16) DbTok {
    __nv_bfloat16 tok[DB_ROWS][DB_KS];              // 8192 B
};

// Token rows are 128 bytes; 16-byte chunk c of row r lives at chunk c ^ (r & 7),
// so the two rows a quarter warp reads (g = 2q, 2q+1) hit different banks.
__device__ __forceinline__ unsigned int db_tok(unsigned int r, unsigned int c) { return ((c ^ (r & 7)) * 8); }
__device__ __forceinline__ unsigned int db_fp8_row(unsigned int f) {
    return (f & ~5u) | ((f & 1) << 2) | ((f >> 2) & 1);
}

// Token formats: BF16, NVFP4 (packed e2m1 [M, K/2], E4M3 scales [M, K/16],
// FP32 per-token scale), E4M3 (bytes [M, K], FP32 per-token scale). The
// low-precision kernels take 128-row chunks (one CTA per SM, 16 token tiles
// per warp: each weight tile is read by half as many CTAs).
// MX8: E4M3 tokens with a UE8M0 scale per token and 32-K MMA block (two per
// 64-K stage: k % 16 in [0, 8) and [8, 16)), written by the W4A8 gate+up
// epilogue for the down projection (dense_w4a8_gate_up_silu_mx); scales
// [M, K/32] bytes, byte 2 s + h of a row for stage s, MMA half h.
enum { DB_A_BF16 = 0, DB_A_NVFP4 = 1, DB_A_FP8 = 2, DB_A_MX8 = 3 };
template <int AF> __host__ __device__ constexpr bool dl_a8() { return AF == DB_A_FP8 || AF == DB_A_MX8; }
// MX8 scales are staged 8 stages (16 bytes a token) at a time, two groups.
#define DL_MX_SG 8
#define DL_ROWS 128
template <int AF> __host__ __device__ constexpr int db_rows() { return AF == DB_A_BF16 ? DB_ROWS : DL_ROWS; }
template <int AF> struct DbT;
template <> struct DbT<DB_A_BF16> { using T = DbTok; };
// NVFP4: 32-byte rows, the two 16-byte chunks swapped on rows 4..7 of each
// 8 (dl_tok4) so a quarter warp's b0 / b1 reads hit 32 banks.
struct __align__(16) DbTok4 {
    unsigned char q[DL_ROWS][DB_KS / 2];            // 4096 B
    unsigned int s[DL_ROWS];                        //  512 B: the 4 group scales
};
template <> struct DbT<DB_A_NVFP4> { using T = DbTok4; };
struct __align__(16) DbTok8 {
    unsigned char q[DL_ROWS][DB_KS];                // 8192 B
};
template <> struct DbT<DB_A_FP8> { using T = DbTok8; };
template <> struct DbT<DB_A_MX8> { using T = DbTok8; };
__device__ __forceinline__ unsigned int dl_tok4(unsigned int r, unsigned int c) { return (c ^ ((r >> 2) & 1)) * 16; }
// W4A4 weight rows: thread t reads k-pair rows 4t + i and 16 + 4t + i, so
// the swizzle keys on bits 2..3 of the row (bm_swz_w keys on 3..4).
__device__ __forceinline__ unsigned int dl_swz_w(unsigned int kp) { return ((kp >> 2) & 3) << 1; }
template <int AF> __device__ __forceinline__ unsigned int db_swz_w(unsigned int kp) {
    return AF == DB_A_NVFP4 ? dl_swz_w(kp) : bm_swz_w(kp);
}

// Bytes b of four words -> word b: a 4x4 byte transpose.
__device__ __forceinline__ void dl_transpose4(const unsigned int (&w)[4], unsigned int (&o)[4]) {
    const unsigned int t0 = __byte_perm(w[0], w[1], 0x5140), t1 = __byte_perm(w[0], w[1], 0x7362);
    const unsigned int t2 = __byte_perm(w[2], w[3], 0x5140), t3 = __byte_perm(w[2], w[3], 0x7362);
    o[0] = __byte_perm(t0, t2, 0x5410);
    o[1] = __byte_perm(t0, t2, 0x7632);
    o[2] = __byte_perm(t1, t3, 0x5410);
    o[3] = __byte_perm(t1, t3, 0x7632);
}

__device__ __forceinline__ void dl_cp4(void* dst, const void* src, bool pred) {
    unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(pred ? 4 : 0));
}

// mbarriers for the low-precision pipeline.
__device__ __forceinline__ void dl_mbar_init(unsigned long long* b, unsigned int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((unsigned int)__cvta_generic_to_shared(b)), "r"(count));
}
// Arrives on b once this thread's cp.async copies so far have landed.
__device__ __forceinline__ void dl_cp_arrive(unsigned long long* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"((unsigned int)__cvta_generic_to_shared(b)));
}
__device__ __forceinline__ void dl_mbar_arrive(unsigned long long* b) {
    asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared.b64 st, [%0];\n\t}"
                 :: "r"((unsigned int)__cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void dl_mbar_wait(unsigned long long* b, unsigned int parity) {
    asm volatile("{\n\t.reg .pred p;\n\tWAIT_%=:\n\t"
                 "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t@!p bra WAIT_%=;\n\t}"
                 :: "r"((unsigned int)__cvta_generic_to_shared(b)), "r"(parity) : "memory");
}

__device__ __forceinline__ void dl_mma_f4(float c[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1,
                                          unsigned int sa, unsigned int sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// mxf8f6f4 block-scaled E4M3 MMA: one UE8M0 scale per A row (thread t even:
// row g, odd: g + 8), B unscaled (2^0).
__device__ __forceinline__ void dl_mma_mx8(float c[4], unsigned int a0, unsigned int a1, unsigned int a2,
                                           unsigned int a3, unsigned int b0, unsigned int b1, unsigned int sa,
                                           unsigned int sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// Four e2m1 pairs (bytes of w, low nibble = even k) -> four f16x2, exact.
__device__ __forceinline__ void dl_e2m1_f16(unsigned int w, unsigned int out[4]) {
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]) : "r"(w));
}

// f16x2 -> two E4M3 bytes (low half in the low byte), round to nearest.
__device__ __forceinline__ unsigned int dl_e4m3x2(unsigned int h) {
    unsigned short d;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(h));
    return d;
}

__device__ __forceinline__ unsigned int dl_hmul2(unsigned int a, unsigned int b) {
    unsigned int d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ void dl_mma_f8(float c[4], unsigned int a0, unsigned int a1, unsigned int a2,
                                          unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four MMAs per token tile of one k16 pair h: d[i][slot] holds k pair
// 16t + 8h + 2i of the four slots. Token fragments load DB_BG tiles ahead
// (4: 8 spills more registers at the 128-register cap of two CTAs per SM).
#define DB_BG 4
template <int TT>
__device__ __forceinline__ void db_mmas(const __nv_bfloat16 (*tok)[DB_KS], const unsigned int (&d)[4][4],
                                        float (&acc)[2][8][4], unsigned int g, unsigned int t, int h) {
    #pragma unroll
    for (int t0 = 0; t0 < TT; t0 += DB_BG) {
        uint4 b[DB_BG];
        #pragma unroll
        for (int j = 0; j < DB_BG; j++)
            if (t0 + j < TT) b[j] = *(const uint4*)&tok[(t0 + j) * 8 + g][db_tok(g, 2 * t + h)];
        // The two k16 steps of an accumulator go in separate passes, so that
        // consecutive MMAs are independent.
        #pragma unroll
        for (int kk = 0; kk < 2; kk++)
            #pragma unroll
            for (int j = 0; j < DB_BG; j++)
                if (t0 + j < TT)
                    #pragma unroll
                    for (int f = 0; f < 2; f++)
                        bm_mma(acc[f][t0 + j], d[2 * kk][2 * f], d[2 * kk][2 * f + 1], d[2 * kk + 1][2 * f],
                               d[2 * kk + 1][2 * f + 1], kk ? b[j].z : b[j].x, kk ? b[j].w : b[j].y);
    }
}

// One 64-wide K stage for TT token tiles into acc[f][0..TT).
template <int WF, bool FUSED, int TT, int HV>
__device__ __forceinline__ void db_stage(const DbTok& tk, const DbW<WF, HV>& sb, unsigned int half,
                                         float (&acc)[2][8][4], unsigned int warp, unsigned int g,
                                         unsigned int t) {
    if constexpr (WF == DB_NVFP4) {
        unsigned int sc[4];
        bm_scales(bm_slots<FUSED>(sb.s[half][t], bm_swz_s(t), warp, g), sc);
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int d[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int kp = 8 * t + 4 * h + i;
                bm_dequant4(bm_slots<FUSED>(sb.w[half][kp], bm_swz_w(kp), warp, g), sc, d[i]);
            }
            db_mmas<TT>(tk.tok, d, acc, g, t, h);
        }
    } else {
        uint4 q[4];
        #pragma unroll
        for (int s = 0; s < 4; s++)
            q[s] = *(const uint4*)&sb.w[half][db_fp8_row(32 * warp + 4 * g + s)][16 * t];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int d[4][4];
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                const unsigned int w0 = h ? q[s].z : q[s].x, w1 = h ? q[s].w : q[s].y;
                d[0][s] = bm_fp8_pair<false>(w0);
                d[1][s] = bm_fp8_pair<true>(w0);
                d[2][s] = bm_fp8_pair<false>(w1);
                d[3][s] = bm_fp8_pair<true>(w1);
            }
            db_mmas<TT>(tk.tok, d, acc, g, t, h);
        }
    }
}

// The smallest power of two p with amax / p <= lim; 1 for amax == 0.
__device__ __forceinline__ float dl_pow2_scale(float amax, float lim) {
    if (!(amax > 0.0f)) return 1.0f;
    int e;
    const float f = frexpf(__fdiv_rn(amax, lim), &e);   // f in [0.5, 1)
    return ldexpf(1.0f, f == 0.5f ? e - 1 : e);
}

template <int N> struct dl_int { static constexpr int value = N; };

// W4A8 weights of one stage, re-encoded per thread: the raw transposed
// words f[h][slot] (k 16t+8h .. 16t+8h+7), the f16 scales s / 2^E, the
// E4M3 MMA operands a[h][slot] and the UE8M0 row scales of the two
// fragments. Split so the next stage's re-encode can interleave with this
// stage's MMAs (dl_body's W4A8 loop).
struct DlA8 {
    unsigned int f[2][4], sh[4], a[2][4][2], sa0, sa1, ue[4];
};

template <bool FUSED, int HV>
__device__ __forceinline__ void dl_a8_begin(DlA8& e, const void* sbv, unsigned int half, unsigned int warp,
                                            unsigned int g, unsigned int t) {
    const DbW<DB_NVFP4, HV>& sb = *(const DbW<DB_NVFP4, HV>*)sbv;
    // Group t's scale for the four slots, and the stage maximum per slot
    // (positive E4M3 bytes order as their values).
    const unsigned int s4 = bm_slots<FUSED>(sb.s[half][t], bm_swz_s(t), warp, g);
    unsigned int m4 = __vmaxu4(s4, __shfl_xor_sync(0xffffffffu, s4, 1));
    m4 = __vmaxu4(m4, __shfl_xor_sync(0xffffffffu, m4, 2));
    unsigned int (&ue)[4] = e.ue;
    #pragma unroll
    for (int sl = 0; sl < 4; sl++) {
        // E = ceil(log2(6 * max / 448)) from the FP32 bits (max * 6 / 448
        // is a normal float for every nonzero E4M3 scale); 0 for max 0.
        const unsigned int xb = __float_as_uint(nvfp4q_e4m3_value((m4 >> (8 * sl)) & 0xFF) * (6.0f / 448.0f));
        const int ex = xb ? (int)(xb >> 23) - 127 + ((xb & 0x7FFFFFu) != 0) : 0;
        ue[sl] = (unsigned int)(127 + ex);
        // s / 2^E in f16: <= 448 / 6, and exact wherever its values
        // reach E4M3 (below 2^-14 they round to zero either way).
        const float sv = nvfp4q_e4m3_value((s4 >> (8 * sl)) & 0xFF) * __uint_as_float((unsigned int)(127 - ex) << 23);
        const unsigned int hb = __half_as_ushort(__float2half_rn(sv));
        e.sh[sl] = hb | (hb << 16);
    }
    const unsigned int odd = t & 1;
    e.sa0 = odd ? ue[1] : ue[0];
    e.sa1 = odd ? ue[3] : ue[2];
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned int w[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned int kp = 8 * t + 4 * h + i;
            w[i] = bm_slots<FUSED>(sb.w[half][kp], bm_swz_w(kp), warp, g);
        }
        dl_transpose4(w, e.f[h]);
    }
}

__device__ __forceinline__ void dl_a8_slot(DlA8& e, int sl) {
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned int x[4], p[4];
        dl_e2m1_f16(e.f[h][sl], x);
        #pragma unroll
        for (int j = 0; j < 4; j++) p[j] = dl_e4m3x2(dl_hmul2(x[j], e.sh[sl]));
        e.a[h][sl][0] = p[0] | (p[1] << 16);
        e.a[h][sl][1] = p[2] | (p[3] << 16);
    }
}

// The MMAs of token tiles [J0, J1) of a W4A8 stage.
// mxs: MX8 token scales of this stage (token r's at mxs[16 r], halves 0 and
// 1), or null for unscaled tokens.
template <int J0, int J1, int TTM>
__device__ __forceinline__ void dl_a8_mma(const DbTok8& tk, const DlA8& e, float (&acc)[2][TTM][4], unsigned int g,
                                          unsigned int t, const unsigned char* mxs) {
    #pragma unroll
    for (int j = J0; j < J1; j++) {
        const uint4 b = *(const uint4*)&tk.q[j * 8 + g][16 * t];
        const unsigned int bs = mxs ? *(const unsigned short*)&mxs[16 * (j * 8 + g)] : 0x7f7fu;
        #pragma unroll
        for (int f = 0; f < 2; f++) {
            const unsigned int sa = f ? e.sa1 : e.sa0;
            dl_mma_mx8(acc[f][j], e.a[0][2 * f][0], e.a[0][2 * f + 1][0], e.a[0][2 * f][1], e.a[0][2 * f + 1][1], b.x,
                       b.y, sa, bs & 0xff);
            dl_mma_mx8(acc[f][j], e.a[1][2 * f][0], e.a[1][2 * f + 1][0], e.a[1][2 * f][1], e.a[1][2 * f + 1][1], b.z,
                       b.w, sa, bs >> 8);
        }
    }
}

// One 64-wide K stage of the low-precision paths for TT token tiles.
//  W4A4: the MMA's K is the stage's K in order. Thread t's A registers are
//  k-pair rows 4t..4t+3 (a0, a1) and 16+4t..16+4t+3 (a2, a3) of its four
//  slot features, byte-transposed into one word per feature; its scale word
//  holds a feature's four group scales (thread t even: MMA row g, odd: g+8).
//  W8A8: thread t owns k 16t..16t+15 of each feature and token (any K order
//  works without scale groups): MMA 0 takes bytes 0..7, MMA 1 bytes 8..15.
//  W4A8: the BF16 path's K order (thread t owns k 16t..16t+15, MMA h takes
//  16t+8h..16t+8h+7). Each feature's NVFP4 values are re-encoded per stage
//  as E4M3 of w * s / 2^E, 2^E the smallest power of two that fits the
//  stage's largest block scale * 6 under 448, and the MMA applies 2^E as a
//  UE8M0 scale: values with at most 4 significant bits are exact, the rest
//  round once to E4M3.
template <int WF, int AF, bool FUSED, int TT, int HV, int TTM>
__device__ __forceinline__ void dl_stage(const typename DbT<AF>::T& tk, const void* sbv, unsigned int half,
                                         float (&acc)[2][TTM][4], unsigned int warp, unsigned int g,
                                         unsigned int t) {
    if constexpr (AF == DB_A_FP8 && WF == DB_NVFP4) {
        DlA8 e;
        dl_a8_begin<FUSED, HV>(e, sbv, half, warp, g, t);
        #pragma unroll
        for (int sl = 0; sl < 4; sl++) dl_a8_slot(e, sl);
        dl_a8_mma<0, TT, TTM>(tk, e, acc, g, t, nullptr);
    } else if constexpr (AF == DB_A_NVFP4) {
        const DbW<DB_NVFP4, HV>& sb = *(const DbW<DB_NVFP4, HV>*)sbv;
        unsigned int a[2][4], sc[4];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int w[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int kp = 16 * h + 4 * t + i;
                w[i] = bm_slots<FUSED>(sb.w[half][kp], dl_swz_w(kp), warp, g);
            }
            dl_transpose4(w, a[h]);
        }
        {
            unsigned int w[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) w[i] = bm_slots<FUSED>(sb.s[half][i], bm_swz_s(i), warp, g);
            dl_transpose4(w, sc);
        }
        const unsigned int odd = t & 1;
        const unsigned int f0[4] = {a[0][0], a[0][1], a[1][0], a[1][1]};
        const unsigned int f1[4] = {a[0][2], a[0][3], a[1][2], a[1][3]};
        const unsigned int s0 = odd ? sc[1] : sc[0], s1 = odd ? sc[3] : sc[2];
        #pragma unroll
        for (int j = 0; j < TT; j++) {
            const unsigned int r = j * 8 + g;
            const unsigned int b0 = *(const unsigned int*)&tk.q[r][dl_tok4(r, 0) + 4 * t];
            const unsigned int b1 = *(const unsigned int*)&tk.q[r][dl_tok4(r, 1) + 4 * t];
            const unsigned int bs = tk.s[r];
            dl_mma_f4(acc[0][j], f0, b0, b1, s0, bs);
            dl_mma_f4(acc[1][j], f1, b0, b1, s1, bs);
        }
    } else {
        uint4 q[4];
        if constexpr (WF == DB_FP8T) {
            // k rows 16t + 4m + i of the four slot features, transposed so
            // that word m of slot s is k 16t + 4m .. 16t + 4m + 3.
            const DbW<DB_FP8T, HV>& sb = *(const DbW<DB_FP8T, HV>*)sbv;
            const unsigned int o = 32 * warp + 4 * g;
            unsigned int m4[4][4];
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                unsigned int w[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const unsigned int r = 16 * t + 4 * m + i;
                    w[i] = *(const unsigned int*)&sb.w[half][r][((o >> 4) ^ dl_swz_f8t(r)) * 16 + (o & 15)];
                }
                dl_transpose4(w, m4[m]);
            }
            #pragma unroll
            for (int sl = 0; sl < 4; sl++) q[sl] = make_uint4(m4[0][sl], m4[1][sl], m4[2][sl], m4[3][sl]);
        } else {
            const DbW<DB_FP8, HV>& sb = *(const DbW<DB_FP8, HV>*)sbv;
            #pragma unroll
            for (int s = 0; s < 4; s++)
                q[s] = *(const uint4*)&sb.w[half][db_fp8_row(32 * warp + 4 * g + s)][16 * t];
        }
        #pragma unroll
        for (int j = 0; j < TT; j++) {
            const uint4 b = *(const uint4*)&tk.q[j * 8 + g][16 * t];
            #pragma unroll
            for (int f = 0; f < 2; f++) {
                dl_mma_f8(acc[f][j], q[2 * f].x, q[2 * f + 1].x, q[2 * f].y, q[2 * f + 1].y, b.x, b.y);
                dl_mma_f8(acc[f][j], q[2 * f].z, q[2 * f + 1].z, q[2 * f].w, q[2 * f + 1].w, b.z, b.w);
            }
        }
    }
}

// Dynamic shared memory: two token stages and WST weight stages, or the
// epilogue's [64 tokens][features + 8] BF16 tile if larger.
// Token stages: two for BF16 tokens (L2 hits, issued one stage ahead), WST
// for the low-precision ones (issued with the weights, WST - 1 ahead).
template <int AF, int WST> __host__ __device__ constexpr int db_tst() { return AF == DB_A_BF16 ? 2 : WST; }

template <int WF, bool FUSED, int HV, int WST, int AF>
__host__ __device__ constexpr unsigned int db_smem_bytes() {
    constexpr unsigned int stages = db_tst<AF, WST>() * sizeof(typename DbT<AF>::T) + WST * sizeof(DbW<WF, HV>)
                                    + (AF == DB_A_MX8 ? 2 * DL_ROWS * 2 * DL_MX_SG : 0);
    constexpr unsigned int out = db_rows<AF>() * ((FUSED ? 64 : 128) * HV + 8) * sizeof(__nv_bfloat16);
    return stages > out ? stages : out;
}

// C[M, ldc] (columns [0, N)) = A[M, K] . W^T. FUSED (NVFP4 only): W0 gate,
// W1 up, C = silu(gate) * up with N = inter; per half 64 features of each.
// Otherwise 128 features of W0 per half. NVFP4: w = packed [K/2, N],
// s = scales [K/16, N], gs = the FP32 tensor scale. FP8: w = [N, K] e4m3,
// s = the FP32 scale grid [N/128, K/128] (column 0 is read). AF: A is BF16
// [M, K]; or NVFP4 packed [M, K/2] with As the E4M3 scales [M, K/16]; or
// E4M3 [M, K]; Ag the FP32 per-token scales of the last two.
// OUTMX (FUSED W4A8): C is written as MX8 tokens for the down projection,
// E4M3 codes [M, N] at C and the UE8M0 scales [M, N/32] at Cs (DbT MX8).
template <int WF, bool FUSED, int HV, int WST, int AF = DB_A_BF16, bool OUTMX = false>
__device__ __forceinline__ void db_body(
    const void* __restrict__ Av, const unsigned char* __restrict__ As, const float* __restrict__ Ag,
    const unsigned char* __restrict__ W0, const void* __restrict__ S0, float gs0,
    const unsigned char* __restrict__ W1, const void* __restrict__ S1, float gs1,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K, unsigned int ldc,
    unsigned int nchunks, unsigned char* __restrict__ Cs = nullptr)
{
    // PDL entry before any other work (no-op unless launched with PDL).
    atlas_pdl_enter();
    static_assert(AF != DB_A_BF16 || WST == 2 || WST == 3, "the wait_group order below holds for 2 or 3 weight stages");
    constexpr int TST = db_tst<AF, WST>();
    static_assert(AF != DB_A_NVFP4 || WF == DB_NVFP4, "NVFP4 tokens take NVFP4 weights (W4A4)");
    using Tok = typename DbT<AF>::T;
    constexpr int TTM = db_rows<AF>() / 8;               // token tiles per chunk
    constexpr unsigned int FEAT = FUSED ? 64 : 128;      // output features per half
    constexpr unsigned int NT = 128 * HV;                // threads
    extern __shared__ __align__(16) unsigned char db_smem[];
    Tok* tk = (Tok*)db_smem;                                       // TST token stages
    DbW<WF, HV>* wt = (DbW<WF, HV>*)(db_smem + TST * sizeof(Tok));  // WST weight stages
    // MX8: two groups of DL_MX_SG stages' token scales, 16 bytes a token.
    unsigned char* mxb = db_smem + TST * sizeof(Tok) + WST * sizeof(DbW<WF, HV>);

    // Raster: groups of DB_GROUP row chunks, chunk index fastest. The host
    // picks nchunks >= ceil(M / 64) for the wave count.
    const unsigned int cs = (M + nchunks - 1) / nchunks;
    const unsigned int nft = N / (FEAT * HV);
    const unsigned int grp = blockIdx.x / (DB_GROUP * nft);
    const unsigned int first = grp * DB_GROUP;
    const unsigned int gc = min(nchunks - first, (unsigned int)DB_GROUP);
    const unsigned int in = blockIdx.x - grp * DB_GROUP * nft;
    const unsigned int chunk = first + in % gc;
    const unsigned int n0 = (in / gc) * FEAT * HV;
    const unsigned int r0 = chunk * cs;
    if (r0 >= M) return;                                 // whole CTA, before any barrier
    const unsigned int rows = min(cs, M - r0);
    const int tt_valid = (rows + 7) >> 3;

    const unsigned int tid = threadIdx.x;
    const unsigned int half = tid >> 7, ht = tid & 127;
    const unsigned int warp = ht >> 5, lane = tid & 31;
    const unsigned int g = lane >> 2, t = lane & 3;
    const unsigned int nh = n0 + half * FEAT;            // this half's first feature (per matrix)
    const int ksteps = K / DB_KS;

    // Token loader: row tid / TPR, CPT consecutive 16-byte chunks of it.
    constexpr unsigned int TPR = 2 * HV, CPT = 8 / TPR;
    const unsigned int lr = tid / TPR;
    const bool tok_live = (int)lr < tt_valid * 8;
    const bool tok_valid = lr < rows;
    const unsigned int tc = (tid % TPR) * CPT, tc0 = tc * 8;
    const __nv_bfloat16* a_src = (const __nv_bfloat16*)Av + (unsigned long long)(r0 + (tok_valid ? lr : 0)) * K + tc0;
    // Low-precision token loaders (128 rows). NVFP4: one 16-byte chunk per
    // thread (row tid / 2), and threads 0..127 row tid's four scales. E4M3:
    // chunk tid % 4 of rows tid / 4 and tid / 4 + 64.
    const unsigned int qrow = AF == DB_A_NVFP4 ? tid >> 1 : tid >> 2;
    const unsigned int qc = AF == DB_A_NVFP4 ? tid & 1 : tid & 3;
    constexpr unsigned int QSTEP = 64;                   // E4M3: the second row
    const unsigned int QROW_B = AF == DB_A_NVFP4 ? 0 : qrow + QSTEP;
    const unsigned int qbytes = AF == DB_A_NVFP4 ? K / 2 : K;
    auto q_row = [&](unsigned int r) { return (const unsigned char*)Av + (unsigned long long)(r0 + (r < rows ? r : 0)) * qbytes + qc * 16; };
    const unsigned char* q_src = q_row(qrow);
    const unsigned char* q_src_b = q_row(QROW_B);
    const unsigned char* qs_src = As + (unsigned long long)(r0 + (tid < rows ? tid : 0)) * (K / 16);

    // NVFP4 weight loader: two 16-byte chunks of k-pair row `wkp`; scale
    // loader (threads 0..31 of each half): one chunk of group row `sgr`.
    // FUSED chunks 0..3 are gate, 4..7 up.
    const unsigned int wkp = FUSED ? (ht & 63) >> 1 : ht >> 2;
    const unsigned int wc = FUSED ? (ht >> 6) * 4 + (ht & 1) * 2 : (ht & 3) * 2;
    const unsigned int sgr = (ht >> 3) & 3, sc_c = ht & 7;
    const unsigned char* w_src;
    const unsigned char* s_src = nullptr;
    if constexpr (WF == DB_NVFP4) {
        w_src = ((FUSED && wc >= 4) ? W1 : W0) + nh + (wc & (FUSED ? 3 : 7)) * 16
                + (unsigned long long)wkp * N;
        s_src = (const unsigned char*)((FUSED && sc_c >= 4) ? S1 : S0) + nh + (sc_c & (FUSED ? 3 : 7)) * 16
                + (unsigned long long)sgr * N;
    } else if constexpr (WF == DB_FP8T) {
        // FP8T loader: k rows ht/8 + 16j (j = 0..3), chunk ht%8 of each.
        w_src = W0 + (unsigned long long)(ht >> 3) * N + nh + (ht & 7) * 16;
    } else {
        // FP8 loader: rows ht/4 + 32j (j = 0..3), chunk ht%4 of each.
        w_src = W0 + (unsigned long long)(nh + (ht >> 2)) * K + (ht & 3) * 16;
    }

    auto issue_tok = [&](int s) {
        const unsigned int kb = s * DB_KS;
        if constexpr (AF == DB_A_BF16) {
            if (tok_live) {
                #pragma unroll
                for (unsigned int c = 0; c < CPT; c++)
                    bm_cp16(&tk[s % TST].tok[lr][db_tok(lr, tc + c)], a_src + kb + 8 * c, tok_valid);
            }
        } else if constexpr (AF == DB_A_NVFP4) {
            if ((int)qrow < tt_valid * 8)
                bm_cp16(&tk[s % TST].q[qrow][dl_tok4(qrow, qc)], q_src + kb / 2, qrow < rows);
            if ((int)tid < tt_valid * 8 && tid < DL_ROWS)
                dl_cp4(&tk[s % TST].s[tid], qs_src + kb / 16, tid < rows);
        } else {
            if ((int)qrow < tt_valid * 8) bm_cp16(&tk[s % TST].q[qrow][qc * 16], q_src + kb, qrow < rows);
            if ((int)QROW_B < tt_valid * 8)
                bm_cp16(&tk[s % TST].q[QROW_B][qc * 16], q_src_b + kb, QROW_B < rows);
            if (AF == DB_A_MX8 && s % DL_MX_SG == 0 && (int)tid < tt_valid * 8 && tid < DL_ROWS)
                bm_cp16(&mxb[((s / DL_MX_SG) & 1) * (DL_ROWS * 16) + tid * 16],
                        As + (unsigned long long)(r0 + (tid < rows ? tid : 0)) * (K / 32) + 2 * s, tid < rows);
        }
    };
    auto issue_w = [&](int s) {
        const unsigned int kb = s * DB_KS;
        DbW<WF, HV>& sw = wt[s % WST];
        if constexpr (WF == DB_NVFP4) {
            const unsigned char* ws = w_src + (unsigned long long)(kb / 2) * N;
            const unsigned int z = db_swz_w<AF>(wkp);
            bm_cp16(&sw.w[half][wkp][(wc ^ z) * 16], ws, true);
            bm_cp16(&sw.w[half][wkp][((wc + 1) ^ z) * 16], ws + 16, true);
            if (ht < 32)
                bm_cp16(&sw.s[half][sgr][(sc_c ^ bm_swz_s(sgr)) * 16],
                        s_src + (unsigned long long)(kb / 16) * N, true);
        } else if constexpr (WF == DB_FP8T) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int r = (ht >> 3) + 16 * j;
                bm_cp16(&sw.w[half][r][((ht & 7) ^ dl_swz_f8t(r)) * 16],
                        w_src + (unsigned long long)(kb + 16 * j) * N, true);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++)
                bm_cp16(&sw.w[half][db_fp8_row((ht >> 2) + 32 * j)][(ht & 3) * 16],
                        w_src + (unsigned long long)(32 * j) * K + kb, true);
        }
    };

    float acc[2][TTM][4];
    #pragma unroll
    for (int f = 0; f < 2; f++)
        #pragma unroll
        for (int i = 0; i < TTM; i++)
            acc[f][i][0] = acc[f][i][1] = acc[f][i][2] = acc[f][i][3] = 0.0f;

    // BF16 tokens: commit order W0 T0 W1 .. W(WST-2), then per step T(s+1)
    // W(s+WST-1): at step s only the newest WST-2 groups (weights ahead) may
    // be pending. Tokens are L2 hits two stages deep; weights stream from
    // DRAM. Low-precision tokens: one group per stage, WST - 1 ahead.
    if constexpr (AF == DB_A_BF16) {
        issue_w(0);
        bm_commit();
        issue_tok(0);
        bm_commit();
        #pragma unroll
        for (int s = 1; s < WST - 1; s++) {
            if (s < ksteps) issue_w(s);
            bm_commit();
        }
    }
    if constexpr (AF != DB_A_BF16) {
        // Split barriers instead of one CTA barrier per stage: full[b] counts
        // every thread's copies into buffer b (cp.async.mbarrier.arrive), and
        // empty[b] every thread done reading it, so a warp can run up to a
        // stage ahead of the slowest.
        __shared__ __align__(8) unsigned long long full[WST], empty[WST];
        if (tid == 0) {
            #pragma unroll
            for (int b = 0; b < WST; b++) {
                dl_mbar_init(&full[b], NT);
                dl_mbar_init(&empty[b], NT);
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < WST - 1; s++) {
            if (s < ksteps) {
                issue_tok(s);
                issue_w(s);
            }
            dl_cp_arrive(&full[s]);
        }
        // The tile count is fixed per CTA: one loop per count.
        auto kloop = [&](auto ttc) {
            constexpr int TT = decltype(ttc)::value;
            if constexpr (dl_a8<AF>() && WF == DB_NVFP4) {
                // W4A8: stage s + 1 is re-encoded slot by slot between the
                // MMA groups of stage s, so the two warps of a scheduler that
                // run in step still overlap ALU and tensor work.
                DlA8 cur;
                dl_mbar_wait(&full[0], 0);
                dl_a8_begin<FUSED, HV>(cur, &wt[0], half, warp, g, t);
                #pragma unroll
                for (int sl = 0; sl < 4; sl++) dl_a8_slot(cur, sl);
                for (int s = 0; s < ksteps; s++) {
                    const int b = s % WST;
                    const bool more = s + 1 < ksteps;
                    const unsigned char* mxs = AF == DB_A_MX8
                        ? mxb + ((s / DL_MX_SG) & 1) * (DL_ROWS * 16) + 2 * (s % DL_MX_SG) : nullptr;
                    DlA8 nxt;
                    constexpr int Q = (TT + 3) / 4;
                    dl_a8_mma<0, (Q < TT ? Q : TT), TTM>(tk[b], cur, acc, g, t, mxs);
                    if (more) {
                        dl_mbar_wait(&full[(s + 1) % WST], ((s + 1) / WST) & 1);
                        dl_a8_begin<FUSED, HV>(nxt, &wt[(s + 1) % WST], half, warp, g, t);
                        dl_a8_slot(nxt, 0);
                    }
                    dl_a8_mma<(Q < TT ? Q : TT), (2 * Q < TT ? 2 * Q : TT), TTM>(tk[b], cur, acc, g, t, mxs);
                    if (more) dl_a8_slot(nxt, 1);
                    dl_a8_mma<(2 * Q < TT ? 2 * Q : TT), (3 * Q < TT ? 3 * Q : TT), TTM>(tk[b], cur, acc, g, t, mxs);
                    if (more) dl_a8_slot(nxt, 2);
                    dl_a8_mma<(3 * Q < TT ? 3 * Q : TT), TT, TTM>(tk[b], cur, acc, g, t, mxs);
                    if (more) dl_a8_slot(nxt, 3);
                    dl_mbar_arrive(&empty[b]);
                    const int sn = s + WST - 1;
                    if (sn < ksteps) {
                        if (sn >= WST) dl_mbar_wait(&empty[sn % WST], (sn / WST - 1) & 1);
                        issue_tok(sn);
                        issue_w(sn);
                        dl_cp_arrive(&full[sn % WST]);
                    }
                    if (more) cur = nxt;
                }
            } else {
                for (int s = 0; s < ksteps; s++) {
                    const int b = s % WST;
                    dl_mbar_wait(&full[b], (s / WST) & 1);
                    dl_stage<WF, AF, FUSED, TT, HV, TTM>(tk[b], &wt[b], half, acc, warp, g, t);
                    dl_mbar_arrive(&empty[b]);
                    const int sn = s + WST - 1;    // into buffer (s - 1) % WST
                    if (sn < ksteps) {
                        if (sn >= WST) dl_mbar_wait(&empty[sn % WST], (sn / WST - 1) & 1);
                        issue_tok(sn);
                        issue_w(sn);
                        dl_cp_arrive(&full[sn % WST]);
                    }
                }
            }
        };
        switch (tt_valid) {
            case 1: kloop(dl_int<1>{}); break;
            case 2: kloop(dl_int<2>{}); break;
            case 3: kloop(dl_int<3>{}); break;
            case 4: kloop(dl_int<4>{}); break;
            case 5: kloop(dl_int<5>{}); break;
            case 6: kloop(dl_int<6>{}); break;
            case 7: kloop(dl_int<7>{}); break;
            case 8: kloop(dl_int<8>{}); break;
            case 9: kloop(dl_int<9>{}); break;
            case 10: kloop(dl_int<10>{}); break;
            case 11: kloop(dl_int<11>{}); break;
            case 12: kloop(dl_int<12>{}); break;
            case 13: kloop(dl_int<13>{}); break;
            case 14: kloop(dl_int<14>{}); break;
            case 15: kloop(dl_int<15>{}); break;
            default: kloop(dl_int<TTM>{}); break;
        }
    } else
    for (int s = 0; s < ksteps; s++) {
        bm_wait<WST - 2>();
        __syncthreads();
        if (s + 1 < ksteps) issue_tok(s + 1);
        bm_commit();
        if (s + WST - 1 < ksteps) issue_w(s + WST - 1);
        bm_commit();
        const Tok& tb = tk[s % TST];
        const DbW<WF, HV>& sb = wt[s % WST];
        switch (tt_valid) {
            case 1: db_stage<WF, FUSED, 1>(tb, sb, half, acc, warp, g, t); break;
            case 2: db_stage<WF, FUSED, 2>(tb, sb, half, acc, warp, g, t); break;
            case 3: db_stage<WF, FUSED, 3>(tb, sb, half, acc, warp, g, t); break;
            case 4: db_stage<WF, FUSED, 4>(tb, sb, half, acc, warp, g, t); break;
            case 5: db_stage<WF, FUSED, 5>(tb, sb, half, acc, warp, g, t); break;
            case 6: db_stage<WF, FUSED, 6>(tb, sb, half, acc, warp, g, t); break;
            case 7: db_stage<WF, FUSED, 7>(tb, sb, half, acc, warp, g, t); break;
            default: db_stage<WF, FUSED, 8>(tb, sb, half, acc, warp, g, t); break;
        }
    }
    bm_wait<0>();

    // Scales of the FP32 accumulators (BF16 weights carry 2^-8; the
    // low-precision MMAs see the checkpoint codes and block scales).
    constexpr float WK = AF == DB_A_BF16 ? 256.0f : 1.0f;
    float k0, k1;
    if constexpr (WF == DB_NVFP4) {
        k0 = gs0 * WK;
        k1 = FUSED ? gs1 * WK : k0;
    } else {
        // FP8: the [N/128, K/128] grid's column 0; FP8T: the transposed
        // [K/128, N/128] grid's row 0.
        k0 = k1 = ((const float*)S0)[WF == DB_FP8T ? nh >> 7 : (nh >> 7) * (K >> 7)] * WK;
    }

    // Epilogue staged through shared memory as [token][feature], then whole rows.
    __syncthreads();
    constexpr unsigned int OLD = FEAT * HV + 8;
    __nv_bfloat16* os = (__nv_bfloat16*)db_smem;
    #pragma unroll
    for (int tt = 0; tt < TTM; tt++) {
        if (tt < tt_valid) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int tok = tt * 8 + 2 * t + (i & 1), q = i >> 1;
                __nv_bfloat16* o = os + tok * OLD + half * FEAT;
                // Per-token activation scale (1 for BF16 tokens; rows past M
                // read row r0, their outputs are never stored).
                const float ag = AF == DB_A_BF16 || AF == DB_A_MX8 ? 1.0f : Ag[r0 + (tok < rows ? tok : 0)];
                if (FUSED) {
                    // The SiLU reads BF16 gate/up, as the unfused chain does.
                    const float gv = __bfloat162float(__float2bfloat16(acc[0][tt][i] * k0 * ag));
                    const float uv = __bfloat162float(__float2bfloat16(acc[1][tt][i] * k1 * ag));
                    o[16 * warp + 2 * g + q] = __float2bfloat16(gv * (1.0f / (1.0f + __expf(-gv))) * uv);
                } else {
                    o[32 * warp + 4 * g + q] = __float2bfloat16(acc[0][tt][i] * k0 * ag);
                    o[32 * warp + 4 * g + 2 + q] = __float2bfloat16(acc[1][tt][i] * k1 * ag);
                }
            }
        }
    }
    __syncthreads();
    if constexpr (OUTMX) {
        // The tile's 128 outputs are two 64-K stages of the down projection:
        // per row and MMA block (stage st, half h: k = 64 st + 16 c + 8 h + i)
        // E4M3 of x / 2^E, 2^E the smallest power of two with amax / 2^E <=
        // 448 (UE8M0 127 + E; 2^0 for an all-zero block), from the BF16 the
        // unfused chain would write.
        static_assert(FUSED && FEAT * HV == 128, "MX output: the fused 128-wide gate+up tile");
        unsigned char* ct = db_smem + DL_ROWS * OLD * sizeof(__nv_bfloat16);   // [row][128] codes
        for (unsigned int task = tid; task < rows * 4; task += NT) {
            const unsigned int r = task >> 2, st = (task >> 1) & 1, h = task & 1;
            uint4 v[4];
            float am = 0.0f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                v[c] = *(const uint4*)&os[r * OLD + 64 * st + 16 * c + 8 * h];
                const unsigned int* w = (const unsigned int*)&v[c];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 f = __bfloat1622float2(*(const __nv_bfloat162*)&w[j]);
                    am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
            const float sc = dl_pow2_scale(am, 448.0f), inv = 1.0f / sc;
            Cs[(unsigned long long)(r0 + r) * (N / 32) + (n0 / 64 + st) * 2 + h] =
                (unsigned char)((__float_as_uint(sc) >> 23) & 0xFF);   // 127 + E
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const unsigned int* w = (const unsigned int*)&v[c];
                unsigned int o[2] = {0, 0};
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 f = __bfloat1622float2(*(const __nv_bfloat162*)&w[j]);
                    unsigned short pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(f.y * inv), "f"(f.x * inv));
                    o[j >> 1] |= (unsigned int)pair << (16 * (j & 1));
                }
                *(uint2*)&ct[r * 128 + 64 * st + 16 * c + 8 * h] = make_uint2(o[0], o[1]);
            }
        }
        __syncthreads();
        unsigned char* Cq = (unsigned char*)C;
        for (unsigned int i = tid; i < rows * 8; i += NT) {
            const unsigned int r = i >> 3, c = i & 7;
            *(uint4*)&Cq[(unsigned long long)(r0 + r) * N + n0 + c * 16] = *(const uint4*)&ct[r * 128 + c * 16];
        }
    } else {
        for (unsigned int i = tid; i < rows * (FEAT * HV / 8); i += NT) {
            const unsigned int r = i / (FEAT * HV / 8), c = i % (FEAT * HV / 8);
            *(uint4*)&C[(unsigned long long)(r0 + r) * ldc + n0 + c * 8] = *(const uint4*)&os[r * OLD + c * 8];
        }
    }
}

// Two feature halves; three NVFP4 weight stages (43 KiB) and two FP8 ones
// (48 KiB), so both run two CTAs (16 warps) per SM. Measured isolated at the
// 27B shapes: 3 NVFP4 stages within noise of 2; four halves at one CTA/SM
// 5-10 % slower; FP8 at one CTA/SM (three stages) 10-15 % slower.
#define DB_HV 2
#define DB_NV_WST 3
#define DB_F8_WST 2
// Low-precision kernels: one CTA per SM, 128-row chunks, four stages:
// W4A4 4 x 13.5 KiB (under the 66 KiB 128 x 264 BF16 epilogue tile), W8A8
// 4 x 24 KiB, W4A8 4 x 17 KiB. Measured at the 27B shapes: W4A4 five or six
// stages and W8A8 / W4A8 three are slower.
#ifndef DL_NV_WST
#define DL_NV_WST 4
#endif
#ifndef DL_F8_WST
#define DL_F8_WST 4
#endif
#ifndef DL_W4A8_WST
#define DL_W4A8_WST 4
#endif

// Shared memory: 2 * sizeof(DbTok) + WST * sizeof(DbW<WF, HV>) bytes (dynamic).
// Grid: (N / (128 HV)) * nchunks CTAs (FUSED: N = inter, N / (64 HV)), block
// 128 HV; nchunks >= ceil(M / 64) row chunks of ceil(M / nchunks) rows.

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_bf16mma_gate_up_silu(const __nv_bfloat16* __restrict__ A,
                           const unsigned char* __restrict__ gate_packed, const unsigned char* __restrict__ gate_scale,
                           float gate_scale2,
                           const unsigned char* __restrict__ up_packed, const unsigned char* __restrict__ up_scale,
                           float up_scale2,
                           __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                           unsigned int nchunks)
{
    db_body<DB_NVFP4, true, DB_HV, DB_NV_WST>(A, nullptr, nullptr, gate_packed, gate_scale, gate_scale2, up_packed, up_scale,
                                               up_scale2, C, M, N, K, N, nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_bf16mma_nvfp4(const __nv_bfloat16* __restrict__ A,
                    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                    float scale2, __nv_bfloat16* __restrict__ C,
                    unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DB_NV_WST>(A, nullptr, nullptr, packed, scale, scale2, nullptr, nullptr, 0.0f, C, M, N, K, ldc,
                                                 nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_bf16mma_fp8(const __nv_bfloat16* __restrict__ A,
                  const unsigned char* __restrict__ weight, const float* __restrict__ scale_grid,
                  __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                  unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_FP8, false, DB_HV, DB_F8_WST>(A, nullptr, nullptr, weight, scale_grid, 0.0f, nullptr, nullptr, 0.0f, C, M, N, K, ldc,
                                               nchunks);
}

// ---- Low-precision activations (ATLAS_DENSE_PREFILL_LOWP) -----------------
// Same grids as the BF16 kernels. A4: aq = packed e2m1 [M, K/2], as = E4M3
// scales [M, K/16]; A8: aq = E4M3 [M, K]; ag = FP32 per-token scales.

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w4a4_gate_up_silu(const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
                        const float* __restrict__ ag,
                        const unsigned char* __restrict__ gate_packed, const unsigned char* __restrict__ gate_scale,
                        float gate_scale2,
                        const unsigned char* __restrict__ up_packed, const unsigned char* __restrict__ up_scale,
                        float up_scale2,
                        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                        unsigned int nchunks)
{
    db_body<DB_NVFP4, true, DB_HV, DL_NV_WST, DB_A_NVFP4>(aq, as, ag, gate_packed, gate_scale, gate_scale2, up_packed,
                                                          up_scale, up_scale2, C, M, N, K, N, nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w4a4_nvfp4(const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
                 const float* __restrict__ ag,
                 const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                 float scale2, __nv_bfloat16* __restrict__ C,
                 unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DL_NV_WST, DB_A_NVFP4>(aq, as, ag, packed, scale, scale2, nullptr, nullptr, 0.0f,
                                                           C, M, N, K, ldc, nchunks);
}

// W8A8 on the transposed [K, N] FP8 table; scale_t = its [K/128, N/128] grid.
extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w8a8_fp8t(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                const unsigned char* __restrict__ weight_t, const float* __restrict__ scale_t,
                __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_FP8T, false, DB_HV, DL_F8_WST, DB_A_FP8>(aq, nullptr, ag, weight_t, scale_t, 0.0f, nullptr, nullptr,
                                                        0.0f, C, M, N, K, ldc, nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w8a8_fp8(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
               const unsigned char* __restrict__ weight, const float* __restrict__ scale_grid,
               __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
               unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_FP8, false, DB_HV, DL_F8_WST, DB_A_FP8>(aq, nullptr, ag, weight, scale_grid, 0.0f, nullptr, nullptr,
                                                       0.0f, C, M, N, K, ldc, nchunks);
}

// Per-token activation quantization, one 256-thread CTA per row, K % 16 == 0
// and K <= 16 * 256 * DL_QB. The row is read once into registers (thread i
// holds 16-element blocks i, i + 256, ...). The token scale is a power of
// two (2^e >= amax / max-code), so the codes times their scales are exact
// and the GEMM applies it once in FP32.
//   dl_quant_nvfp4: NVFP4 per 16 with the ModelOpt quantizer
//     (nvfp4_block_quant.cuh) under global scale 1 / ag; ag = 2^e >= amax / (6 * 448).
//   dl_quant_fp8: E4M3 of x / ag; ag = 2^e >= amax / 448.
#define DL_QB 5   // blocks per thread: K <= 20480


// Loads this thread's blocks of row x; returns the row's amax (all threads).
__device__ __forceinline__ float dl_load_row(const __nv_bfloat16* __restrict__ x, unsigned int K, uint4 (&v)[DL_QB][2],
                                             float (&bmax)[DL_QB]) {
    __shared__ float red[8];
    float am = 0.0f;
    #pragma unroll
    for (int i = 0; i < DL_QB; i++) {
        const unsigned int b = threadIdx.x + 256 * i;
        bmax[i] = 0.0f;
        if (b < K / 16) {
            v[i][0] = *(const uint4*)&x[16 * b];
            v[i][1] = *(const uint4*)&x[16 * b + 8];
            const unsigned int* w = (const unsigned int*)&v[i][0];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 f = __bfloat1622float2(*(const __nv_bfloat162*)&w[j]);
                bmax[i] = fmaxf(bmax[i], fmaxf(fabsf(f.x), fabsf(f.y)));
            }
            am = fmaxf(am, bmax[i]);
        }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    am = red[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) am = fmaxf(am, red[i]);
    return am;
}

extern "C" __global__ void __launch_bounds__(256)
dl_quant_nvfp4(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
               float* __restrict__ ag, unsigned int K)
{
    atlas_pdl_enter();
    const unsigned long long r = blockIdx.x;
    uint4 v[DL_QB][2];
    float bmax[DL_QB];
    const float sc = dl_pow2_scale(dl_load_row(x + r * K, K, v, bmax), 6.0f * 448.0f);
    const float gl = 1.0f / sc;                          // exact: a power of two
    if (threadIdx.x == 0) ag[r] = sc;
    #pragma unroll
    for (int i = 0; i < DL_QB; i++) {
        const unsigned int b = threadIdx.x + 256 * i;
        if (b >= K / 16) break;
        const unsigned int* w = (const unsigned int*)&v[i][0];
        Nvfp4qGrid grid;
        s[r * (K / 16) + b] = (unsigned char)nvfp4q_block_scale(bmax[i], gl, &grid);
        unsigned int o[2] = {0, 0};
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float2 f = __bfloat1622float2(*(const __nv_bfloat162*)&w[j]);
            o[j >> 2] |= nvfp4q_pack2(f.x, f.y, grid) << (8 * (j & 3));
        }
        *(uint2*)&q[r * (K / 2) + 8 * b] = make_uint2(o[0], o[1]);
    }
}

extern "C" __global__ void __launch_bounds__(256)
dl_quant_fp8(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q, float* __restrict__ ag,
             unsigned int K)
{
    atlas_pdl_enter();
    const unsigned long long r = blockIdx.x;
    uint4 v[DL_QB][2];
    float bmax[DL_QB];
    const float sc = dl_pow2_scale(dl_load_row(x + r * K, K, v, bmax), 448.0f);
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) ag[r] = sc;
    #pragma unroll
    for (int i = 0; i < DL_QB; i++) {
        const unsigned int b = threadIdx.x + 256 * i;
        if (b >= K / 16) break;
        const unsigned int* w = (const unsigned int*)&v[i][0];
        unsigned int o[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float2 f = __bfloat1622float2(*(const __nv_bfloat162*)&w[j]);
            unsigned short pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(f.y * inv), "f"(f.x * inv));
            o[j >> 1] |= (unsigned int)pair << (16 * (j & 1));
        }
        *(uint4*)&q[r * K + 16 * b] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

// W4A8: NVFP4 weights re-encoded per stage as block-scaled E4M3 (dl_stage),
// E4M3 tokens.
extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w4a8_gate_up_silu(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                        const unsigned char* __restrict__ gate_packed, const unsigned char* __restrict__ gate_scale,
                        float gate_scale2,
                        const unsigned char* __restrict__ up_packed, const unsigned char* __restrict__ up_scale,
                        float up_scale2,
                        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                        unsigned int nchunks)
{
    db_body<DB_NVFP4, true, DB_HV, DL_W4A8_WST, DB_A_FP8>(aq, nullptr, ag, gate_packed, gate_scale, gate_scale2,
                                                          up_packed, up_scale, up_scale2, C, M, N, K, N, nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w4a8_nvfp4(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                 const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                 float scale2, __nv_bfloat16* __restrict__ C,
                 unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DL_W4A8_WST, DB_A_FP8>(aq, nullptr, ag, packed, scale, scale2, nullptr, nullptr,
                                                           0.0f, C, M, N, K, ldc, nchunks);
}

// MX8 handoff (ATLAS_DENSE_PREFILL_LOWP `dnmx`): the W4A8 gate+up writes the
// down projection's tokens as MX8 (codes [M, N] at C, scales [M, N/32] at
// cs); the W4A8 down reads them with the scales applied by the MMA.
extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w4a8_gate_up_silu_mx(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                           const unsigned char* __restrict__ gate_packed, const unsigned char* __restrict__ gate_scale,
                           float gate_scale2,
                           const unsigned char* __restrict__ up_packed, const unsigned char* __restrict__ up_scale,
                           float up_scale2,
                           unsigned char* __restrict__ C, unsigned char* __restrict__ cs, unsigned int M,
                           unsigned int N, unsigned int K, unsigned int nchunks)
{
    db_body<DB_NVFP4, true, DB_HV, DL_W4A8_WST, DB_A_FP8, true>(aq, nullptr, ag, gate_packed, gate_scale, gate_scale2,
                                                                up_packed, up_scale, up_scale2, (__nv_bfloat16*)C, M,
                                                                N, K, N, nchunks, cs);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 1)
dense_w4a8mx_nvfp4(const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
                   const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                   float scale2, __nv_bfloat16* __restrict__ C,
                   unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DL_W4A8_WST, DB_A_MX8>(aq, as, nullptr, packed, scale, scale2, nullptr, nullptr,
                                                           0.0f, C, M, N, K, ldc, nchunks);
}

// ---- W4A8, warp-specialized (picked by row count: dense_bf16mma_lowp.rs) ---
// The 8-warp kernels' 256 weight rows and 64-K stages, with the re-encode
// moved off the MMA warps, persistent over the tiles, and token tiles of up
// to 144 rows (DW_ROWS).
//  * Producers (warps 8..15): producer p runs dl_a8_begin / dl_a8_slot for
//    slot warp p of the 8-warp layout (half p / 4, warp p % 4), lane by
//    lane, and stores the E4M3 operands and UE8M0 row scales in lane order.
//    Their lanes 0..15 copy the NVFP4 stages, lanes 16..31 the tokens.
//  * Consumers (warps 0..7): warp c reads slot warp c's operands back in the
//    same lane and issues only MMAs, so each output is the same MMA row of
//    the same operands in the same K order as in the 8-warp kernel; the
//    epilogue computes db_body's values and stores them from registers.
//    Results are bit-identical.
//  * Tiles: CTA b takes raster tiles b, b + grid, ... (the 8-warp grid's
//    order over 144-row chunks); the rings run on across tiles, so the next
//    tile's stages load and re-encode during the epilogue.
//  Rings: tokens DW_RT deep (issued DW_TA stages ahead), operands DW_RA,
//  NVFP4 stages DW_RW (issued DW_RW - 1 ahead). Barriers: tfull (the token
//  lanes' copies), afull[p] (producer p's operands), cdone (every consumer
//  lane done with a stage's tokens and operands).
#define DW_PRODUCERS 8
#define DW_THREADS (32 * (8 + DW_PRODUCERS))
#define DW_RT 4
#define DW_RA 2
#define DW_RW 3
#define DW_TA (DW_RT - DW_RA)
// Registers: 512 threads launch at 128; the producers give theirs to the
// consumers' up to 144 FP32 accumulators (setmaxnreg: 256 x 192 + 256 x 64).
#define DW_CREG 192
#define DW_PREG 64
#define DW_BG 2     // token tiles per MMA group: 4 MMAs between dependent ones

// One stage of re-encoded operands: per slot warp and lane the four MMA A
// fragments (q = 2 h + f: a[h][2f][0], a[h][2f+1][0], a[h][2f][1],
// a[h][2f+1][1]) and per slot warp and row g its four slots' UE8M0 scales.
struct __align__(16) DpA {
    uint4 a[8][4][32];
    unsigned int ue[8][8];
};
// The scales (sa0, sa1) lane (g, t) passes the MMA (dl_a8_begin's).
__device__ __forceinline__ void dp_sa(const DpA& a, unsigned int c, unsigned int g, unsigned int t,
                                      unsigned int& sa0, unsigned int& sa1) {
    const unsigned int u = a.ue[c][g], odd = t & 1;
    sa0 = (u >> (8 * odd)) & 0xFF;
    sa1 = (u >> (8 * (2 + odd))) & 0xFF;
}
__device__ __forceinline__ unsigned int dp_ue(const DlA8& e) {
    return e.ue[0] | (e.ue[1] << 8) | (e.ue[2] << 16) | (e.ue[3] << 24);
}
// Token tiles: up to 144 rows a CTA (288 rows = two tiles, so the N = 5120
// projections' 20 feature tiles x 2 fit one wave of 48 SMs).
#define DW_ROWS 144
struct __align__(16) DwTok {
    unsigned char q[DW_ROWS][DB_KS];
};
// f(dl_int<n>{}) for the tile count n = ttv (TTM: the compiled maximum).
template <int N, int TTM, class F>
__device__ __forceinline__ void dl_by_tt(int ttv, F&& f) {
    if constexpr (N >= TTM) f(dl_int<TTM>{});
    else if (ttv == N) f(dl_int<N>{});
    else dl_by_tt<N + 1, TTM>(ttv, f);
}

// A raster tile (db_body's blockIdx.x mapping over this kernel's chunks).
struct DwTile {
    unsigned int n0, r0, rows;
    int ttv;
};
// Walks one CTA's tiles: L = blockIdx.x, + gridDim.x, ..., skipping chunks
// past M; s is the stage within the tile.
struct DwCursor {
    unsigned int L;
    int s;
    DwTile tl;
};
template <bool FUSED>
__device__ __forceinline__ bool dw_tile(unsigned int L, unsigned int M, unsigned int N, unsigned int nchunks,
                                        DwTile& tl) {
    constexpr unsigned int TW = (FUSED ? 64 : 128) * 2;
    const unsigned int cs = (M + nchunks - 1) / nchunks, nft = N / TW;
    const unsigned int grp = L / (DB_GROUP * nft), first = grp * DB_GROUP;
    const unsigned int gc = min(nchunks - first, (unsigned int)DB_GROUP);
    const unsigned int in = L - grp * DB_GROUP * nft;
    tl.n0 = (in / gc) * TW;
    tl.r0 = (first + in % gc) * cs;
    if (tl.r0 >= M) return false;
    tl.rows = min(cs, M - tl.r0);
    tl.ttv = (tl.rows + 7) >> 3;
    return true;
}
// The first valid tile at or after c.L (c.L >= total: done).
template <bool FUSED>
__device__ __forceinline__ void dw_seek(DwCursor& c, unsigned int M, unsigned int N, unsigned int nchunks) {
    const unsigned int total = (N / ((FUSED ? 64 : 128) * 2)) * nchunks;
    while (c.L < total && !dw_tile<FUSED>(c.L, M, N, nchunks, c.tl)) c.L += gridDim.x;
}
template <bool FUSED>
__device__ __forceinline__ void dw_next(DwCursor& c, int ksteps, unsigned int M, unsigned int N,
                                        unsigned int nchunks) {
    if (++c.s == ksteps) {
        c.s = 0;
        c.L += gridDim.x;
        dw_seek<FUSED>(c, M, N, nchunks);
    }
}

// db_body's epilogue outputs of one consumer warp, stored from registers: FUSED
// bf16 pairs (outputs 16 warp + 2 g + {0, 1}), else four features 32 warp +
// 4 g + {0..3}; rows past `rows` are not stored.
template <bool FUSED, int TTM>
__device__ __forceinline__ void dw_epi_store(__nv_bfloat16* __restrict__ C, unsigned int ldc, unsigned int r0,
                                             unsigned int col, const float (&acc)[2][TTM][4], int ttv,
                                             unsigned int warp, unsigned int g, unsigned int t, float k0, float k1,
                                             const float* ag_row, unsigned int rows) {
    #pragma unroll
    for (int tt = 0; tt < TTM; tt++) {
        if (tt < ttv) {
            #pragma unroll
            for (int par = 0; par < 2; par++) {
                const unsigned int tok = tt * 8 + 2 * t + par;
                if (tok >= rows) continue;
                const float ag = ag_row[tok];
                __nv_bfloat16 o[4];
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    const int i = par + 2 * q;
                    if (FUSED) {
                        const float gv = __bfloat162float(__float2bfloat16(acc[0][tt][i] * k0 * ag));
                        const float uv = __bfloat162float(__float2bfloat16(acc[1][tt][i] * k1 * ag));
                        o[q] = __float2bfloat16(gv * (1.0f / (1.0f + __expf(-gv))) * uv);
                    } else {
                        o[q] = __float2bfloat16(acc[0][tt][i] * k0 * ag);
                        o[2 + q] = __float2bfloat16(acc[1][tt][i] * k1 * ag);
                    }
                }
                __nv_bfloat16* dst = C + (unsigned long long)(r0 + tok) * ldc + col;
                if (FUSED)
                    *(__nv_bfloat162*)&dst[16 * warp + 2 * g] = __halves2bfloat162(o[0], o[1]);
                else
                    *(uint2*)&dst[32 * warp + 4 * g] = make_uint2(
                        *(const unsigned int*)&o[0], *(const unsigned int*)&o[2]);
            }
        }
    }
}

template <bool FUSED>
__device__ __forceinline__ void dw_body(
    const unsigned char* __restrict__ Aq, const float* __restrict__ Ag,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0, float gs0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1, float gs1,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K, unsigned int ldc,
    unsigned int nchunks)
{
    atlas_pdl_enter();
    constexpr int HV = 2, TTM = DW_ROWS / 8;
    constexpr unsigned int FEAT = FUSED ? 64 : 128;      // weight features per half (per matrix)
    extern __shared__ __align__(16) unsigned char db_smem[];
    DwTok* tk = (DwTok*)db_smem;
    DpA* ab = (DpA*)(db_smem + DW_RT * sizeof(DwTok));
    DbW<DB_NVFP4, HV>* wt = (DbW<DB_NVFP4, HV>*)(db_smem + DW_RT * sizeof(DwTok) + DW_RA * sizeof(DpA));

    DwCursor start{blockIdx.x, 0, {}};
    dw_seek<FUSED>(start, M, N, nchunks);
    const unsigned int total = (N / (FEAT * HV)) * nchunks;
    if (start.L >= total) return;                        // whole CTA, before any barrier

    const unsigned int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const unsigned int g = lane >> 2, t = lane & 3;
    const int ksteps = K / DB_KS;

    __shared__ __align__(8) unsigned long long tfull[DW_RT], cdone[DW_RT], afull[DW_RA][DW_PRODUCERS];
    // Per-token scales of the consumers' last two tiles.
    __shared__ float ag_s[2][DW_ROWS];
    if (tid == 0) {
        #pragma unroll
        for (int b = 0; b < DW_RT; b++) {
            dl_mbar_init(&tfull[b], 16 * DW_PRODUCERS);
            dl_mbar_init(&cdone[b], 256);
        }
        #pragma unroll
        for (int b = 0; b < DW_RA; b++)
            #pragma unroll
            for (int p = 0; p < DW_PRODUCERS; p++) dl_mbar_init(&afull[b][p], 32);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    if (wid >= 8) {
        // Producer p. Lanes 0..15 of all producers (WL threads) copy each
        // NVFP4 stage coalesced (thread wq: chunk wq % 8 of k-pair rows
        // wq / 8 + (WL / 8) r of both halves; scales: threads 0..63), each
        // waits on its own copies and a producer barrier publishes them;
        // lanes 16..31 copy the tokens (rows tq / 4 + (WL / 4) i, chunk
        // tq % 4), tracked by tfull, so token arrivals never wait on weights.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(DW_PREG));
        constexpr unsigned int WL = 16 * DW_PRODUCERS, WROWS = WL / 8, TROWS = WL / 4;
        const unsigned int p = wid - 8;
        const bool wl = lane < 16;
        const unsigned int wq = p * 16 + (lane & 15), wc = wq & 7, wr = wq >> 3;
        const unsigned int sh_h = (wq >> 5) & 1, sgr = (wq >> 3) & 3;
        auto src = [&](const unsigned char* a, const unsigned char* b, unsigned int c, unsigned int n0,
                       unsigned long long row) {
            return ((FUSED && c >= 4) ? b : a) + n0 + (c & (FUSED ? 3 : 7)) * 16 + row * N;
        };
        // Weight lanes run DW_RW - 1 stages ahead, token lanes DW_TA: one
        // cursor each (`ahead`), issuing into ring slot Gi.
        const unsigned int qc = wq & 3;
        auto issue = [&](const DwCursor& c, int Gi) {
            const unsigned int kb = c.s * DB_KS;
            if (wl) {
                DbW<DB_NVFP4, HV>& sw = wt[Gi % DW_RW];
                #pragma unroll
                for (int h = 0; h < HV; h++)
                    #pragma unroll
                    for (unsigned int r = 0; r < DB_KS / 2 / WROWS; r++) {
                        const unsigned int row = wr + WROWS * r;
                        bm_cp16(&sw.w[h][row][(wc ^ bm_swz_w(row)) * 16],
                                src(W0, W1, wc, c.tl.n0 + h * FEAT, kb / 2 + row), true);
                    }
                if (wq < 64)
                    bm_cp16(&sw.s[sh_h][sgr][(wc ^ bm_swz_s(sgr)) * 16],
                            src(S0, S1, wc, c.tl.n0 + sh_h * FEAT, kb / 16 + sgr), true);
                bm_commit();
            } else {
                #pragma unroll
                for (unsigned int i = 0; i < (DW_ROWS + TROWS - 1) / TROWS; i++) {
                    const unsigned int r = (wq >> 2) + TROWS * i;
                    if (r < DW_ROWS && (int)r < c.tl.ttv * 8)
                        bm_cp16(&tk[Gi % DW_RT].q[r][qc * 16],
                                Aq + (unsigned long long)(c.tl.r0 + (r < c.tl.rows ? r : 0)) * K + qc * 16 + kb,
                                c.tl.rows > r);
                }
                dl_cp_arrive(&tfull[Gi % DW_RT]);
            }
        };
        // This CTA's stage count, and the lanes' look-ahead cursor.
        int nG = 0;
        for (DwCursor c = start; c.L < total; c.L += gridDim.x, dw_seek<FUSED>(c, M, N, nchunks)) nG += ksteps;
        const int ahead = wl ? DW_RW - 1 : DW_TA;
        DwCursor cur = start;
        int Gi = 0;
        for (; Gi < ahead && Gi < nG; Gi++) {
            issue(cur, Gi);
            dw_next<FUSED>(cur, ksteps, M, N, nchunks);
        }
        if (wl)
            for (int i = Gi; i < DW_RW - 1; i++) bm_commit();
        for (int G = 0; G < nG; G++) {
            if (G >= DW_RA) dl_mbar_wait(&cdone[(G - DW_RA) % DW_RT], ((G - DW_RA) / DW_RT) & 1);
            if (!wl && Gi < nG) {
                issue(cur, Gi++);
                dw_next<FUSED>(cur, ksteps, M, N, nchunks);
            }
            // Stage G's rows have landed everywhere, and every producer is
            // done with stage G - 1's buffer, which takes stage G + RW - 1.
            if (wl) bm_wait<DW_RW - 2>();
            asm volatile("bar.sync 2, %0;" :: "n"(32 * DW_PRODUCERS) : "memory");
            if (wl) {
                if (Gi < nG) {
                    issue(cur, Gi++);
                    dw_next<FUSED>(cur, ksteps, M, N, nchunks);
                } else {
                    bm_commit();
                }
            }
            DpA& out = ab[G % DW_RA];
            static_assert(DW_PRODUCERS == 8, "one slot warp per producer");
            DlA8 a8;
            dl_a8_begin<FUSED, HV>(a8, &wt[G % DW_RW], p >> 2, p & 3, g, t);
            #pragma unroll
            for (int sl = 0; sl < 4; sl++) dl_a8_slot(a8, sl);
            #pragma unroll
            for (int hh = 0; hh < 2; hh++)
                #pragma unroll
                for (int f = 0; f < 2; f++)
                    out.a[p][2 * hh + f][lane] = make_uint4(a8.a[hh][2 * f][0], a8.a[hh][2 * f + 1][0],
                                                            a8.a[hh][2 * f][1], a8.a[hh][2 * f + 1][1]);
            if (t == 0) out.ue[p][g] = dp_ue(a8);
            dl_mbar_arrive(&afull[G % DW_RA][p]);
        }
        bm_wait<0>();
        return;
    }
    // Consumer warp c = 4 half + warp.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(DW_CREG));
    // The lane coordinates re-read after the register handoff, so they are
    // not values carried (and spilled) across it.
    unsigned int ctid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(ctid));
    const unsigned int cw = ctid >> 5, cl = ctid & 31, cg = cl >> 2, cq = cl & 3;
    float acc[2][TTM][4];
    int G = 0;
    auto kloop = [&](auto ttc) {
        constexpr int TT = decltype(ttc)::value;
        for (int s = 0; s < ksteps; s++, G++) {
            dl_mbar_wait(&afull[G % DW_RA][cw % DW_PRODUCERS], (G / DW_RA) & 1);
            dl_mbar_wait(&tfull[G % DW_RT], (G / DW_RT) & 1);
            const DpA& a = ab[G % DW_RA];
            uint4 q[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) q[i] = a.a[cw][i][cl];
            uint2 sa;
            dp_sa(a, cw, cg, cq, sa.x, sa.y);
            const DwTok& tb = tk[G % DW_RT];
            #pragma unroll
            for (int j0 = 0; j0 < TT; j0 += DW_BG) {
                uint4 b[DW_BG];
                #pragma unroll
                for (int j = 0; j < DW_BG; j++)
                    if (j0 + j < TT) b[j] = *(const uint4*)&tb.q[(j0 + j) * 8 + cg][16 * cq];
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int j = 0; j < DW_BG; j++)
                        if (j0 + j < TT)
                            #pragma unroll
                            for (int f = 0; f < 2; f++) {
                                const uint4& x = q[2 * h + f];
                                dl_mma_mx8(acc[f][j0 + j], x.x, x.y, x.z, x.w, h ? b[j].z : b[j].x,
                                           h ? b[j].w : b[j].y, f ? sa.y : sa.x, 0x7f);
                            }
            }
            dl_mbar_arrive(&cdone[G % DW_RT]);
        }
    };
    for (DwCursor c = start; c.L < total;) {
        const DwTile tl = c.tl;
        // ag_s[n % 2]: written here, read after the next barrier; the
        // buffer is reused two tiles on, after every consumer has passed
        // the barrier of the tile between.
        float* ag = ag_s[(G / ksteps) & 1];
        if (ctid < tl.rows) ag[ctid] = Ag[tl.r0 + ctid];
        #pragma unroll
        for (int i = 0; i < TTM; i++)
            #pragma unroll
            for (int f = 0; f < 2; f++) acc[f][i][0] = acc[f][i][1] = acc[f][i][2] = acc[f][i][3] = 0.0f;
        dl_by_tt<1, TTM>(tl.ttv, kloop);
        asm volatile("bar.sync 1, 256;" ::: "memory");
        dw_epi_store<FUSED, TTM>(C, ldc, tl.r0, tl.n0 + (cw >> 2) * FEAT, acc, tl.ttv, cw & 3, cg, cq, gs0,
                                 FUSED ? gs1 : gs0, ag, tl.rows);
        c.L += gridDim.x;
        dw_seek<FUSED>(c, M, N, nchunks);
    }
}

extern "C" __global__ void __launch_bounds__(DW_THREADS, 1)
dense_w4a8ws_gate_up_silu(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                          const unsigned char* __restrict__ gate_packed, const unsigned char* __restrict__ gate_scale,
                          float gate_scale2,
                          const unsigned char* __restrict__ up_packed, const unsigned char* __restrict__ up_scale,
                          float up_scale2,
                          __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                          unsigned int nchunks)
{
    dw_body<true>(aq, ag, gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2, C, M, N, K, N, nchunks);
}

extern "C" __global__ void __launch_bounds__(DW_THREADS, 1)
dense_w4a8ws_nvfp4(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                   const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                   float scale2, __nv_bfloat16* __restrict__ C,
                   unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    dw_body<false>(aq, ag, packed, scale, scale2, nullptr, nullptr, 0.0f, C, M, N, K, ldc, nchunks);
}
