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

#include "../../common/bf16mma_dequant.cuh"
#include "../../common/nvfp4_block_quant.cuh"

#define DB_ROWS 64
#define DB_KS 64
#define DB_GROUP 16     // row chunks per raster group

enum { DB_NVFP4 = 0, DB_FP8 = 1 };

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
// FP32 per-token scale), E4M3 (bytes [M, K], FP32 per-token scale).
enum { DB_A_BF16 = 0, DB_A_NVFP4 = 1, DB_A_FP8 = 2 };
template <int AF> struct DbT;
template <> struct DbT<DB_A_BF16> { using T = DbTok; };
// NVFP4: 32-byte rows, the two 16-byte chunks swapped on rows 4..7 of each
// 8 (dl_tok4) so a quarter warp's b0 / b1 reads hit 32 banks.
struct __align__(16) DbTok4 {
    unsigned char q[DB_ROWS][DB_KS / 2];            // 2048 B
    unsigned int s[DB_ROWS];                        //  256 B: the 4 group scales
};
template <> struct DbT<DB_A_NVFP4> { using T = DbTok4; };
struct __align__(16) DbTok8 {
    unsigned char q[DB_ROWS][DB_KS];                // 4096 B
};
template <> struct DbT<DB_A_FP8> { using T = DbTok8; };
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
                                           unsigned int a3, unsigned int b0, unsigned int b1, unsigned int sa) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(0x7f7f7f7fu));
}

// Four e2m1 pairs (bytes of w, low nibble = even k) -> four f16x2 of
// value * 2^-14 (bm_dequant4's bit placement, three bits higher).
__device__ __forceinline__ void dl_e2m1_f16(unsigned int w, unsigned int out[4]) {
    const unsigned int v = w << 4;
    const unsigned int z01 = __byte_perm(w, v, 0x0145);
    const unsigned int z23 = __byte_perm(w, v, 0x2367);
    const unsigned int S = 0x80008000u, M = 0x0E000E00u;
    out[0] = (z01 & S) | ((z01 >> 3) & M);
    out[1] = ((z01 << 8) & S) | ((z01 << 5) & M);
    out[2] = (z23 & S) | ((z23 >> 3) & M);
    out[3] = ((z23 << 8) & S) | ((z23 << 5) & M);
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
template <int WF, int AF, bool FUSED, int TT, int HV>
__device__ __forceinline__ void dl_stage(const typename DbT<AF>::T& tk, const void* sbv, unsigned int half,
                                         float (&acc)[2][8][4], unsigned int warp, unsigned int g,
                                         unsigned int t) {
    if constexpr (AF == DB_A_FP8 && WF == DB_NVFP4) {
        const DbW<DB_NVFP4, HV>& sb = *(const DbW<DB_NVFP4, HV>*)sbv;
        // Group t's scale for the four slots, and the stage maximum per slot
        // (positive E4M3 bytes order as their values).
        const unsigned int s4 = bm_slots<FUSED>(sb.s[half][t], bm_swz_s(t), warp, g);
        unsigned int m4 = __vmaxu4(s4, __shfl_xor_sync(0xffffffffu, s4, 1));
        m4 = __vmaxu4(m4, __shfl_xor_sync(0xffffffffu, m4, 2));
        unsigned int sh[4], ue[4];
        #pragma unroll
        for (int sl = 0; sl < 4; sl++) {
            const float mx = nvfp4q_e4m3_value((m4 >> (8 * sl)) & 0xFF);
            int e = 0;
            if (mx > 0.0f) {
                const float f = frexpf(mx * (6.0f / 448.0f), &e);
                e = f == 0.5f ? e - 1 : e;
            }
            ue[sl] = (unsigned int)(127 + e);
            // s * 2^(9 - E) in f16: >= 1 for every scale whose values still
            // reach E4M3 under 2^E, below 2^16 for all.
            const __half hs = __float2half_rn(ldexpf(nvfp4q_e4m3_value((s4 >> (8 * sl)) & 0xFF), 9 - e));
            const unsigned int hb = __half_as_ushort(hs);
            sh[sl] = hb | (hb << 16);
        }
        const unsigned int odd = t & 1;
        const unsigned int sa0 = odd ? ue[1] : ue[0], sa1 = odd ? ue[3] : ue[2];
        unsigned int a[2][4][2];   // [h][slot]: E4M3 of k 16t+8h+{0..3}, {4..7}
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int w[4], f[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int kp = 8 * t + 4 * h + i;
                w[i] = bm_slots<FUSED>(sb.w[half][kp], bm_swz_w(kp), warp, g);
            }
            dl_transpose4(w, f);   // f[slot]: k 16t+8h .. 16t+8h+7
            #pragma unroll
            for (int sl = 0; sl < 4; sl++) {
                unsigned int x[4], p[4];
                dl_e2m1_f16(f[sl], x);
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    p[j] = dl_e4m3x2(dl_hmul2(dl_hmul2(x[j], sh[sl]), 0x50005000u));   // x 32
                a[h][sl][0] = p[0] | (p[1] << 16);
                a[h][sl][1] = p[2] | (p[3] << 16);
            }
        }
        #pragma unroll
        for (int j = 0; j < TT; j++) {
            const uint4 b = *(const uint4*)&tk.q[j * 8 + g][16 * t];
            #pragma unroll
            for (int f = 0; f < 2; f++) {
                const unsigned int sa = f ? sa1 : sa0;
                dl_mma_mx8(acc[f][j], a[0][2 * f][0], a[0][2 * f + 1][0], a[0][2 * f][1], a[0][2 * f + 1][1], b.x,
                           b.y, sa);
                dl_mma_mx8(acc[f][j], a[1][2 * f][0], a[1][2 * f + 1][0], a[1][2 * f][1], a[1][2 * f + 1][1], b.z,
                           b.w, sa);
            }
        }
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
        const DbW<DB_FP8, HV>& sb = *(const DbW<DB_FP8, HV>*)sbv;
        uint4 q[4];
        #pragma unroll
        for (int s = 0; s < 4; s++)
            q[s] = *(const uint4*)&sb.w[half][db_fp8_row(32 * warp + 4 * g + s)][16 * t];
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
    constexpr unsigned int stages = db_tst<AF, WST>() * sizeof(typename DbT<AF>::T) + WST * sizeof(DbW<WF, HV>);
    constexpr unsigned int out = DB_ROWS * ((FUSED ? 64 : 128) * HV + 8) * sizeof(__nv_bfloat16);
    return stages > out ? stages : out;
}

// C[M, ldc] (columns [0, N)) = A[M, K] . W^T. FUSED (NVFP4 only): W0 gate,
// W1 up, C = silu(gate) * up with N = inter; per half 64 features of each.
// Otherwise 128 features of W0 per half. NVFP4: w = packed [K/2, N],
// s = scales [K/16, N], gs = the FP32 tensor scale. FP8: w = [N, K] e4m3,
// s = the FP32 scale grid [N/128, K/128] (column 0 is read). AF: A is BF16
// [M, K]; or NVFP4 packed [M, K/2] with As the E4M3 scales [M, K/16]; or
// E4M3 [M, K]; Ag the FP32 per-token scales of the last two.
template <int WF, bool FUSED, int HV, int WST, int AF = DB_A_BF16>
__device__ __forceinline__ void db_body(
    const void* __restrict__ Av, const unsigned char* __restrict__ As, const float* __restrict__ Ag,
    const unsigned char* __restrict__ W0, const void* __restrict__ S0, float gs0,
    const unsigned char* __restrict__ W1, const void* __restrict__ S1, float gs1,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K, unsigned int ldc,
    unsigned int nchunks)
{
    static_assert(AF != DB_A_BF16 || WST == 2 || WST == 3, "the wait_group order below holds for 2 or 3 weight stages");
    constexpr int TST = db_tst<AF, WST>();
    static_assert(AF != DB_A_NVFP4 || WF == DB_NVFP4, "NVFP4 tokens take NVFP4 weights (W4A4)");
    using Tok = typename DbT<AF>::T;
    constexpr unsigned int FEAT = FUSED ? 64 : 128;      // output features per half
    constexpr unsigned int NT = 128 * HV;                // threads
    extern __shared__ __align__(16) unsigned char db_smem[];
    Tok* tk = (Tok*)db_smem;                                       // TST token stages
    DbW<WF, HV>* wt = (DbW<WF, HV>*)(db_smem + TST * sizeof(Tok));  // WST weight stages

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
    // Low-precision token loaders. NVFP4: threads 0..127 one 16-byte chunk
    // (row tid / 2), threads 128..191 a row's four scales. E4M3: one chunk
    // (row tid / 4) per thread.
    const unsigned int qr = AF == DB_A_NVFP4 ? (tid & 127) >> 1 : tid >> 2;
    const unsigned int qc = AF == DB_A_NVFP4 ? tid & 1 : tid & 3;
    const bool q_live = AF == DB_A_NVFP4 ? (tid < 128 ? (int)qr < tt_valid * 8 : (int)(tid - 128) < tt_valid * 8)
                                         : (int)qr < tt_valid * 8;
    const unsigned int qrow = AF == DB_A_NVFP4 && tid >= 128 ? tid - 128 : qr;
    const bool q_valid = qrow < rows;
    const unsigned char* q_src = (const unsigned char*)Av
        + (unsigned long long)(r0 + (q_valid ? qrow : 0)) * (AF == DB_A_NVFP4 ? K / 2 : K) + qc * 16;
    const unsigned char* qs_src = As + (unsigned long long)(r0 + (q_valid ? qrow : 0)) * (K / 16);

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
            if (q_live && tid < 128)
                bm_cp16(&tk[s % TST].q[qrow][dl_tok4(qrow, qc)], q_src + kb / 2, q_valid);
            else if (q_live && tid < 192)
                dl_cp4(&tk[s % TST].s[qrow], qs_src + kb / 16, q_valid);
        } else {
            if (q_live) bm_cp16(&tk[s % TST].q[qrow][qc * 16], q_src + kb, q_valid);
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
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++)
                bm_cp16(&sw.w[half][db_fp8_row((ht >> 2) + 32 * j)][(ht & 3) * 16],
                        w_src + (unsigned long long)(32 * j) * K + kb, true);
        }
    };

    float acc[2][8][4];
    #pragma unroll
    for (int f = 0; f < 2; f++)
        #pragma unroll
        for (int i = 0; i < 8; i++)
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
    } else {
        #pragma unroll
        for (int s = 0; s < WST - 1; s++) {
            if (s < ksteps) {
                issue_tok(s);
                issue_w(s);
            }
            bm_commit();
        }
    }
    for (int s = 0; s < ksteps; s++) {
        bm_wait<WST - 2>();
        __syncthreads();
        if constexpr (AF == DB_A_BF16) {
            if (s + 1 < ksteps) issue_tok(s + 1);
            bm_commit();
            if (s + WST - 1 < ksteps) issue_w(s + WST - 1);
            bm_commit();
        } else {
            if (s + WST - 1 < ksteps) {
                issue_tok(s + WST - 1);
                issue_w(s + WST - 1);
            }
            bm_commit();
        }
        const Tok& tb = tk[s % TST];
        const DbW<WF, HV>& sb = wt[s % WST];
        if constexpr (AF != DB_A_BF16) {
            switch (tt_valid) {
                case 1: dl_stage<WF, AF, FUSED, 1, HV>(tb, &sb, half, acc, warp, g, t); break;
                case 2: dl_stage<WF, AF, FUSED, 2, HV>(tb, &sb, half, acc, warp, g, t); break;
                case 3: dl_stage<WF, AF, FUSED, 3, HV>(tb, &sb, half, acc, warp, g, t); break;
                case 4: dl_stage<WF, AF, FUSED, 4, HV>(tb, &sb, half, acc, warp, g, t); break;
                case 5: dl_stage<WF, AF, FUSED, 5, HV>(tb, &sb, half, acc, warp, g, t); break;
                case 6: dl_stage<WF, AF, FUSED, 6, HV>(tb, &sb, half, acc, warp, g, t); break;
                case 7: dl_stage<WF, AF, FUSED, 7, HV>(tb, &sb, half, acc, warp, g, t); break;
                default: dl_stage<WF, AF, FUSED, 8, HV>(tb, &sb, half, acc, warp, g, t); break;
            }
        } else switch (tt_valid) {
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
        k0 = k1 = ((const float*)S0)[(nh >> 7) * (K >> 7)] * WK;
    }

    // Epilogue staged through shared memory as [token][feature], then whole rows.
    __syncthreads();
    constexpr unsigned int OLD = FEAT * HV + 8;
    __nv_bfloat16* os = (__nv_bfloat16*)db_smem;
    #pragma unroll
    for (int tt = 0; tt < 8; tt++) {
        if (tt < tt_valid) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int tok = tt * 8 + 2 * t + (i & 1), q = i >> 1;
                __nv_bfloat16* o = os + tok * OLD + half * FEAT;
                // Per-token activation scale (1 for BF16 tokens; rows past M
                // read row r0, their outputs are never stored).
                const float ag = AF == DB_A_BF16 ? 1.0f : Ag[r0 + (tok < rows ? tok : 0)];
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
    for (unsigned int i = tid; i < rows * (FEAT * HV / 8); i += NT) {
        const unsigned int r = i / (FEAT * HV / 8), c = i % (FEAT * HV / 8);
        *(uint4*)&C[(unsigned long long)(r0 + r) * ldc + n0 + c * 8] = *(const uint4*)&os[r * OLD + c * 8];
    }
}

// Two feature halves; three NVFP4 weight stages (43 KiB) and two FP8 ones
// (48 KiB), so both run two CTAs (16 warps) per SM. Measured isolated at the
// 27B shapes: 3 NVFP4 stages within noise of 2; four halves at one CTA/SM
// 5-10 % slower; FP8 at one CTA/SM (three stages) 10-15 % slower.
#define DB_HV 2
#define DB_NV_WST 3
#define DB_F8_WST 2
// Low-precision tokens are small: four W4A4 stages (46 KiB) still fit two
// CTAs per SM, W8A8 keeps two (40 KiB).
#define DL_NV_WST 4
#define DL_F8_WST 2
#define DL_W4A8_WST 3   // 4 KiB E4M3 tokens + 9 KiB NVFP4 weights per stage: 39 KiB

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

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
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

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_w4a4_nvfp4(const unsigned char* __restrict__ aq, const unsigned char* __restrict__ as,
                 const float* __restrict__ ag,
                 const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                 float scale2, __nv_bfloat16* __restrict__ C,
                 unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DL_NV_WST, DB_A_NVFP4>(aq, as, ag, packed, scale, scale2, nullptr, nullptr, 0.0f,
                                                           C, M, N, K, ldc, nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
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

// The smallest power of two p with amax / p <= lim; 1 for amax == 0.
__device__ __forceinline__ float dl_pow2_scale(float amax, float lim) {
    if (!(amax > 0.0f)) return 1.0f;
    int e;
    const float f = frexpf(__fdiv_rn(amax, lim), &e);   // f in [0.5, 1)
    return ldexpf(1.0f, f == 0.5f ? e - 1 : e);
}

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
extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
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

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_w4a8_nvfp4(const unsigned char* __restrict__ aq, const float* __restrict__ ag,
                 const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                 float scale2, __nv_bfloat16* __restrict__ C,
                 unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DL_W4A8_WST, DB_A_FP8>(aq, nullptr, ag, packed, scale, scale2, nullptr, nullptr,
                                                           0.0f, C, M, N, K, ldc, nchunks);
}
