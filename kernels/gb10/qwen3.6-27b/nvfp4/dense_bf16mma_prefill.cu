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

#include "../../common/bf16mma_dequant.cuh"

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

// C[M, ldc] (columns [0, N)) = A[M, K] . W^T. FUSED (NVFP4 only): W0 gate,
// W1 up, C = silu(gate) * up with N = inter; per half 64 features of each.
// Otherwise 128 features of W0 per half. NVFP4: w = packed [K/2, N],
// s = scales [K/16, N], gs = the FP32 tensor scale. FP8: w = [N, K] e4m3,
// s = the FP32 scale grid [N/128, K/128] (column 0 is read).
template <int WF, bool FUSED, int HV, int WST>
__device__ __forceinline__ void db_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0, const void* __restrict__ S0, float gs0,
    const unsigned char* __restrict__ W1, const void* __restrict__ S1, float gs1,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K, unsigned int ldc,
    unsigned int nchunks)
{
    static_assert(WST == 2 || WST == 3, "the wait_group order below holds for 2 or 3 weight stages");
    constexpr unsigned int FEAT = FUSED ? 64 : 128;      // output features per half
    constexpr unsigned int NT = 128 * HV;                // threads
    extern __shared__ __align__(16) unsigned char db_smem[];
    DbTok* tk = (DbTok*)db_smem;                               // 2 token stages
    DbW<WF, HV>* wt = (DbW<WF, HV>*)(db_smem + 2 * sizeof(DbTok));  // WST weight stages

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
    const __nv_bfloat16* a_src = A + (unsigned long long)(r0 + (tok_valid ? lr : 0)) * K + tc0;

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
        if (tok_live) {
            #pragma unroll
            for (unsigned int c = 0; c < CPT; c++)
                bm_cp16(&tk[s & 1].tok[lr][db_tok(lr, tc + c)], a_src + kb + 8 * c, tok_valid);
        }
    };
    auto issue_w = [&](int s) {
        const unsigned int kb = s * DB_KS;
        DbW<WF, HV>& sw = wt[s % WST];
        if constexpr (WF == DB_NVFP4) {
            const unsigned char* ws = w_src + (unsigned long long)(kb / 2) * N;
            const unsigned int z = bm_swz_w(wkp);
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

    // Commit order W0 T0 W1 .. W(WST-2), then per step T(s+1) W(s+WST-1):
    // at step s only the newest WST-2 groups (weights ahead) may be pending.
    // Tokens are L2 hits two stages deep; weights stream from DRAM.
    issue_w(0);
    bm_commit();
    issue_tok(0);
    bm_commit();
    #pragma unroll
    for (int s = 1; s < WST - 1; s++) {
        if (s < ksteps) issue_w(s);
        bm_commit();
    }
    for (int s = 0; s < ksteps; s++) {
        bm_wait<WST - 2>();
        __syncthreads();
        if (s + 1 < ksteps) issue_tok(s + 1);
        bm_commit();
        if (s + WST - 1 < ksteps) issue_w(s + WST - 1);
        bm_commit();
        const DbTok& tb = tk[s & 1];
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

    // Scales of the FP32 accumulators (weights carry 2^-8).
    float k0, k1;
    if constexpr (WF == DB_NVFP4) {
        k0 = gs0 * 256.0f;
        k1 = FUSED ? gs1 * 256.0f : k0;
    } else {
        k0 = k1 = ((const float*)S0)[(nh >> 7) * (K >> 7)] * 256.0f;
    }

    // Epilogue staged through shared memory as [token][feature], then whole rows.
    __syncthreads();
    constexpr unsigned int OLD = FEAT * HV + 8;
    static_assert(2 * sizeof(DbTok) + WST * sizeof(DbW<WF, HV>) >= DB_ROWS * OLD * sizeof(__nv_bfloat16),
                  "epilogue tile fits");
    __nv_bfloat16* os = (__nv_bfloat16*)db_smem;
    #pragma unroll
    for (int tt = 0; tt < 8; tt++) {
        if (tt < tt_valid) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int tok = tt * 8 + 2 * t + (i & 1), q = i >> 1;
                __nv_bfloat16* o = os + tok * OLD + half * FEAT;
                if (FUSED) {
                    // The SiLU reads BF16 gate/up, as the unfused chain does.
                    const float gv = __bfloat162float(__float2bfloat16(acc[0][tt][i] * k0));
                    const float uv = __bfloat162float(__float2bfloat16(acc[1][tt][i] * k1));
                    o[16 * warp + 2 * g + q] = __float2bfloat16(gv * (1.0f / (1.0f + __expf(-gv))) * uv);
                } else {
                    o[32 * warp + 4 * g + q] = __float2bfloat16(acc[0][tt][i] * k0);
                    o[32 * warp + 4 * g + 2 + q] = __float2bfloat16(acc[1][tt][i] * k1);
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
    db_body<DB_NVFP4, true, DB_HV, DB_NV_WST>(A, gate_packed, gate_scale, gate_scale2, up_packed, up_scale,
                                               up_scale2, C, M, N, K, N, nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_bf16mma_nvfp4(const __nv_bfloat16* __restrict__ A,
                    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
                    float scale2, __nv_bfloat16* __restrict__ C,
                    unsigned int M, unsigned int N, unsigned int K, unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_NVFP4, false, DB_HV, DB_NV_WST>(A, packed, scale, scale2, nullptr, nullptr, 0.0f, C, M, N, K, ldc,
                                                 nchunks);
}

extern "C" __global__ void __launch_bounds__(128 * DB_HV, 2)
dense_bf16mma_fp8(const __nv_bfloat16* __restrict__ A,
                  const unsigned char* __restrict__ weight, const float* __restrict__ scale_grid,
                  __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
                  unsigned int ldc, unsigned int nchunks)
{
    db_body<DB_FP8, false, DB_HV, DB_F8_WST>(A, weight, scale_grid, 0.0f, nullptr, nullptr, 0.0f, C, M, N, K, ldc,
                                               nchunks);
}
