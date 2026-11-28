// SPDX-License-Identifier: AGPL-3.0-only
//
// One 128-row x 128-column W4A4 tile on mma.sync kind::mxf4nvf4 m16n8k64
// (E2M1 x E2M1 under both operands' E4M3 block scales per 16, FP32
// accumulators over the k64 MMAs in increasing k), with its cp.async ring.
// Operands in the checkpoints' own planes: packed [*, K/2] (low nibble = even
// k) and scales [*, K/16]. `w4_tile<true>` is the same tile over MXFP8
// operands (W8A8: E4M3 [*, K] and UE8M0 scales per 32, [*, K/32]) on
// kind::mxf8f6f4 m16n8k32: a stage's 64 bytes of a row are 64 values instead of
// 128, laid out and loaded exactly as the NVFP4 bytes are. Shared by the
// expert GEMMs of qwen3.8-flash-next (moe_prefill_w4a4.cu) and inkling-2-mini
// (inkling2_moe_gemm.cu).

#pragma once
#include <cuda_bf16.h>
#define W4_BM 128                       // rows per CTA (2 warp rows x 4 m16 tiles)
#ifndef W4_BK
#define W4_BK 128                       // K per pipeline stage (two k64 MMAs)
#endif
#ifndef W4_STAGES
#define W4_STAGES 2                     // cp.async ring depth
#endif
#define W4_ROW (W4_BK / 2)              // packed bytes of one row per stage
#define W4_CHUNKS (W4_ROW / 16)         // 16-byte chunks of one row per stage
#define W4_SROW (W4_BK / 16)            // scale bytes of one row per stage
static_assert(W4_BK == 64 || W4_BK == 128, "W4_BK: 64 or 128");

struct W4Smem {
    unsigned char a[W4_STAGES][W4_BM][W4_ROW];   // activation rows (chunks swizzled)
    unsigned char b[W4_STAGES][128][W4_ROW];     // weight columns (chunks swizzled)
    unsigned char as[W4_STAGES][W4_BM][W4_SROW];
    unsigned char bs[W4_STAGES][128][W4_SROW];
    int tok[W4_BM];
};

// Physical 16-byte chunk of logical chunk `j` in row `r`: the eight rows a
// fragment load spans land on eight distinct bank groups.
__device__ __forceinline__ unsigned int w4_swz(unsigned int r, unsigned int j) {
    return W4_CHUNKS == 4 ? (j ^ ((r >> 1) & 3u)) : (j ^ ((r >> 2) & 1u));
}

__device__ __forceinline__ void w4_cp16(void* dst, const void* src, bool pred) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(pred ? 16u : 0u));
}
template <unsigned int N = W4_SROW>
__device__ __forceinline__ void w4_cps(void* dst, const void* src, bool pred) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(d), "l"(src), "n"(N), "r"(pred ? N : 0u));
}

// Four 8x8 b16 matrices from shared memory, one 16-byte row address a lane (lanes 8m ..
// 8m + 7: matrix m's rows): each register is a lane's 4 bytes of its matrix, exactly the
// operand layout of the FP4 (and E4M3) MMAs below, so one instruction replaces four 4-byte
// loads.
__device__ __forceinline__ void w4_ldsm_x4(const void* row, unsigned int& r0, unsigned int& r1, unsigned int& r2,
                                           unsigned int& r3) {
    const unsigned int a = (unsigned int)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void w4_mma(float (&c)[4], unsigned int a0, unsigned int a1, unsigned int a2,
                                       unsigned int a3, unsigned int b0, unsigned int b1, unsigned int sfa,
                                       unsigned int sfb) {
    const unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
          "h"(z));
}

// E4M3 x E4M3 under one UE8M0 scale per row per 32 (A: lanes 4i row i, 4i + 1 row i + 8).
__device__ __forceinline__ void w8_mma(float (&c)[4], unsigned int a0, unsigned int a1, unsigned int a2,
                                       unsigned int a3, unsigned int b0, unsigned int b1, unsigned int sfa,
                                       unsigned int sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

// n-tile (of 16, 8 columns each) j (0..3) of warp column wn (0..3): two of
// the first 64 columns and the same two of the second, so a gate value and
// its up value share a thread and a warp's gate columns are one 16-group.
__device__ __forceinline__ unsigned int w4_nt(unsigned int wn, unsigned int j) {
    return (j < 2u ? 0u : 8u) + 2u * wn + (j & 1u);
}

// One 128-row x 128-column tile over K: rows tok[r] of (A, As) -- [*, K/2]
// packed and [*, K/16] scales -- against columns n0 .. n0+63 of (B0, S0)
// and n1 .. n1+63 of (B1, S1), each [N, K/2] / [N, K/16]. Warp (wm, wn) =
// (warp & 1, warp >> 1) owns rows wm*64 .. +63 (four m16 tiles) and n-tiles
// w4_nt(wn, 0..3); acc[mt * 4 + j]. MX8: rows [*, K] and scales [*, K/32]
// (rows 4-byte aligned), K a multiple of 128; a stage's two scales are loaded
// with the next stage's as one 4-byte copy every other stage.
template <bool MX8 = false>
__device__ __forceinline__ void w4_tile(
    W4Smem& sm, float (&acc)[16][4],
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ As,
    unsigned int K, unsigned int rows_left,
    const unsigned char* B0, const unsigned char* S0, unsigned int n0,
    const unsigned char* B1, const unsigned char* S1, unsigned int n1
) {
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned int q = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    static_assert(!MX8 || W4_BK == 128, "MX8: 64-byte stages");
    const unsigned int steps = K / (MX8 ? W4_ROW : W4_BK);   // a stage: W4_ROW bytes of a row
    const unsigned int kh = MX8 ? K : K / 2, ks = MX8 ? K / 32 : K / 16;   // bytes a row

    auto issue = [&](unsigned int s, unsigned int buf) {
        if (s < steps) {
            const unsigned int kb = s * W4_ROW;   // byte offset of the stage in a row
            #pragma unroll
            for (unsigned int i = tid; i < W4_BM * W4_CHUNKS; i += 256) {
                const unsigned int r = i / W4_CHUNKS, j = i % W4_CHUNKS;
                const bool valid = r < rows_left;
                const unsigned char* src = A + (size_t)(valid ? sm.tok[r] : 0) * kh + kb + 16 * j;
                w4_cp16(&sm.a[buf][r][16 * w4_swz(r, j)], src, valid);
            }
            #pragma unroll
            for (unsigned int i = tid; i < 128 * W4_CHUNKS; i += 256) {
                const unsigned int col = i / W4_CHUNKS, j = i % W4_CHUNKS;
                const unsigned char* src = (col < 64 ? B0 + (size_t)(n0 + col) * kh : B1 + (size_t)(n1 + col - 64) * kh);
                w4_cp16(&sm.b[buf][col][16 * w4_swz(col, j)], src + kb + 16 * j, true);
            }
            // MX8: stages s and s + 1's scales (2 + 2 bytes) into pair buffer s / 2
            const unsigned int sb = MX8 ? (s / 2) % W4_STAGES : buf, so = MX8 ? (s / 2) * 4 : s * W4_SROW;
            constexpr unsigned int SN = MX8 ? 4 : W4_SROW;
            if (!MX8 || s % 2 == 0) {
                if (tid < W4_BM) {
                    const bool valid = tid < rows_left;
                    w4_cps<SN>(&sm.as[sb][tid][0], As + (size_t)(valid ? sm.tok[tid] : 0) * ks + so, valid);
                } else {
                    const unsigned int col = tid - W4_BM;
                    const unsigned char* src = col < 64 ? S0 + (size_t)(n0 + col) * ks : S1 + (size_t)(n1 + col - 64) * ks;
                    w4_cps<SN>(&sm.bs[sb][col][0], src + so, true);
                }
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    #pragma unroll
    for (unsigned int i = 0; i + 1 < W4_STAGES; ++i) issue(i, i);
    for (unsigned int s = 0; s < steps; ++s) {
        const unsigned int buf = s % W4_STAGES;
        issue(s + W4_STAGES - 1, (s + W4_STAGES - 1) % W4_STAGES);
        asm volatile("cp.async.wait_group %0;" ::"n"(W4_STAGES - 1));
        __syncthreads();
        const unsigned int sb = MX8 ? (s / 2) % W4_STAGES : buf;
        #pragma unroll
        for (unsigned int kk = 0; kk < W4_BK / 64; ++kk) {
            const unsigned int j0 = 2 * kk, j1 = j0 + 1;
            // the scale bytes of this MMA's K: NVFP4 four (k64), MX8 one (k32)
            const unsigned int sk = MX8 ? 2 * (s % 2) + kk : 4 * kk;
            // B: n-tiles 2h and 2h + 1, both K halves of each, in one ldmatrix (lanes 8m ..:
            // n-tile 2h + m / 2, half m % 2); b[j][0 / 1] = rows q of chunks j0 / j1
            unsigned int b[4][2], sfb[4];
            #pragma unroll
            for (unsigned int h = 0; h < 2; ++h) {
                const unsigned int m = lane >> 3, nl = w4_nt(wn, 2 * h + (m >> 1)) * 8 + (lane & 7u);
                w4_ldsm_x4(&sm.b[buf][nl][16 * w4_swz(nl, (m & 1u) ? j1 : j0)], b[2 * h][0], b[2 * h][1],
                           b[2 * h + 1][0], b[2 * h + 1][1]);
            }
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                const unsigned int nc = w4_nt(wn, j) * 8 + q;
                sfb[j] = MX8 ? sm.bs[sb][nc][sk] : *(const unsigned int*)&sm.bs[sb][nc][sk];
            }
            #pragma unroll
            for (unsigned int mt = 0; mt < 4; ++mt) {
                const unsigned int ra = wm * 64 + mt * 16 + q;
                // A: rows 0-7 / 8-15 of the m16 tile, chunks j0 then j1 (lanes 8m ..: rows
                // 8 (m % 2) .., chunk m / 2) -> a0 (q, j0), a1 (q + 8, j0), a2 (q, j1), a3 (q + 8, j1)
                const unsigned int rl = wm * 64 + mt * 16 + (lane & 7u) + 8u * ((lane >> 3) & 1u);
                unsigned int a0, a1, a2, a3;
                w4_ldsm_x4(&sm.a[buf][rl][16 * w4_swz(rl, (lane >> 4) ? j1 : j0)], a0, a1, a2, a3);
                // Scale lanes: rows 0-7 on lanes 4i, rows 8-15 on lanes 4i + 1.
                const unsigned char* sa = &sm.as[sb][ra + 8 * (c & 1u)][sk];
                const unsigned int sfa = MX8 ? *sa : *(const unsigned int*)sa;
                #pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    if (MX8) w8_mma(acc[mt * 4 + j], a0, a1, a2, a3, b[j][0], b[j][1], sfa, sfb[j]);
                    else w4_mma(acc[mt * 4 + j], a0, a1, a2, a3, b[j][0], b[j][1], sfa, sfb[j]);
                }
            }
        }
        __syncthreads();   // every warp done with `buf` before it is refilled
    }
    asm volatile("cp.async.wait_group 0;");
}

// Gather rows [cta_m, cta_m + 128) of an expert: token ids through
// `sorted_token_ids`, or the sorted rows themselves when it is null.
__device__ __forceinline__ void w4_rows(W4Smem& sm, const int* __restrict__ sorted_token_ids,
                                        unsigned int row0, unsigned int rows_left) {
    __syncthreads();   // the previous tile's readers of sm.tok are done
    if (threadIdx.x < W4_BM) {
        const unsigned int r = threadIdx.x;
        sm.tok[r] = r < rows_left ? (sorted_token_ids ? sorted_token_ids[row0 + r] : (int)(row0 + r)) : 0;
    }
    __syncthreads();
}
