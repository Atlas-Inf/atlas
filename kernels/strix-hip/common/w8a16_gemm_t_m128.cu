// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas W8A16 Transposed M128 GEMM — FP8 E4M3 block-scaled, 128x128 tile.
// HIP/gfx1151 (AMD WMMA) port of kernels/gb10/common/w8a16_gemm_t_m128.cu.
//
// C[M,N] = A[M,K] (BF16) * dequant(B_t[K,N] (FP8 E4M3, transposed))
//
// Weights stored as B_t[K, N] so the N-dimension is contiguous (coalesced reads).
// Block scales: block_scale_t[K/128, N/128] FP32. Dequant: LUT[byte] * scale.
//
// WMMA: __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 (16x16x16, n16).
//   Store: lane l, acc element e: row = row_base + 2*e + (l>>4), col = col_base + (l&15)
//
// Register-prefetch double-buffered pipeline — the SAME skeleton as the proven
// w4a16_gemm_t_m128 (NVFP4), widened to 8 warps (256 threads): warps 0..7 each
// own one 16-row M-band (no chunk loop), so a CTA covers 128 M-rows x 128 N-cols.
// Only the B path differs from the NVFP4 template:
//   NVFP4: B_packed[K/2,N] nibbles + per-16-group FP8 scale + scalar scale2.
//   FP8:   B_t[K,N] E4M3 bytes (N contiguous) + block_scale_t[K/128,N/128] FP32.
// Because K_STEP(32) divides 128 and the tile sits inside a single 128x128 scale
// block, ONE scale covers the whole tile -> dequant is E4M3_LUT[byte] * scale.
// The B tile is transposed on dequant into smem_B[n][k] (MMA-ready, K contiguous).
//
// Grid: (ceil(N/128), ceil(M/128), 1), Block: (256,1,1). Tail-M/N/K predicated.

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Bit-math OCP-e4m3fn decode — register ALU, no LDS. Produces the SAME value as
// the E4M3_LUT table the base w8a16 kernels use, including the S.1111.111 NaN.
// Keeps the dequant off the shared-memory ports so it doesn't contend with the
// WMMA smem operand reads (the LUT's 4096 LDS reads per K-step were the cost).
__device__ __forceinline__ float w8m128_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                v = (float)m * 0.001953125f;              // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = __uint_as_float(0x7fc00000u);       // e4m3fn NaN
    else                        v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

#define W8M128_M_TILE 128   // M rows per CTA (8 warps x 16 rows)
#define W8M128_N_TILE 128   // N cols per CTA
#define W8M128_KSTEP  32
#define W8M128_APAD   8     // smem_A row stride 40 -> 80 B, 16-B aligned
#define W8M128_BPAD   8     // smem_B[n][k] row stride pad
#define W8M128_FP8B   128   // FP8 block-scale granularity (128x128)

// Invariants the single-scale-per-tile dequant relies on: a K_STEP tile never
// straddles a 128-K scale block, and the N tile is exactly one 128-N block.
static_assert(W8M128_FP8B % W8M128_KSTEP == 0, "K_STEP must divide the 128-K scale block");
static_assert(W8M128_N_TILE == W8M128_FP8B, "N tile must equal one 128-N scale block");

extern "C" __global__
__launch_bounds__(256, 1)
void w8a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,                // [M, K] BF16
    const unsigned char* __restrict__ B_t,              // [K, N] FP8 E4M3 transposed
    const float* __restrict__ block_scale_t,            // [K/128, N/128] FP32
    __nv_bfloat16* __restrict__ C,                      // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * W8M128_N_TILE;
    const unsigned int cta_m = blockIdx.y * W8M128_M_TILE;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int warp_id = threadIdx.x >> 5;       // 0..7
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int warp_m_offset = warp_id * 16;     // 0..112 (one 16-row band/warp)

    __shared__ __align__(16) __nv_bfloat16 smem_A[2][W8M128_M_TILE][W8M128_KSTEP + W8M128_APAD];
    __shared__ __align__(16) __nv_bfloat16 smem_B[2][W8M128_N_TILE][W8M128_KSTEP + W8M128_BPAD];

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    // Per-thread register-prefetch state for one K_STEP=32 tile:
    //   A: 128 rows x 32 cols bf16 -> 512 uint4; 256 threads -> 2 uint4 (one
    //      8-col slice of a row, two rounds cover 64+64 rows).
    //   B: 32 K-rows x 128 N-cols FP8 = 4096 B -> 256 uint4; 256 threads -> 1
    //      uint4 (16 contiguous N-bytes of one K-row). N-contiguous global reads.
    const unsigned int a_row_base = threadIdx.x >> 2;          // 0..63
    const unsigned int a_col      = (threadIdx.x & 3) << 3;    // 0,8,16,24
    const unsigned int b_row      = threadIdx.x >> 3;          // 0..31 (K row)
    const unsigned int b_col      = (threadIdx.x & 7) << 4;    // 0,16,...,112 (N col)
    const unsigned int n_scale_blocks = (N + W8M128_FP8B - 1) / W8M128_FP8B;
    const unsigned int n_block      = cta_n >> 7;              // == blockIdx.x

    #define W8M128_LOAD_REGS(kb, ra, rb, rs) do { \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
            unsigned int gr  = cta_m + row; \
            unsigned int gc  = (kb) + a_col; \
            /* 16-B load needs a 16-B-aligned src: A[gr*K+gc] aligned iff K%8==0 */ \
            /* (gc is already %8). Uniform per-CTA; misaligned K -> scalar path. */ \
            if ((gr < M) && (gc + 7 < K) && ((K & 7) == 0)) { \
                (ra)[rnd] = *(const uint4*)&A[(unsigned long long)gr * K + gc]; \
            } else { \
                /* Partial tail: per-element so a slice straddling K keeps its */ \
                /* in-bounds cols (and OOB rows still produce zeros). Union pun */ \
                /* avoids a misaligned uint4* cast / strict-aliasing hazard. */ \
                union { __nv_bfloat16 h[8]; uint4 v; } u; \
                _Pragma("unroll") \
                for (int i = 0; i < 8; i++) \
                    u.h[i] = ((gr < M) && (gc + i < K)) \
                        ? A[(unsigned long long)gr * K + gc + i] \
                        : __float2bfloat16(0.0f); \
                (ra)[rnd] = u.v; \
            } \
        } \
        { \
            unsigned int gk = (kb) + b_row; \
            unsigned int gn = cta_n + b_col; \
            /* 16-B load needs a 16-B-aligned src: B_t[gk*N+gn] aligned iff */ \
            /* N%16==0 (gn is already %16). Uniform branch; misaligned N -> scalar. */ \
            if ((gk < K) && (gn + 15 < N) && ((N & 15) == 0)) { \
                (rb) = *(const uint4*)&B_t[(unsigned long long)gk * N + gn]; \
            } else { \
                /* Partial tail: fill byte-by-byte so a 16-col chunk that */ \
                /* straddles N (or an OOB K row) keeps its in-bounds bytes. */ \
                /* Union pun avoids a misaligned uint4* cast. */ \
                union { unsigned char b[16]; uint4 v; } u; \
                _Pragma("unroll") \
                for (int i = 0; i < 16; i++) \
                    u.b[i] = ((gk < K) && (gn + i < N)) \
                        ? B_t[(unsigned long long)gk * N + gn + i] : 0; \
                (rb) = u.v; \
            } \
            /* Guard the scale read: when K==0 the K-block dim is empty and */ \
            /* block_scale_t[0] would be OOB. rs=0 -> smem_B=0 -> C stays 0. */ \
            (rs) = ((kb) < K) ? block_scale_t[ \
                (unsigned long long)((kb) >> 7) * n_scale_blocks + n_block] \
                : 0.0f; \
        } \
    } while(0)

    // Commit prefetched tile: A straight through; FP8 B dequants in registers
    // (bytes never round-trip through smem) and writes transposed into
    // smem_B[n][k]. One scale covers the whole 32-K x 128-N tile.
    #define W8M128_STORE_TILE(buf, ra, rb, rs) do { \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
            *(uint4*)&smem_A[(buf)][row][a_col] = (ra)[rnd]; \
        } \
        { \
            /* Extract the 16 FP8 bytes from the uint4 words with shifts — */ \
            /* keeps reg_B register-resident (no &rb aliasing -> local mem). */ \
            /* NOTE: the strided smem_B[n][k] store is bank-conflicted (the */ \
            /* n-groups are 16 apart and 16*rowstride_words aliases mod 32), */ \
            /* but a measured A/B showed a rotated-store order costs MORE */ \
            /* (runtime-indexed rw[] spills) than the conflicts it saves.   */ \
            const unsigned int rw[4] = { (rb).x, (rb).y, (rb).z, (rb).w }; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) { \
                unsigned char byte = (unsigned char)((rw[i >> 2] >> ((i & 3) * 8)) & 0xFFu); \
                smem_B[(buf)][b_col + i][b_row] = \
                    __float2bfloat16(w8m128_e4m3(byte) * (rs)); \
            } \
        } \
    } while(0)

    // Each warp covers one 16-row band x all 8 n-sub-tiles. smem_B[n][k]
    // fragment read identical to w4a16_gemm_t_m128's WMMA compute.
    #define W8M128_COMPUTE(a_buf, b_buf) do { \
        unsigned int m_row = warp_m_offset + (lane_id & 15); \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            v16bf a; \
            memcpy(&a, &smem_A[(a_buf)][m_row][h * 16], 32); \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                memcpy(&b, &smem_B[(b_buf)][nc][h * 16], 32); \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    uint4 reg_A[2], reg_B;
    float reg_S;
    W8M128_LOAD_REGS(0, reg_A, reg_B, reg_S);
    W8M128_STORE_TILE(0, reg_A, reg_B, reg_S);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = W8M128_KSTEP; k_base < K; k_base += W8M128_KSTEP) {
        int nxt = 1 - cur;
        W8M128_LOAD_REGS(k_base, reg_A, reg_B, reg_S);   // global->regs in flight
        W8M128_COMPUTE(cur, cur);                        // WMMA overlaps the loads
        W8M128_STORE_TILE(nxt, reg_A, reg_B, reg_S);     // regs->smem (+ B dequant)
        __syncthreads();
        cur = nxt;
    }
    W8M128_COMPUTE(cur, cur);

    #undef W8M128_LOAD_REGS
    #undef W8M128_STORE_TILE
    #undef W8M128_COMPUTE

    // Each warp writes its own 16-row band x 128 cols (no shuffle).
    const unsigned int row_base = cta_m + warp_m_offset;
    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = row_base + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}
