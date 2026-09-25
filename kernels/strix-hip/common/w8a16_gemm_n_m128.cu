// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas W8A16 non-transposed M128 GEMM — FP8 E4M3 block-scaled, 128x128 tile.
// HIP/gfx1151 (AMD WMMA) port.
//
// C[M,N] = A[M,K] (BF16) * dequant(B[N,K] (FP8 E4M3, NON-transposed))
//
// B is stored [N, K] (k-contiguous — the checkpoint's native row-major weight),
// NOT transposed. Each thread loads 16 consecutive k at one n (a uint4), so the
// dequant writes a CONTIGUOUS 16-element run into smem_B[n][k] — two uint4
// stores, no strided-store bank conflicts, no in-kernel transpose.
// Block scales: block_scale[N/128, K/128] FP32 (non-transposed). Dequant:
// E4M3_decode[byte] * scale (one scale covers the whole 128x128 tile).
//
// WMMA: __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 (16x16x16, n16).
//   Store: lane l, acc element e: row = row_base + 2*e + (l>>4), col = col_base + (l&15)
//
// Register-prefetch double-buffered pipeline — the SAME skeleton as
// w8a16_gemm_t_m128, differing ONLY in the B path: the non-transposed [N,K]
// layout lets the B tile load + store contiguously (vs the [K,N] transposed
// variant's strided smem_B[n][k] writes that bank-conflict ~16-way).
//
// Grid: (ceil(N/128), ceil(M/128), 1), Block: (256,1,1). Tail-M/N/K predicated.

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Bit-math OCP-e4m3fn decode — register ALU, no LDS. Produces the SAME value as
// the E4M3_LUT table the base w8a16 kernels use, including the S.1111.111 NaN.
// E4M3 -> fp32, bit-exact to the reference decode it replaces (normal, subnormal,
// e4m3fn NaN -> qNaN, sign) in ~6 VALU instead of ~14 with selects. (b & 0x7F) << 20
// as an fp32 is |E4M3| * 2^-120; for E4M3 subnormals it is an fp32 DENORMAL
// (m * 2^-129), so * 2^120 is exact either way (this code runs with f32 denormals
// preserved: .amdhsa_float_denorm_mode_32 3). Sign by XOR, as -v does. The caller's
// * block_scale is unchanged, so the product rounds exactly as before. Host-checked:
// 256 bytes x 12 scales (incl. negative, 1e-9 .. 1000): 0 fp32 bit mismatches.
// Reference, for the record:
//   e == 0            -> m * 2^-9
//   e == 15 && m == 7 -> qNaN
//   else              -> ((e + 120) << 23) | (m << 20);   negated when s
__device__ __forceinline__ float w8n128_e4m3(unsigned char b) {
    const unsigned int u = b;
    const float mag = ((u & 0x7Fu) == 0x7Fu) ? __uint_as_float(0x7fc00000u)
                                             : __uint_as_float((u & 0x7Fu) << 20) * 0x1p120f;
    return __uint_as_float(__float_as_uint(mag) ^ ((u & 0x80u) << 24));
}

#define W8N128_M_TILE 128   // M rows per CTA (8 warps x 16 rows)
#define W8N128_N_TILE 128   // N cols per CTA
#define W8N128_KSTEP  32
#define W8N128_APAD   8     // smem_A row stride 40 -> 80 B, 16-B aligned
#define W8N128_BPAD   8     // smem_B[n][k] row stride pad
#define W8N128_FP8B   128   // FP8 block-scale granularity (128x128)

// Invariants the single-scale-per-tile dequant relies on: a K_STEP tile never
// straddles a 128-K scale block, and the N tile is exactly one 128-N block.
static_assert(W8N128_FP8B % W8N128_KSTEP == 0, "K_STEP must divide the 128-K scale block");
static_assert(W8N128_N_TILE == W8N128_FP8B, "N tile must equal one 128-N scale block");


// Grouped (GROUP_M) CTA order: GM consecutive CTAs share one N tile across GM M tiles, so a B
// tile is fetched from DRAM once per group instead of once per M tile. Pure index remap over
// the same (gridDim.x x gridDim.y) set of tiles, so every output tile is computed exactly as before.
#define W8N128_GROUP_M 8
__device__ __forceinline__ void w8n128_swizzle_mn(unsigned int gm, unsigned int& m_tile, unsigned int& n_tile) {
    const unsigned int num_n = gridDim.x, num_m = gridDim.y;
    const unsigned int bid = blockIdx.y * num_n + blockIdx.x;
    const unsigned int per_group = gm * num_n;
    const unsigned int group = bid / per_group;
    const unsigned int first_m = group * gm;
    const unsigned int gsize = (num_m - first_m) < gm ? (num_m - first_m) : gm;
    const unsigned int local = bid - group * per_group;
    m_tile = first_m + local % gsize;
    n_tile = local / gsize;
}
extern "C" __global__
__launch_bounds__(256, 1)
void w8a16_gemm_n_m128(
    const __nv_bfloat16* __restrict__ A,                // [M, K] BF16
    const unsigned char* __restrict__ B,                // [N, K] FP8 E4M3 (k-contiguous)
    const float* __restrict__ block_scale,              // [N/128, K/128] FP32 (non-transposed)
    __nv_bfloat16* __restrict__ C,                      // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int m_t, n_t;
    w8n128_swizzle_mn(W8N128_GROUP_M, m_t, n_t);
    const unsigned int cta_n = n_t * W8N128_N_TILE;
    const unsigned int cta_m = m_t * W8N128_M_TILE;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int warp_id = threadIdx.x >> 5;       // 0..7
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int warp_m_offset = warp_id * 16;     // 0..112 (one 16-row band/warp)

    __shared__ __align__(16) __nv_bfloat16 smem_A[2][W8N128_M_TILE][W8N128_KSTEP + W8N128_APAD];
    __shared__ __align__(16) __nv_bfloat16 smem_B[2][W8N128_N_TILE][W8N128_KSTEP + W8N128_BPAD];

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    // Per-thread register-prefetch state for one K_STEP=32 tile:
    //   A: 128 rows x 32 cols bf16 -> 512 uint4; 256 threads -> 2 uint4 (one
    //      8-col slice of a row, two rounds cover 64+64 rows).
    //   B: 128 N-rows x 32 K-cols FP8 = 4096 B -> 256 uint4; 256 threads -> 1
    //      uint4 (16 CONTIGUOUS K-bytes of one N-row, since B is [N,K]).
    const unsigned int a_row_base = threadIdx.x >> 2;          // 0..63
    const unsigned int a_col      = (threadIdx.x & 3) << 3;    // 0,8,16,24
    // Non-transposed B: thread t -> n_local = t>>1 (0..127), k_half = t&1.
    // Each thread reads B[gn][k_base + k_half*16 .. +16] = 16 contiguous k bytes.
    const unsigned int b_n        = threadIdx.x >> 1;          // 0..127 (N row)
    const unsigned int b_koff     = (threadIdx.x & 1) << 4;    // 0 or 16 (K col offset)
    const unsigned int k_scale_blocks = (K + W8N128_FP8B - 1) / W8N128_FP8B;
    const unsigned int n_block      = cta_n >> 7;              // == blockIdx.x

    #define W8N128_LOAD_REGS(kb, ra, rb, rs) do { \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
            unsigned int gr  = cta_m + row; \
            unsigned int gc  = (kb) + a_col; \
            /* 16-B load needs a 16-B-aligned src: A[gr*K+gc] aligned iff K%8==0 */ \
            if ((gr < M) && (gc + 7 < K) && ((K & 7) == 0)) { \
                (ra)[rnd] = *(const uint4*)&A[(unsigned long long)gr * K + gc]; \
            } else { \
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
            /* Non-transposed B[N,K]: read 16 CONTIGUOUS k at one n. */ \
            unsigned int gn = cta_n + b_n; \
            unsigned int gk = (kb) + b_koff; \
            /* 16-B load needs a 16-B-aligned src: B[gn*K+gk] aligned iff */ \
            /* K%16==0 (gk is already %16). Uniform branch; else scalar.   */ \
            if ((gn < N) && (gk + 15 < K) && ((K & 15) == 0)) { \
                (rb) = *(const uint4*)&B[(unsigned long long)gn * K + gk]; \
            } else { \
                union { unsigned char b[16]; uint4 v; } u; \
                _Pragma("unroll") \
                for (int i = 0; i < 16; i++) \
                    u.b[i] = ((gn < N) && (gk + i < K)) \
                        ? B[(unsigned long long)gn * K + gk + i] : 0; \
                (rb) = u.v; \
            } \
            /* Non-transposed block scale [N/128, K/128]: scale[nb][kb]. */ \
            (rs) = ((kb) < K) ? block_scale[ \
                (unsigned long long)n_block * k_scale_blocks + ((kb) >> 7)] \
                : 0.0f; \
        } \
    } while(0)

    // Commit prefetched tile: A straight through; FP8 B dequants in registers
    // into a bf16[16] union (constant indices) then writes TWO contiguous uint4
    // into smem_B[n][k] — no transpose, no strided/conflicted scalar stores.
    #define W8N128_STORE_TILE(buf, ra, rb, rs) do { \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
            *(uint4*)&smem_A[(buf)][row][a_col] = (ra)[rnd]; \
        } \
        { \
            union { __nv_bfloat16 h[16]; uint4 v[2]; } u; \
            const unsigned int rw[4] = { (rb).x, (rb).y, (rb).z, (rb).w }; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) { \
                unsigned char byte = (unsigned char)((rw[i >> 2] >> ((i & 3) * 8)) & 0xFFu); \
                u.h[i] = __float2bfloat16(w8n128_e4m3(byte) * (rs)); \
            } \
            /* smem_B[b_n][b_koff .. +15] is contiguous: two 16-B stores. */ \
            *(uint4*)&smem_B[(buf)][b_n][b_koff]     = u.v[0]; \
            *(uint4*)&smem_B[(buf)][b_n][b_koff + 8] = u.v[1]; \
        } \
    } while(0)

    // Each warp covers one 16-row band x all 8 n-sub-tiles. smem_B[n][k]
    // fragment read identical to w8a16_gemm_t_m128's WMMA compute.
    #define W8N128_COMPUTE(a_buf, b_buf) do { \
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
    W8N128_LOAD_REGS(0, reg_A, reg_B, reg_S);
    W8N128_STORE_TILE(0, reg_A, reg_B, reg_S);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = W8N128_KSTEP; k_base < K; k_base += W8N128_KSTEP) {
        int nxt = 1 - cur;
        W8N128_LOAD_REGS(k_base, reg_A, reg_B, reg_S);   // global->regs in flight
        W8N128_COMPUTE(cur, cur);                        // WMMA overlaps the loads
        W8N128_STORE_TILE(nxt, reg_A, reg_B, reg_S);     // regs->smem (+ B dequant)
        __syncthreads();
        cur = nxt;
    }
    W8N128_COMPUTE(cur, cur);

    #undef W8N128_LOAD_REGS
    #undef W8N128_STORE_TILE
    #undef W8N128_COMPUTE

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
