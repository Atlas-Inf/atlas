// SPDX-License-Identifier: AGPL-3.0-only

// Native MXFP8 projections for nvidia/Qwen3.8-Flash-Next-FP4-Local.
//
// MXFP8 (OCP microscaling): B is [N, K] E4M3 bytes and S is [N, K/32] E8M0
// bytes, one power-of-two scale 2^(S-127) per 32 consecutive inputs of a row.
// The checkpoint ships the GDN, attention, indexer and shared-expert
// projections this way; these kernels read those bytes as they are on disk.
//
//   mxfp8_proj_gemv     C[n]    = sum_k A[k]    * W[n,k]          (M = 1)
//   mxfp8_proj_batchm   C[t, n] = sum_k A[t, k] * W[n,k]          (M <= 32)
//   mxfp8_proj_dequant  D[n, k] = W[n,k] as BF16                  (exact)
//
// with W[n,k] = e4m3(B[n,k]) * 2^(S[n, k/32] - 127).
//
// Both GEMVs run the same templated body, so every row of `mxfp8_proj_batchm` is
// accumulated in exactly the order `mxfp8_proj_gemv` uses: a verify row and the
// sequential decode of the same token produce identical bits.
//
// Accumulation: each thread owns 16-input chunks (half an MXFP8 block). It
// sums the 16 products a * e4m3 in a fixed sequential order, scales that
// partial by the block's power of two (exact barring under/overflow), and
// adds it into an FP32 accumulator. Lanes then reduce by warp shuffle and the
// two warps of an output through shared memory.
//
// The dequant kernel feeds prefill: it expands one weight into a BF16 scratch
// that cuBLASLt multiplies. Every E4M3 value times a power of two in range is
// exactly representable in BF16, so the scratch is the checkpoint's weight,
// not an approximation of it.
//
// Grid (GEMVs): (ceil(N / 4)), Block: 256 — four outputs, 64 threads each.
// Grid (dequant): (ceil(N * K / 16 / 256)), Block: 256.
// Requires K % 32 == 0 and 16-byte aligned A, B, D rows.

#include <cuda_bf16.h>

#define MX8_BLOCK_THREADS 256
#define MX8_N_PER_BLOCK 4
#define MX8_THREADS_PER_OUT (MX8_BLOCK_THREADS / MX8_N_PER_BLOCK)
#define MX8_GROUP 32
#define MX8_MAX_M 32

// E4M3 byte -> f32. NaN (0x7F / 0xFF) maps to 0, as in the W8A16 tables.
__device__ __forceinline__ float mx8_e4m3_to_f32(unsigned int v) {
    const unsigned int e = (v >> 3) & 0xF;
    const unsigned int m = v & 0x7;
    float r;
    if (e == 0) {
        r = (float)m * 0.001953125f;  // subnormal: m * 2^-9
    } else if (e == 15 && m == 7) {
        r = 0.0f;
    } else {
        r = __int_as_float(((e + 120) << 23) | (m << 20));
    }
    return (v & 0x80) ? -r : r;
}

// E8M0 byte -> 2^(e - 127). 0xFF (NaN) maps to 0; 0 is the subnormal 2^-127.
__device__ __forceinline__ float mx8_e8m0_to_f32(unsigned int e) {
    if (e == 0xFF) return 0.0f;
    if (e == 0) return __int_as_float(0x00400000);
    return __int_as_float(e << 23);
}

__device__ __forceinline__ float mx8_bf16_lo(unsigned int x) {
    return __uint_as_float(x << 16);
}

__device__ __forceinline__ float mx8_bf16_hi(unsigned int x) {
    return __uint_as_float(x & 0xFFFF0000u);
}

// Sum of the 16 products of one weight chunk and one activation chunk, in
// input order. The weight chunk reaches this pre-decoded through the E4M3
// LUT: w_f[i] is exactly lut[byte i of w] (same float value), so each row's
// FMA expression is unchanged; the decode just happens once per chunk for
// every row in the bucket instead of once per (chunk, row) pair.
__device__ __forceinline__ float mx8_chunk_dot(
    const float* __restrict__ w_f, uint4 a0, uint4 a1
) {
    const unsigned int av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    float p = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned int lo = av[2 * i];
        const unsigned int hi = av[2 * i + 1];
        p += mx8_bf16_lo(lo) * w_f[4 * i];
        p += mx8_bf16_hi(lo) * w_f[4 * i + 1];
        p += mx8_bf16_lo(hi) * w_f[4 * i + 2];
        p += mx8_bf16_hi(hi) * w_f[4 * i + 3];
    }
    return p;
}

// M is the launch bucket; m is the live row count (m <= M). Row t >= m reads
// row m-1's activations and its accumulator is never stored, so out-of-bounds
// A reads can't happen while per-row arithmetic stays identical for t < m.
template <int M>
__device__ __forceinline__ void mx8_gemv_rows(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int m
) {
    __shared__ float s_lut[256];
    __shared__ float s_red[MX8_N_PER_BLOCK * 2 * M];
    s_lut[threadIdx.x] = mx8_e4m3_to_f32(threadIdx.x);
    __syncthreads();

    const unsigned int local_out = threadIdx.x / MX8_THREADS_PER_OUT;
    const unsigned int lane = threadIdx.x % MX8_THREADS_PER_OUT;
    const unsigned int n = blockIdx.x * MX8_N_PER_BLOCK + local_out;
    const bool live = n < N;

    float acc[M];
    #pragma unroll
    for (int t = 0; t < M; t++) acc[t] = 0.0f;

    if (live) {
        const unsigned int k16s = K / 16;
        const uint4* wrow = (const uint4*)(B + (unsigned long long)n * K);
        const unsigned char* srow = S + (unsigned long long)n * (K / MX8_GROUP);
        for (unsigned int c = lane; c < k16s; c += MX8_THREADS_PER_OUT) {
            const uint4 w = wrow[c];
            const float scale = mx8_e8m0_to_f32(srow[c >> 1]);
            // One LUT decode for all rows in the bucket: w_f[i] is the same
            // float the per-row expression used to fetch.
            float w_f[16];
            const unsigned int wv[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                w_f[4 * i] = s_lut[wv[i] & 0xFF];
                w_f[4 * i + 1] = s_lut[(wv[i] >> 8) & 0xFF];
                w_f[4 * i + 2] = s_lut[(wv[i] >> 16) & 0xFF];
                w_f[4 * i + 3] = s_lut[wv[i] >> 24];
            }
            #pragma unroll
            for (int t = 0; t < M; t++) {
                const unsigned int tr = (unsigned int)t < m ? (unsigned int)t : m - 1;
                const uint4* arow = (const uint4*)(A + (unsigned long long)tr * K);
                const float p = mx8_chunk_dot(w_f, arow[2 * c], arow[2 * c + 1]);
                acc[t] += p * scale;
            }
        }
    }

    const unsigned int warp_in_out = lane / 32;
    #pragma unroll
    for (int t = 0; t < M; t++) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if ((lane & 31) == 0) s_red[(local_out * 2 + warp_in_out) * M + t] = v;
    }
    __syncthreads();
    if (live && lane < m) {
        const unsigned int t = lane;
        const float r = s_red[(local_out * 2) * M + t] + s_red[(local_out * 2 + 1) * M + t];
        C[(unsigned long long)t * N + n] = __float2bfloat16_rn(r);
    }
}

extern "C" __global__ void __launch_bounds__(MX8_BLOCK_THREADS) mxfp8_proj_gemv(
    const __nv_bfloat16* __restrict__ A,   // [1, K]
    const unsigned char* __restrict__ B,   // [N, K] E4M3
    const unsigned char* __restrict__ S,   // [N, K/32] E8M0
    __nv_bfloat16* __restrict__ C,         // [1, N]
    unsigned int N,
    unsigned int K
) {
    mx8_gemv_rows<1>(A, B, S, C, N, K, 1);
}

extern "C" __global__ void __launch_bounds__(MX8_BLOCK_THREADS) mxfp8_proj_batchm(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const unsigned char* __restrict__ B,   // [N, K] E4M3
    const unsigned char* __restrict__ S,   // [N, K/32] E8M0
    __nv_bfloat16* __restrict__ C,         // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int bucket = M <= 8 ? M : (M <= 16 ? 16 : (M <= 24 ? 24 : 32));
    const unsigned int m = M;
    switch (bucket) {
        case 1: mx8_gemv_rows<1>(A, B, S, C, N, K, 1); break;
        case 2: mx8_gemv_rows<2>(A, B, S, C, N, K, 2); break;
        case 3: mx8_gemv_rows<3>(A, B, S, C, N, K, 3); break;
        case 4: mx8_gemv_rows<4>(A, B, S, C, N, K, 4); break;
        case 5: mx8_gemv_rows<5>(A, B, S, C, N, K, 5); break;
        case 6: mx8_gemv_rows<6>(A, B, S, C, N, K, 6); break;
        case 7: mx8_gemv_rows<7>(A, B, S, C, N, K, 7); break;
        case 8: mx8_gemv_rows<8>(A, B, S, C, N, K, 8); break;
        // Buckets past 8 clamp their live rows at runtime m: one weight pass
        // serves the whole verify window.
        case 16: mx8_gemv_rows<16>(A, B, S, C, N, K, m); break;
        case 24: mx8_gemv_rows<24>(A, B, S, C, N, K, m); break;
        case 32: mx8_gemv_rows<32>(A, B, S, C, N, K, m); break;
        default: break;  // the host refuses M outside 1..=32 or non-bucket M
    }
}

extern "C" __global__ void __launch_bounds__(MX8_BLOCK_THREADS) mxfp8_proj_dequant(
    const unsigned char* __restrict__ B,   // [N, K] E4M3
    const unsigned char* __restrict__ S,   // [N, K/32] E8M0
    __nv_bfloat16* __restrict__ D,         // [N, K] BF16
    unsigned int N,
    unsigned int K
) {
    const unsigned long long chunk = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long chunks = (unsigned long long)N * (K / 16);
    if (chunk >= chunks) return;
    const unsigned int k16s = K / 16;
    const unsigned long long n = chunk / k16s;
    const unsigned int c = (unsigned int)(chunk % k16s);
    const float scale = mx8_e8m0_to_f32(S[n * (K / MX8_GROUP) + (c >> 1)]);
    const uint4 w = ((const uint4*)B)[chunk];
    const unsigned int wv[4] = {w.x, w.y, w.z, w.w};
    unsigned int out[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const unsigned int b0 = (wv[i] >> (16 * j)) & 0xFF;
            const unsigned int b1 = (wv[i] >> (16 * j + 8)) & 0xFF;
            const __nv_bfloat16 v0 = __float2bfloat16_rn(mx8_e4m3_to_f32(b0) * scale);
            const __nv_bfloat16 v1 = __float2bfloat16_rn(mx8_e4m3_to_f32(b1) * scale);
            out[2 * i + j] = (unsigned int)__bfloat16_as_ushort(v0)
                | ((unsigned int)__bfloat16_as_ushort(v1) << 16);
        }
    }
    uint4* dst = (uint4*)(D + chunk * 16);
    dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
    dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

// mxfp8_proj_rows32: the rows32t-style tiling over the same arithmetic. Each
// thread owns one lane of TWO outputs (n0, n0+4 within the block), so every
// activation chunk it loads feeds both: the per-(row, output) computation is
// mx8_gemv_rows<1>'s exactly -- 64 lanes walk chunks c = lane + 64j in order,
// p is the 16 products in input order, acc += p * scale, the two-warp shfl
// tree, lane0 + lane32 rounded once. The weight chunk is decoded through the
// LUT once per (chunk, output) and reused for all rows, and dead rows
// (t >= m) read row m-1 and are never stored.
// Grid: (ceil(N / 8)), Block: 256 -- eight outputs, 64 threads each.
extern "C" __global__ void __launch_bounds__(256) mxfp8_proj_rows32(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const unsigned char* __restrict__ B,   // [N, K] E4M3
    const unsigned char* __restrict__ S,   // [N, K/32] E8M0
    __nv_bfloat16* __restrict__ C,         // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ float s_lut[256];
    __shared__ float s_red[8 * 2 * 32];
    s_lut[threadIdx.x] = mx8_e4m3_to_f32(threadIdx.x);
    __syncthreads();

    const unsigned int grp = threadIdx.x / 64;   // output pair group 0..3
    const unsigned int lane = threadIdx.x % 64;
    const unsigned int n0 = blockIdx.x * 8 + grp * 2;
    const bool live0 = n0 < N, live1 = n0 + 1 < N;

    float acc0[MX8_MAX_M], acc1[MX8_MAX_M];
    #pragma unroll
    for (int t = 0; t < MX8_MAX_M; t++) { acc0[t] = 0.0f; acc1[t] = 0.0f; }

    if (live0) {
        const unsigned int k16s = K / 16;
        const uint4* wrow0 = (const uint4*)(B + (unsigned long long)n0 * K);
        const uint4* wrow1 = live1
            ? (const uint4*)(B + (unsigned long long)(n0 + 1) * K)
            : wrow0;
        const unsigned char* srow0 = S + (unsigned long long)n0 * (K / MX8_GROUP);
        const unsigned char* srow1 = live1
            ? S + (unsigned long long)(n0 + 1) * (K / MX8_GROUP)
            : srow0;
        for (unsigned int c = lane; c < k16s; c += 64) {
            const uint4 w0 = wrow0[c];
            const uint4 w1 = wrow1[c];
            const float sc0 = mx8_e8m0_to_f32(srow0[c >> 1]);
            const float sc1 = mx8_e8m0_to_f32(srow1[c >> 1]);
            float w0f[16], w1f[16];
            const unsigned int v0[4] = {w0.x, w0.y, w0.z, w0.w};
            const unsigned int v1[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                w0f[4 * i] = s_lut[v0[i] & 0xFF];
                w0f[4 * i + 1] = s_lut[(v0[i] >> 8) & 0xFF];
                w0f[4 * i + 2] = s_lut[(v0[i] >> 16) & 0xFF];
                w0f[4 * i + 3] = s_lut[v0[i] >> 24];
                w1f[4 * i] = s_lut[v1[i] & 0xFF];
                w1f[4 * i + 1] = s_lut[(v1[i] >> 8) & 0xFF];
                w1f[4 * i + 2] = s_lut[(v1[i] >> 16) & 0xFF];
                w1f[4 * i + 3] = s_lut[v1[i] >> 24];
            }
            #pragma unroll
            for (int t = 0; t < MX8_MAX_M; t++) {
                const unsigned int tr = (unsigned int)t < M ? (unsigned int)t : M - 1;
                const uint4* arow = (const uint4*)(A + (unsigned long long)tr * K);
                acc0[t] += mx8_chunk_dot(w0f, arow[2 * c], arow[2 * c + 1]) * sc0;
                acc1[t] += mx8_chunk_dot(w1f, arow[2 * c], arow[2 * c + 1]) * sc1;
            }
        }
    }

    const unsigned int wi = lane / 32;
    #pragma unroll
    for (int t = 0; t < MX8_MAX_M; t++) {
        float v0 = acc0[t], v1 = acc1[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v0 += __shfl_down_sync(0xFFFFFFFF, v0, o);
            v1 += __shfl_down_sync(0xFFFFFFFF, v1, o);
        }
        if ((lane & 31) == 0) {
            s_red[(grp * 2 * 2 + wi) * 32 + t] = v0;
            s_red[(grp * 2 * 2 + 2 + wi) * 32 + t] = v1;
        }
    }
    __syncthreads();
    if (lane < M) {
        const unsigned int t = lane;
        if (live0) C[(unsigned long long)t * N + n0] = __float2bfloat16_rn(
            s_red[(grp * 4) * 32 + t] + s_red[(grp * 4 + 1) * 32 + t]);
        if (live1) C[(unsigned long long)t * N + n0 + 1] = __float2bfloat16_rn(
            s_red[(grp * 4 + 2) * 32 + t] + s_red[(grp * 4 + 3) * 32 + t]);
    }
}

// mxfp8_proj_tc: tensor-core MXFP8 projection for 1..=32 rows, opt-in
// (ATLAS_QWEN4EXP_MXFP8_TC=1 — a different numerics family from the GEMV
// tiers, like MoE TC's contract: deterministic and row-invariant, but not
// bit-identical to mxfp8_proj_gemv). Grid (ceil(N/64), ceil(M/16)), block
// 128 = 4 warps x 2 n8 MMA tiles; each CTA owns a 16-row x 64-column output
// tile and streams its K range once: E4M3 + E8M0 decoded to BF16 into
// shared memory (E4M3 x 2^s is exactly representable in BF16, as in
// mxfp8_proj_dequant), then mma.sync m16n8k16 BF16->FP32 in fixed k order,
// rounded once to BF16 on store. Rows past m stage as zeros — every row's
// bits depend neither on M nor on its neighbours. K must be a multiple of
// 16 (the host refuses otherwise).
extern "C" __global__ void __launch_bounds__(128) mxfp8_proj_tc(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const unsigned char* __restrict__ B,   // [N, K] E4M3
    const unsigned char* __restrict__ S,   // [N, K/32] E8M0
    __nv_bfloat16* __restrict__ C,         // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ float s_lut[256];
    __shared__ __nv_bfloat16 smem_A[16][16 + 8];
    __shared__ __nv_bfloat16 smem_B[16][64 + 8];
    const unsigned int tid = threadIdx.x;
    s_lut[tid] = mx8_e4m3_to_f32(tid);
    s_lut[tid + 128] = mx8_e4m3_to_f32(tid + 128);
    __syncthreads();

    const unsigned int m_block = blockIdx.y * 16;
    const unsigned int n_block = blockIdx.x * 64;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int n_warp_base = warp_id * 16;

    float acc[2][4];
    #pragma unroll
    for (int t = 0; t < 2; t++)
        for (int j = 0; j < 4; j++) acc[t][j] = 0.0f;

    const unsigned short* sA_u16 = (const unsigned short*)smem_A;
    const unsigned short* sB_u16 = (const unsigned short*)smem_B;

    for (unsigned int k_base = 0; k_base < K; k_base += 16) {
        // A tile: 16x16 BF16, 2 elements per thread.
        for (unsigned int idx = tid; idx < 16 * 16; idx += 128) {
            const unsigned int r = idx / 16, c = idx % 16;
            const unsigned int gr = m_block + r;
            smem_A[r][c] = gr < M ? A[(unsigned long long)gr * K + k_base + c]
                                  : __float2bfloat16(0.0f);
        }
        // B tile: one uint4 (16 E4M3, one k16 block) of one output row per
        // thread tid < 64, decoded through the LUT by the row's scale byte.
        if (tid < 64) {
            const unsigned int gn = n_block + tid;
            if (gn < N) {
                const uint4 w =
                    *(const uint4*)(B + (unsigned long long)gn * K + k_base);
                const float sc = mx8_e8m0_to_f32(
                    S[(unsigned long long)gn * (K / 32) + (k_base >> 5)]);
                const unsigned int wv[4] = {w.x, w.y, w.z, w.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        smem_B[4 * i + j][tid] = __float2bfloat16_rn(
                            s_lut[(wv[i] >> (8 * j)) & 0xFF] * sc);
                    }
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < 16; kk++)
                    smem_B[kk][tid] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        const unsigned int ar0 = group_id, ar1 = group_id + 8;
        const unsigned int ac0 = tid_in_group * 2, ac1 = ac0 + 8;
        // PTX m16n8k16 A fragment, as dense_gemm_tc: a0 = row g, k 2t..2t+1;
        // a1 = row g+8, same k; a2 = row g, k +8; a3 = row g+8, k +8.
        const unsigned int a0 = *(const unsigned int*)&sA_u16[ar0 * 24 + ac0];
        const unsigned int a1 = *(const unsigned int*)&sA_u16[ar1 * 24 + ac0];
        const unsigned int a2 = *(const unsigned int*)&sA_u16[ar0 * 24 + ac1];
        const unsigned int a3 = *(const unsigned int*)&sA_u16[ar1 * 24 + ac1];
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const unsigned int n_col = n_warp_base + nt * 8 + group_id;
            const unsigned int k0 = tid_in_group * 2, k1 = k0 + 8;
            const unsigned int b0 = sB_u16[k0 * 72 + n_col]
                | ((unsigned int)sB_u16[(k0 + 1) * 72 + n_col] << 16);
            const unsigned int b1 = sB_u16[k1 * 72 + n_col]
                | ((unsigned int)sB_u16[(k1 + 1) * 72 + n_col] << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]),
                  "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]),
                  "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        const unsigned int r0 = m_block + group_id;
        const unsigned int r1 = m_block + group_id + 8;
        const unsigned int c0 = n_block + n_warp_base + nt * 8 + tid_in_group * 2;
        const unsigned int c1 = c0 + 1;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16_rn(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16_rn(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16_rn(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16_rn(acc[nt][3]);
    }
}
