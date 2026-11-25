// SPDX-License-Identifier: AGPL-3.0-only
// Derived from CIRU AI's Apache-2.0 `hip_moe_decode.hip` (ciru-ai v1.5.0,
// section 2, BANK_MODE 0 = source NVFP4) and `source_wmma_pretile.py`:
// zero-smem_B register-unpack K64 software-pipelined WMMA on a pretiled
// weight layout. Original code copyright (c) 2026 CIRU AI, licensed under
// Apache-2.0; this derivative is AGPL-3.0-only.

// strix-hip only: batched NVFP4 projection for decode/verify rows (M <= 16).
//   y[M,N] = x[M,K] . dequant(W[N,K])^T
//
// Two kernels:
//   w4a16_wmma_pretile — load-time repack of the shared NVFP4 layout
//     (B_packed[N, K/2] + B_scale[N, K/16]) into the WMMA tile replay:
//     [(N/128) x (K/64)] tiles of [128 rows x 32 packed bytes] + [128] u32
//     scale words. Each lane's 16-byte load lands directly in its WMMA
//     B-fragment — no LDS for weights. Same bytes, permuted.
//   w4a16_wmma — one block per 128-column tile, 8 warps: warp w covers
//     N-rows (w%4)*32 and the K64 tiles kt = w/4 (mod 2), so two warp
//     groups interleave independent K streams per block. A fragments load
//     straight from global (bf16 -> f16 convert in registers; the A tile
//     is 2 KB and stays L1-hot) — no __syncthreads in the K loop, each
//     warp free-runs its software pipeline. Partial accumulators merge
//     across the two K groups in LDS once at the end.

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned short ink_f32_to_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

__device__ __forceinline__ float ink_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// BF16x8 -> F16x8 (rtz); activations are small so the range always fits.
__device__ __forceinline__ uint4 ink_bf16x8_to_f16x8(uint4 v) {
    auto cvt2 = [](unsigned int u) -> unsigned int {
        float f0 = ink_bf16_to_f32((unsigned short)(u & 0xFFFFu));
        float f1 = ink_bf16_to_f32((unsigned short)(u >> 16));
        unsigned int pk;
        asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(pk) : "v"(f0), "v"(f1));
        return pk;
    };
    return make_uint4(cvt2(v.x), cvt2(v.y), cvt2(v.z), cvt2(v.w));
}

// Row lane16, K elements [k_off, k_off+16) of X[M,K] as one A fragment.
__device__ __forceinline__ half16 ink_a_frag(
    const unsigned short* __restrict__ X, int row, int k_off, int M, int K
) {
    half16 out;
    if (row < M) {
        const unsigned short* p = X + (long long)row * K + k_off;
        uint4 lo = ink_bf16x8_to_f16x8(*reinterpret_cast<const uint4*>(p));
        uint4 hi = ink_bf16x8_to_f16x8(*reinterpret_cast<const uint4*>(p + 8));
        unsigned int w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        __builtin_memcpy(&out, w, 32);
    } else {
        unsigned int w[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        __builtin_memcpy(&out, w, 32);
    }
    return out;
}

// 8 packed bytes = 16 NVFP4 nibbles, with one E4M3 scale byte -> 16 f16.
// The perm LUT emits {0,1,2,3,4,6,8,12} f16 (E2M1 magnitudes x2) and the
// scale decode (byte<<7)+0x1C00 is e4m3/2, so the product is exactly
// E2M1 * E4M3 — the shared kernels' decode.
__device__ __forceinline__ void ink_unpack_nvfp4(
    unsigned int p0, unsigned int p1, unsigned int scale_byte, uint4 &o0, uint4 &o1
) {
    unsigned int scale_f16 = (scale_byte << 7) + 0x1C00u;
    unsigned int scale_pk = scale_f16 | (scale_f16 << 16);
    unsigned int dst[8];
    const unsigned int words[2] = {p0, p1};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        unsigned int u = words[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            unsigned int packed = (u >> (b * 8)) & 0xFFu;
            unsigned int sel = 0x000C000Cu | ((packed & 7u) << 8) | ((packed & 0x70u) << 20);
            unsigned int mags;
            asm("v_perm_b32 %0, %1, %2, %3"
                : "=v"(mags) : "v"(0x4A484644u), "v"(0x42403C00u), "v"(sel));
            unsigned int sgn = mags | ((packed & 8u) << 12) | ((packed & 128u) << 24);
            unsigned int scaled;
            asm("v_pk_mul_f16 %0, %1, %2"
                : "=v"(scaled) : "v"(sgn), "v"(scale_pk));
            dst[w * 4 + b] = scaled;
        }
    }
    o0 = make_uint4(dst[0], dst[1], dst[2], dst[3]);
    o1 = make_uint4(dst[4], dst[5], dst[6], dst[7]);
}

// Pretiled weights: Packed[(n_blk*K/64 + kt)*128 + row][32B] u8,
// Raw (same index) [1 u32] = the row's 4 E4M3 scale bytes for the tile.
extern "C" __global__ __launch_bounds__(256, 4) void w4a16_wmma(
    const unsigned short* __restrict__ X,       // [M, K] BF16
    const unsigned int* __restrict__ Packed,    // pretiled codes (u32 view)
    const unsigned int* __restrict__ Raw,       // pretiled scale words
    float* __restrict__ Partials,               // [n_split, M, N] F32 scratch
    unsigned int M, unsigned int N, unsigned int K
) {
    const int n_blk = (int)blockIdx.x;
    const int kg = (int)blockIdx.y;     // split-K slice index
    const int n_split = (int)gridDim.y; // K slices total
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int half = lane >> 4;
    const int kgrp = warp >> 2;   // K tile parity this warp covers
    const int wrow = warp & 3;    // N16 row pair inside the 128 tile

    const int row0 = wrow * 32 + lane16;
    const int row1 = row0 + 16;

    const int n_kt64 = (int)K >> 6;
    const int khalf = (n_kt64 + n_split - 1) / n_split; // K64 tiles per slice
    const int kbase = kg * khalf;
    const int ktiles = min(n_kt64 - kbase, khalf);
    const unsigned int* p0_base = Packed + ((long long)n_blk * n_kt64 * 128 + row0) * 8 + half * 4;
    const unsigned int* p1_base = Packed + ((long long)n_blk * n_kt64 * 128 + row1) * 8 + half * 4;
    const unsigned int* r0_base = Raw + (long long)n_blk * n_kt64 * 128 + row0;
    const unsigned int* r1_base = Raw + (long long)n_blk * n_kt64 * 128 + row1;

    // Prologue: issue this group's first weight tile immediately. A K group
    // whose parity has no tile in a short last slice (e.g. ktiles == 1)
    // must not touch memory past the buffer.
    uint4 p0_nxt = make_uint4(0, 0, 0, 0), p1_nxt = make_uint4(0, 0, 0, 0);
    unsigned int r0_nxt = 0, r1_nxt = 0;
    if (kgrp < ktiles) {
        const long long t0 = (long long)(kbase + kgrp) * 128;
        p0_nxt = *reinterpret_cast<const uint4*>(p0_base + t0 * 8);
        p1_nxt = *reinterpret_cast<const uint4*>(p1_base + t0 * 8);
        r0_nxt = r0_base[t0];
        r1_nxt = r1_base[t0];
    }

    float8 acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8 acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto unpack_pair = [&](unsigned int pw0, unsigned int pw1, unsigned int raw_w,
                           int kk_lo, half16 &b_lo, half16 &b_hi) {
        const int kk4 = (half == 0) ? kk_lo : (kk_lo + 2);
        unsigned int sb = (raw_w >> (kk4 * 8)) & 0xFFu;
        uint4 o0, o1;
        ink_unpack_nvfp4(pw0, pw1, sb, o0, o1);
        unsigned int w[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
        unsigned int o[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __shfl_xor(w[i], 16, 32);
        __builtin_memcpy(&b_lo, (half == 0) ? w : o, 32);
        __builtin_memcpy(&b_hi, (half == 0) ? o : w, 32);
    };

    const int n_iter = (ktiles + 1) >> 1;
    for (int it = 0; it < n_iter; ++it) {
        const int kt = kbase + it * 2 + kgrp;
        if (kt >= kbase + ktiles) break;
        const int k_off = kt << 6;

        uint4 p0 = p0_nxt, p1 = p1_nxt;
        unsigned int r0w = r0_nxt, r1w = r1_nxt;
        if (kt + 2 < kbase + ktiles) {
            const long long next = (long long)(kt + 2) * 128;
            p0_nxt = *reinterpret_cast<const uint4*>(p0_base + next * 8);
            p1_nxt = *reinterpret_cast<const uint4*>(p1_base + next * 8);
            r0_nxt = r0_base[next];
            r1_nxt = r1_base[next];
        }

        {
            half16 a0 = ink_a_frag(X, lane16, k_off + 0, (int)M, (int)K);
            half16 a2 = ink_a_frag(X, lane16, k_off + 32, (int)M, (int)K);
            half16 b0, b2;
            unpack_pair(p0.x, p0.y, r0w, 0, b0, b2);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b2, acc0);
            unpack_pair(p1.x, p1.y, r1w, 0, b0, b2);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b2, acc1);
        }
        {
            half16 a1 = ink_a_frag(X, lane16, k_off + 16, (int)M, (int)K);
            half16 a3 = ink_a_frag(X, lane16, k_off + 48, (int)M, (int)K);
            half16 b1, b3;
            unpack_pair(p0.z, p0.w, r0w, 1, b1, b3);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3, b3, acc0);
            unpack_pair(p1.z, p1.w, r1w, 1, b1, b3);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3, b3, acc1);
        }
    }

    // Merge the two K groups' partial accumulators in LDS, then the first
    // 4 warps write this block's F32 partials (the reduce pass sums the two
    // K halves — f32 a+b is order-exact, so the result stays deterministic).
    __shared__ float red[8][2][32][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        red[warp][0][lane][i] = acc0[i];
        red[warp][1][lane][i] = acc1[i];
    }
    __syncthreads();
    if (warp >= 4) return;
    float* P = Partials + (long long)kg * M * N;
    const int cb = n_blk * 128 + wrow * 32;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int m = half + 2 * i;
        if (m >= (int)M) continue;
        P[(long long)m * N + cb + lane16] = acc0[i] + red[warp + 4][0][lane][i];
        P[(long long)m * N + cb + 16 + lane16] = acc1[i] + red[warp + 4][1][lane][i];
    }
}

// Second pass of the split-K pair: Y = bf16((sum_s P[s]) * scale2). One
// u32 store covers two adjacent outputs.
extern "C" __global__ __launch_bounds__(256) void w4a16_wmma_reduce(
    const float* __restrict__ P,            // [n_split, M, N]
    const float scale2,
    unsigned short* __restrict__ Y,
    unsigned int MN,
    unsigned int n_split
) {
    const unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i * 2 >= MN) return;
    const float* p = P + (long long)i * 2;
    float a = 0.f, b = 0.f;
    for (unsigned int s = 0; s < n_split; ++s) {
        a += p[0];
        b += p[1];
        p += MN;
    }
    unsigned int pk = (unsigned int)ink_f32_to_bf16(a * scale2);
    if (i * 2 + 1 < MN)
        pk |= ((unsigned int)ink_f32_to_bf16(b * scale2)) << 16;
    *reinterpret_cast<unsigned int*>(Y + (long long)i * 2) = pk;
}

// Load-time repack into the WMMA tile replay (see header). One thread per
// (n128 row, K64 tile): copies 32 packed bytes + 4 scale bytes.
extern "C" __global__ __launch_bounds__(128) void w4a16_wmma_pretile(
    const unsigned char* __restrict__ packed,   // [N, K/2]
    const unsigned char* __restrict__ scales,   // [N, K/16]
    unsigned char* __restrict__ out_packed,     // [N/128 x K/64][128][32]
    unsigned char* __restrict__ out_scales,     // [N/128 x K/64][128][4]
    unsigned int N, unsigned int K
) {
    const int kt64 = (int)(K >> 6);
    const int t = (int)blockIdx.x;
    const int nb = t / kt64;
    const int kt = t % kt64;
    const int r = (int)threadIdx.x;
    const long long dst = ((long long)t * 128 + r);
    const long long src_row = (long long)nb * 128 + r;
    const unsigned char* sp = packed + src_row * (K >> 1) + kt * 32;
    uint4* dp = reinterpret_cast<uint4*>(out_packed + dst * 32);
    dp[0] = *reinterpret_cast<const uint4*>(sp);
    dp[1] = *reinterpret_cast<const uint4*>(sp + 16);
    *reinterpret_cast<unsigned int*>(out_scales + dst * 4) =
        *reinterpret_cast<const unsigned int*>(scales + src_row * (K >> 4) + kt * 4);
}
