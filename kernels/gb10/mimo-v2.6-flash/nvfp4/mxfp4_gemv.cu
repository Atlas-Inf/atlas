// SPDX-License-Identifier: AGPL-3.0-only
//
// MXFP4 tensor-core expert GEMV for MiMo-V2.6-Flash decode/verify.
//
// Replaces the scalar `moe_expert_gate_up_shared_t_e8m0` CUDA-core dot
// products with mma.sync m16n8k16 BF16 MMA, mirroring the proven
// `mxfp8_gemv.cu` structure (same warp/K-split/reduction order). Differences:
//   * Weights are FP4 (E2M1) nibbles, two values per byte (low nibble = even
//     K, high = odd — the scalar kernel's order), with E8M0 power-of-two
//     scales per 32 K values.
//   * Weights are stored TRANSPOSED [K/2, N] (the scalar GEMV's coalesced
//     layout), so each warp stages its own 64-K chunk of the CTA's 16-row
//     weight tile through smem into the row-major layout the MMA fragments
//     consume, then K-splits across the 8 warps exactly like the MXFP8 tiers.
//   * Decode is single-token (M = 1); C is [expert_slot * N + m * N + n].
//
// C[m, n] = sum_k A[m, k] * dequant(W[n, k]) with W[n, k] =
// lut[nibble] * 2^(scale - 127); the E8M0 scale multiply is exact in BF16
// (power of two), as in the MXFP8 tiers. Requires N % 16 == 0, K % 64 == 0.

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>

#define MX4_WARPS 8
#define MX4_WARP 32
#define MX4_TILE_N 16  // weight rows per CTA
#define MX4_CHUNK 64   // K values per staging chunk (= 32 k_half rows)

// E2M1 values, exact in BF16 (same set as the scalar kernel's E2M1_LUT_T).
__device__ __constant__ float MX4_LUT_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// BF16 2^(s-127) (E8M0 scale; 0 decodes to 0, as does the NaN code 255).
__device__ __forceinline__ __nv_bfloat162 mx4_scale(unsigned int s) {
    const unsigned short bits = (s == 0u || s == 255u) ? 0 : (unsigned short)(s << 7);
    const __nv_bfloat16 v = __ushort_as_bfloat16(bits);
    return __halves2bfloat162(v, v);
}

// One m16n8k16 BF16 MMA step into `acc` (FP32).
__device__ __forceinline__ void mx4_mma(float (&acc)[4], const unsigned int (&a)[4],
                                        unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two FP4 values of one byte -> one BF16 pair, times the E8M0 scale.
// `byte` carries (even K in the low nibble, odd K in the high nibble);
// `lut` maps a nibble to BF16 bits.
__device__ __forceinline__ unsigned int mx4_pair(unsigned int byte, unsigned int scale,
                                                 const unsigned short* lut) {
    __nv_bfloat162 v = __halves2bfloat162(__ushort_as_bfloat16(lut[byte & 0xFu]),
                                          __ushort_as_bfloat16(lut[byte >> 4]));
    v = __hmul2(v, mx4_scale(scale));
    return *reinterpret_cast<unsigned int*>(&v);
}

// Fused gate+up decode/verify GEMV over routed MXFP4 experts (transposed
// weights). Mirrors `moe_expert_gate_up_shared_t_e8m0`'s grid:
//   grid [ceil(N/16), top_k + 1, 2]  (x = N tile, y = expert slot, z = proj)
//   block 256
// A: BF16 [M, K] row-major (M = 1 for single-token decode; lda = K).
// C: gate_out / up_out [expert_slot * N + m * N + n] BF16; the shared slot
//    (expert_slot == top_k) writes sh_*_out[m * N + n] and has no weights
//    (MiMo MoE layers ship no shared expert; a null expert pointer zeroes
//    the tile, matching the scalar kernel).
extern "C" __global__ void __launch_bounds__(MX4_WARPS * MX4_WARP) mxfp4_gate_up_tc(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    __nv_bfloat16* __restrict__ sh_gate_out,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda,
    unsigned int top_k
) {
    atlas_pdl_enter();
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const unsigned int n0 = blockIdx.x * MX4_TILE_N;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed = nullptr;
    const unsigned char* B_scale = nullptr;
    __nv_bfloat16* C = nullptr;
    unsigned long long c_base = 0;
    if (is_shared) {
        C = (proj == 0) ? sh_gate_out : sh_up_out;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            C = up_out;
        }
        c_base = (unsigned long long)expert_slot * N;
    }

    // Per-warp staged tiles: s_pk[w][row * 32 + kh] = FP4 byte for weight
    // row (n0 + row), K values chunk*64 + 2*kh .. +1; s_sc[w][block * 16 +
    // row] = E8M0 scale byte for K block (chunk*2 + block).
    __shared__ unsigned char s_pk[MX4_WARPS][MX4_TILE_N * (MX4_CHUNK / 2)];
    __shared__ unsigned char s_sc[MX4_WARPS][2 * MX4_TILE_N];
    __shared__ unsigned short s_lut[16];
    __shared__ float s_red[MX4_WARPS][MX4_WARP][4];

    const unsigned int warp = threadIdx.x / MX4_WARP;
    const unsigned int lane = threadIdx.x % MX4_WARP;

    if (is_shared || B_packed == nullptr) {
        // No weights for this slot: zero the tile (scalar-kernel behavior).
        for (unsigned int m = 0; m < M; m++)
            for (unsigned int n = n0; n < min(n0 + MX4_TILE_N, N); n++)
                C[c_base + (unsigned long long)m * N + n] = __float2bfloat16(0.0f);
        return;
    }

    if (threadIdx.x < 16)
        s_lut[threadIdx.x] = __bfloat16_as_ushort(__float2bfloat16(MX4_LUT_T[threadIdx.x]));
    __syncthreads();

    const unsigned int g = lane >> 2, c = lane & 3u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int chunks = K / MX4_CHUNK;
    unsigned char* pk = s_pk[warp];
    unsigned char* sc = s_sc[warp];
    for (unsigned int ch = warp; ch < chunks; ch += MX4_WARPS) {
        // Stage this warp's chunk: lane kh loads the 16 n-bytes of k_half row
        // ch*32 + lane (one uint4, coalesced in the transposed layout) and
        // scatters to n-major smem; lanes 0..1 stage the two scale blocks.
        {
            const uint4 v = *(const uint4*)(B_packed +
                (unsigned long long)(ch * (MX4_CHUNK / 2) + lane) * N + n0);
            const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
            #pragma unroll
            for (unsigned int n = 0; n < MX4_TILE_N; n++)
                pk[n * (MX4_CHUNK / 2) + lane] = b[n];
        }
        if (lane < 2) {
            const uint4 v = *(const uint4*)(B_scale +
                (unsigned long long)(ch * 2u + lane) * N + n0);
            const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
            #pragma unroll
            for (unsigned int n = 0; n < MX4_TILE_N; n++)
                sc[lane * MX4_TILE_N + n] = b[n];
        }
        __syncwarp();

        // Lane (g, c) covers K values ch*64 + 16c .. +15 of the chunk.
        const unsigned int kb = ch * MX4_CHUNK + 16u * c;
        uint4 xa[2];
        xa[0] = xa[1] = make_uint4(0u, 0u, 0u, 0u);
        const unsigned int row = g;  // activation row for tile 0 (NT = 1)
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * lda + kb);
            xa[0] = xp[0];
            xa[1] = xp[1];
        }
        const unsigned int xw[8] = {xa[0].x, xa[0].y, xa[0].z, xa[0].w,
                                    xa[1].x, xa[1].y, xa[1].z, xa[1].w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Run values 4j..4j+3 = smem bytes 2j, 2j+1 of the 8-byte run at
            // row*32 + 8c; weight rows g and g+8 of the 16-row tile.
            const unsigned short ug = *(const unsigned short*)(pk + g * (MX4_CHUNK / 2) + 8u * c + 2u * j);
            const unsigned short uh = *(const unsigned short*)(pk + (g + 8u) * (MX4_CHUNK / 2) + 8u * c + 2u * j);
            const unsigned int sc_g = sc[(c >> 1) * MX4_TILE_N + g];
            const unsigned int sc_h = sc[(c >> 1) * MX4_TILE_N + g + 8u];
            unsigned int a[4] = {
                mx4_pair(ug & 0xFFu, sc_g, s_lut),
                mx4_pair(uh & 0xFFu, sc_h, s_lut),
                mx4_pair(ug >> 8, sc_g, s_lut),
                mx4_pair(uh >> 8, sc_h, s_lut),
            };
            mx4_mma(acc, a, xw[2 * j], xw[2 * j + 1]);
        }
    }

    // K-split reduction across warps (same order as mxfp8_gemv_tc).
    #pragma unroll
    for (int q = 0; q < 4; q++) s_red[warp][lane][q] = acc[q];
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < MX4_WARPS; w++) v += s_red[w][lane][q];
        const unsigned int n = (q < 2) ? (n0 + g) : (n0 + g + 8u);
        const unsigned int m = 2u * c + (q & 1u);
        if (n < N && m < M)
            C[c_base + (unsigned long long)m * N + n] = __float2bfloat16(v);
    }
}
