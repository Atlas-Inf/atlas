// SPDX-License-Identifier: AGPL-3.0-only

// W4A16 MMA stage of the GLM stream decode twins (the `_a16` kernels of
// glm_moe_decode_stream.cuh, ATLAS_GLM_FP4L_DECODE_A16): BF16 activations
// against the same NVFP4 weight tiles, as the reference engine runs every
// non-prompt row (vLLM b12x W4A16, `B12X_W4A16_A4_PREFILL`: prompt rows A4,
// the rest A16). Included by moe_w4a16_grouped_gemm.cu after
// glm_moe_decode_m16.cuh.
//
// Weights. The stage's weight and scale tiles are the A4 twins' (pqs_load:
// same loads, same shared bytes), and so is the B fragment: pqw_ldsm_t2
// leaves lane (g = lane / 4, c = lane % 4) the 4 packed bytes of column
// 8h + g at packed rows 4c .. 4c + 3 (K values 8c .. 8c + 7 of the k64 slice)
// in b[h], and those of rows 16 + 4c .. (K 32 + 8c ..) in b[2 + h]. Each byte
// dequantizes to a BF16 pair, E2M1 value times the E4M3 block scale, both
// exact in BF16 (a 2-bit by 4-bit significand product), as the engine's
// W4A16 tiers (w4a16_tc_mma) and the reference's dequantized weights.
//
// MMAs. The k64 slice runs as four m16n8k16 BF16 MMAs. The k16 MMA's K slots
// are a permutation of any 16 K values, the same for both operands: MMA j
// takes, in lane c's slots {2c, 2c+1}, K values 8c + 2j + {0, 1} (byte j of
// b[h]) and in slots {2c+8, 2c+9} K values 32 + 8c + 2j + {0, 1} (byte j of
// b[2+h]); the activation fragment takes the same K values of rows g and
// g + 8, word j of each lane's two 16-byte activation runs. Over j = 0..3
// every K value of the slice meets its own weight once. FP32 accumulation;
// the epilogue applies the per-expert global scale as the A4 twins'.
//
// Activations. `A16_PITCH`-byte BF16 rows of one K128 stage (256 bytes and
// 64 of padding: rows g and g + 1 then start 16 banks apart, so the
// quarter-warp phases of the 16-byte activation reads are conflict free).
#pragma once

#define PQS_A16_PITCH 320

typedef unsigned char PqdA16[PQD_M][PQS_A16_PITCH];

// The byte -> BF16 pair table of E2M1 codes (low nibble in .x), one entry
// per thread of a 256-thread block; publish with __syncthreads.
__device__ __forceinline__ void pqd_a16_stage_pairs(unsigned int* s_pair, unsigned int t) {
    if (t < 256) {
        const unsigned int lo = __float_as_uint(E2M1_LUT_MOE[t & 15u]) >> 16;
        const unsigned int hi = __float_as_uint(E2M1_LUT_MOE[t >> 4]) >> 16;
        s_pair[t] = (hi << 16) | lo;
    }
}

__device__ __forceinline__ __nv_bfloat162 pqd_a16_scale(unsigned char byte) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = byte;
    return __float2bfloat162_rn((float)f);
}

// Byte j of `word` as a BF16 pair times `sc`.
__device__ __forceinline__ unsigned int pqd_a16_dequant(
    const unsigned int* s_pair, unsigned int word, int j, __nv_bfloat162 sc
) {
    __nv_bfloat162 v = __hmul2(
        *reinterpret_cast<const __nv_bfloat162*>(&s_pair[(word >> (8 * j)) & 0xFFu]), sc);
    return *reinterpret_cast<unsigned int*>(&v);
}

// Warp warp_id's MMAs of one K128 stage over one 16-row BF16 slab: the A4
// twins' column chunks (pqd_warp_chunk) and accumulator layout.
template<bool GATE_UP>
__device__ __forceinline__ void pqd_mma_stage_a16(
    PqdAcc& acc, const PqdA16& sA, const PqwB& sB, const PqwS& sS, const unsigned int* s_pair,
    unsigned int warp_id, unsigned int lane_id
) {
    const unsigned int g = lane_id >> 2, c = lane_id & 3u;
    #pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        // K values 64 sl + 8c .. + 7 (run 0) and + 32 (run 1) of rows g, g + 8.
        const unsigned int k0 = sl * 64 + 8 * c;
        const uint4 x00 = *(const uint4*)&sA[g][k0 * 2];
        const uint4 x01 = *(const uint4*)&sA[g + 8][k0 * 2];
        const uint4 x10 = *(const uint4*)&sA[g][(k0 + 32) * 2];
        const uint4 x11 = *(const uint4*)&sA[g + 8][(k0 + 32) * 2];
        const unsigned int a00[4] = {x00.x, x00.y, x00.z, x00.w};
        const unsigned int a01[4] = {x01.x, x01.y, x01.z, x01.w};
        const unsigned int a10[4] = {x10.x, x10.y, x10.z, x10.w};
        const unsigned int a11[4] = {x11.x, x11.y, x11.z, x11.w};
        // The block-scale groups of the two runs: K / 16 within the stage.
        const unsigned int g0 = sl * 4 + (c >> 1), g1 = g0 + 2;
        #pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            const unsigned int ch = pqd_warp_chunk<GATE_UP>(warp_id, nb);
            const unsigned int kp = sl * 32 + lane_id;
            unsigned int b[4];
            pqw_ldsm_t2(b, __cvta_generic_to_shared(&sB[kp][(ch ^ (kp & 7)) << 4]));
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nt = nb * 2 + h;
                const unsigned int col = 8 * h + g;
                const __nv_bfloat162 sc0 = pqd_a16_scale(sS[g0][((ch ^ g0) << 4) + col]);
                const __nv_bfloat162 sc1 = pqd_a16_scale(sS[g1][((ch ^ g1) << 4) + col]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned int b0 = pqd_a16_dequant(s_pair, b[h], j, sc0);
                    const unsigned int b1 = pqd_a16_dequant(s_pair, b[2 + h], j, sc1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                        : "r"(a00[j]), "r"(a01[j]), "r"(a10[j]), "r"(a11[j]), "r"(b0), "r"(b1));
                }
            }
        }
    }
}
