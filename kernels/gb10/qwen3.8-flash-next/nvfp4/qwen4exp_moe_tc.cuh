// SPDX-License-Identifier: AGPL-3.0-only
//
// The tensor-core MoE numerics shared by qwen4exp_moe_c8_tc.cu (verify rows
// and serial decode, ATLAS_QWEN4EXP_MOE_TC) and qwen4exp_moe_tcp.cu (prefill,
// ATLAS_QWEN4EXP_PREFILL_MOE_BF16): BF16 weights lut * dec(scale) (exact),
// BF16 activations, mma.sync m16n8k16 with FP32 accumulation, and ONE k order
// (see qwen4exp_moe_c8_tc.cu) -- so a row's MoE bytes are the same from
// either. Include after qwen4exp_moe_c8.cuh.
#pragma once

__device__ __forceinline__ void tc_mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned tc_bf16x2(float lo, float hi) {
    unsigned d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

// The thread's 8 weights of a k32 block (packed word q, scale byte e) as the
// 4 BF16 pairs (i, i + 4), i < 4: lut * dec(e), exact.
__device__ __forceinline__ void tc_b8(unsigned q, unsigned char e, unsigned (&b)[4]) {
    const float d = c8_dec_e4m3(e) * 16384.0f;  // * 2^14 undoes the f16 trick's scale
#pragma unroll
    for (unsigned s = 0; s < 16; s += 4) {
        const unsigned a = s <= 9 ? q << (9 - s) : q >> (s - 9);
        const unsigned x = (a & 0x0E000E00u) | ((q << (12 - s)) & 0x80008000u);
        const __half2 h = *(const __half2*)&x;
        b[s / 4] = tc_bf16x2(__low2float(h) * d, __high2float(h) * d);
    }
}

// A pairs for one row of a k32 block from its 8 BF16 at k = kb + 8t + i.
__device__ __forceinline__ void tc_a8(uint4 w, unsigned (&a)[4]) {
    a[0] = __byte_perm(w.x, w.z, 0x5410);  // (k0, k4)
    a[1] = __byte_perm(w.x, w.z, 0x7632);  // (k1, k5)
    a[2] = __byte_perm(w.y, w.w, 0x5410);  // (k2, k6)
    a[3] = __byte_perm(w.y, w.w, 0x7632);  // (k3, k7)
}

// One k32 block of an n8 tile: the thread's packed word q and scale byte e
// (row g of the tile), A pairs of rows g and g + 8.
__device__ __forceinline__ void tc_block(float (&acc)[4], unsigned q, unsigned char e,
                                         const unsigned (&a)[4], const unsigned (&a8)[4]) {
    unsigned b[4];
    tc_b8(q, e, b);
    tc_mma(acc, a[0], a8[0], a[1], a8[1], b[0], b[1]);
    tc_mma(acc, a[2], a8[2], a[3], a8[3], b[2], b[3]);
}

// ── B straight to registers, 16 bytes a thread a k128 super-block ──
// Thread t of row g holds real k = kb + 32t + 8j + i (j < 4, i < 8) of a
// super-block: one uint4 of B (word j: k32 MMA pair j), two scale bytes
// (groups 2t, 2t + 1 of the super-block), A rows as 64 contiguous bytes.
// The super-block loop is unrolled so the B loads issue ahead of use.
// Super-blocks [sb0, sb0 + SB) of the tile (all of K: sb0 = 0, SB = K/128).
template <unsigned K, unsigned SB, typename Act>
__device__ __forceinline__ void tc_tile(const unsigned char* __restrict__ B, const unsigned char* __restrict__ Sc,
                                         unsigned lane, float (&acc)[4], Act&& act, unsigned sb0 = 0) {
    constexpr unsigned ROW = K / 2, K16 = K / 16;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned char* b = B + (size_t)g * ROW + 16 * t + 64 * sb0;
    const unsigned char* s = Sc + (size_t)g * K16 + 2 * t + 8 * sb0;
#pragma unroll
    for (unsigned sb = 0; sb < SB; sb++) {
        const uint4 w = *(const uint4*)(b + 64 * sb);
        const unsigned short e = *(const unsigned short*)(s + 8 * sb);
        const unsigned q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            unsigned a[4], a8[4];
            act((sb0 + sb) * 128 + 32 * t + 8 * j, a, a8);
            tc_block(acc, q[j], (unsigned char)(e >> (8 * (j >> 1))), a, a8);
        }
    }
}


// SiLU * up of a gate/up pair already rounded to BF16, as every qwen4_exp
// decode kernel forms it; CLAMP: the routed +-10 SwiGLU clamp (off under
// ATLAS_QWEN4EXP_MOE_NO_CLAMP; the checkpoint declares no limit).
template <bool CLAMP>
__device__ __forceinline__ float tc_silu_up(unsigned short g, unsigned short u, bool routed) {
    float gf = __uint_as_float((unsigned)g << 16);
    float uf = __uint_as_float((unsigned)u << 16);
    const float SWIGLU_LIMIT = 10.0f;
    if (CLAMP && routed) {
        gf = fminf(gf, SWIGLU_LIMIT);
        uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    }
    return (gf / (1.0f + __expf(-gf))) * uf;
}
