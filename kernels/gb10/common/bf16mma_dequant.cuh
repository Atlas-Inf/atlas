// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Shared pieces of the BF16-MMA prefill GEMMs with register dequant
// (moe_bf16mma_prefill.cu, dense_bf16mma_prefill.cu): cp.async staging, the
// exact NVFP4 / FP8 -> BF16 weight decode, and the m16n8k16 BF16 MMA.
//
// Exactness: an e2m1 value times an e4m3 block scale has at most 6
// significant bits and an e4m3 value 4, so both are exact in BF16. Bit
// placement yields e2m1 * 2^-126 and e4m3 * 2^-120 (subnormal for the small
// values); one bf16x2 multiply by a power of two (folded into the NVFP4 block
// scale) makes them normal, exactly. The kernels apply the remaining 2^8 and
// the FP32 tensor scales to the FP32 accumulators.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Staged NVFP4 weight and scale rows are 128 bytes (eight 16-byte chunks);
// chunk c of row r lives at chunk c ^ swz(r) so that the four quad lanes t,
// which read rows 8t + i (weights) or t (scales), hit four different chunks.
__device__ __forceinline__ unsigned int bm_swz_w(unsigned int kp) { return ((kp >> 3) & 3) << 1; }
__device__ __forceinline__ unsigned int bm_swz_s(unsigned int gr) { return (gr & 3) << 1; }

__device__ __forceinline__ void bm_cp16(void* dst, const void* src, bool pred) {
    unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;"
                 :: "r"(d), "l"(src), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void bm_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void bm_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// Four e4m3 block scales -> bf16x2 (both halves) of scale * 2^118. Exact.
__device__ __forceinline__ void bm_scales(unsigned int s4, unsigned int out[4]) {
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)((s4 >> (16 * i)) & 0xFFFF), __NV_E4M3);
        const float2 v = __half22float2(__half2(h));
        const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x * 0x1p118f, v.x * 0x1p118f);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(v.y * 0x1p118f, v.y * 0x1p118f);
        out[2 * i] = *(const unsigned int*)&lo;
        out[2 * i + 1] = *(const unsigned int*)&hi;
    }
}

__device__ __forceinline__ unsigned int bm_mul_bf16x2(unsigned int a, unsigned int s) {
    unsigned int d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(s));
    return d;
}

// Four e2m1 pairs (bytes B0..B3 of w, low nibble = even k) -> four bf16x2
// (low half = even k) of value * 2^-126, byte i times scale s[i].
__device__ __forceinline__ void bm_dequant4(unsigned int w, const unsigned int s[4], unsigned int out[4]) {
    const unsigned int v = w << 4;
    const unsigned int z01 = __byte_perm(w, v, 0x0145);
    const unsigned int z23 = __byte_perm(w, v, 0x2367);
    const unsigned int S = 0x80008000u, M = 0x01C001C0u;
    out[0] = bm_mul_bf16x2((z01 & S) | ((z01 >> 6) & M), s[0]);
    out[1] = bm_mul_bf16x2(((z01 << 8) & S) | ((z01 << 2) & M), s[1]);
    out[2] = bm_mul_bf16x2((z23 & S) | ((z23 >> 6) & M), s[2]);
    out[3] = bm_mul_bf16x2(((z23 << 8) & S) | ((z23 << 2) & M), s[3]);
}

// Two e4m3 bytes of w (bytes 0,1 or, with HI, bytes 2,3; the lower byte is
// the even k) -> bf16x2 of value * 2^-8: sign to bit 15, exponent and
// mantissa to bits 10..4 (value * 2^-120), times 2^112.
template <bool HI>
__device__ __forceinline__ unsigned int bm_fp8_pair(unsigned int w) {
    const unsigned int x = __byte_perm(w, 0, HI ? 0x4342 : 0x4140);
    return bm_mul_bf16x2(((x << 8) & 0x80008000u) | ((x << 4) & 0x07F007F0u), 0x77807780u);
}

__device__ __forceinline__ void bm_mma(float c[4], unsigned int a0, unsigned int a1,
                                       unsigned int a2, unsigned int a3,
                                       unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The four weight bytes (one per MMA row slot) of thread (w, g) at row `r` of
// a 128-byte staged row: FUSED stages [gate 64 | up 64] and gives
// gate[16w + 2g + {0,1}], up[16w + 2g + {0,1}]; otherwise features
// 32w + 4g + {0..3}. Slot order: frag0 row g, frag0 row g+8, frag1 row g,
// frag1 row g+8.
template <bool FUSED>
__device__ __forceinline__ unsigned int bm_slots(const unsigned char* row, unsigned int swz,
                                                 unsigned int warp, unsigned int g) {
    if (FUSED) {
        const unsigned int o = 16 * warp + 2 * g;
        const unsigned int gate = *(const unsigned short*)&row[((o >> 4) ^ swz) * 16 + (o & 15)];
        const unsigned int up = *(const unsigned short*)&row[((4 + (o >> 4)) ^ swz) * 16 + (o & 15)];
        return gate | (up << 16);
    }
    const unsigned int o = 32 * warp + 4 * g;
    return *(const unsigned int*)&row[((o >> 4) ^ swz) * 16 + (o & 15)];
}
