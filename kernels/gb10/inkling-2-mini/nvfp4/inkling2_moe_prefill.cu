// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 routed experts for prefill rows: the pieces around the grouped GEMMs
// (moe_sort_by_expert, inkling2_moe_gemm_{w4a4,nvfp4,mxfp8}, moe_unpermute_reduce_indexed).
// Reference: transformers PR #51050 `Inkling2SoftcappedSwiGLU` (Apache-2.0).
//
// The checkpoint quantizes the inputs of its expert projections
// (hf_quant_config.json `required_activation_quantization`): NVFP4 experts
// take NVFP4 rows -- a per-row outer scale amax / 1536 and, per 16 values, the
// E4M3 block scale scaling the block's max to 6 or to 4, whichever
// reconstructs the block with the smaller absolute error ("fouroversix_mae")
// -- and MXFP8 experts MXFP8 rows (E4M3 values, a round-up power-of-two scale
// per 32). The quantizers below follow vLLM's Inkling 2 kernels
// (vllm/models/inkling2/nvidia/ops/nvfp4.py, mxfp8.py, Apache-2.0) operation
// for operation.

#include "../../common/atlas_pdl.cuh"
#if !(defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__))
#include "../../common/nvfp4_block_quant.cuh"
#endif
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>

// The softcapped SwiGLU of one (gate, up) pair, rounded to BF16 as the reference does.
__device__ __forceinline__ float ink_situ(const float g, const float u, const float cg, const float cu) {
    const float v = tanhf(g / cg) * cg * (1.0f / (1.0f + __expf(-g))) * tanhf(u / cu) * cu;
    return __bfloat162float(__float2bfloat16(v));
}

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip): the activation quantizers are e2m1 / e4m3 cvt PTX. They feed only
// the block-scaled W4A4 / W8A8 prefill GEMMs, which spark-model does not take under
// cfg!(atlas_scale) (the experts take BF16 rows there, as decode does).
extern "C" __global__ void inkling2_quant_fp4() { __builtin_trap(); }
extern "C" __global__ void inkling2_situ_quant_fp4() { __builtin_trap(); }
extern "C" __global__ void inkling2_quant_mxfp8() { __builtin_trap(); }
extern "C" __global__ void inkling2_situ_quant_mxfp8() { __builtin_trap(); }
#else
// x's MXFP8 E4M3 code under the scale of its 32-value block (absolute max `amax`):
// the power of two at or above amax / 448; `e` its UE8M0 exponent.
__device__ __forceinline__ unsigned int ink_mxfp8(const float x, const float amax, unsigned int& e) {
    const float raw = __fmul_rn(amax, 1.0f / 448.0f);
    const int up = ((__float_as_int(raw) + 0x007FFFFF) & 0x7F800000) >> 23;
    e = raw <= __int_as_float(0x00400000) ? 0u : (unsigned int)min(up, 254);
    const float scale = e == 0 ? __int_as_float(0x00400000) : __int_as_float((int)(e << 23));
    const float inv = e == 0 ? __int_as_float(0x7F000000) : __frcp_rn(scale);
    return nvfp4q_e4m3(fminf(fmaxf(__fmul_rn(x, inv), -448.0f), 448.0f));
}

__device__ __forceinline__ float ink_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One NVFP4 row's scales from its absolute max: encoder 1536 / amax, its
// inverse, and the outer scale amax / 1536.
struct InkFp4Row {
    float enc, dec, outer;
};
__device__ __forceinline__ InkFp4Row ink_fp4_row(const float amax) {
    InkFp4Row r;
    r.enc = fminf(__fdiv_rn(1536.0f, amax), FLT_MAX);
    r.dec = __fdiv_rn(1.0f, r.enc);
    r.outer = __fmul_rn(amax, 1.0f / 1536.0f);
    return r;
}

// E2M1 codes of (lo, hi) (lo in the low nibble), round to nearest even and
// saturating, and their values.
__device__ __forceinline__ unsigned int ink_e2m1x2(const float lo, const float hi, float& rlo, float& rhi) {
    unsigned int b, h;
    asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
        : "=r"(b), "=r"(h)
        : "f"(lo), "f"(hi));
    rlo = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
    rhi = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
    return b;
}

// One block of 16 under block scale `s` (inverse `inv`): its 8 code bytes and
// its absolute reconstruction error, summed pairwise.
__device__ __forceinline__ float ink_fp4_candidate(const float (&x)[16], const float s, const float inv,
                                                   const float outer, uint2& packed) {
    float e[8];
    unsigned int w[2] = {0u, 0u};
#pragma unroll
    for (int k = 0; k < 8; k++) {
        float a, b;
        const unsigned int byte = ink_e2m1x2(__fmul_rn(x[2 * k], inv), __fmul_rn(x[2 * k + 1], inv), a, b);
        w[k / 4] |= byte << (8 * (k % 4));
        e[k] = __fadd_rn(fabsf(__fsub_rn(__fmul_rn(__fmul_rn(a, s), outer), x[2 * k])),
                         fabsf(__fsub_rn(__fmul_rn(__fmul_rn(b, s), outer), x[2 * k + 1])));
    }
    packed = make_uint2(w[0], w[1]);
#pragma unroll
    for (int n = 4; n > 0; n >>= 1)
#pragma unroll
        for (int k = 0; k < n; k++) e[k] = __fadd_rn(e[2 * k], e[2 * k + 1]);
    return e[0];
}

// A block of 16 to NVFP4: its code bytes and E4M3 block scale byte.
__device__ __forceinline__ unsigned int ink_fp4_block(const float (&x)[16], const InkFp4Row& r, uint2& packed) {
    float bm = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) bm = fmaxf(bm, fabsf(x[j]));
    const unsigned int b6 = nvfp4q_e4m3(fminf(__fmul_rn(bm, __fmul_rn(r.enc, 1.0f / 6.0f)), 448.0f));
    const unsigned int b4 = nvfp4q_e4m3(fminf(__fmul_rn(bm, __fmul_rn(r.enc, 0.25f)), 448.0f));
    const float s6 = nvfp4q_e4m3_value(b6), s4 = nvfp4q_e4m3_value(b4);
    const float i6 = fminf(__fdiv_rn(1.0f, __fmul_rn(s6, r.dec)), FLT_MAX);
    const float i4 = fminf(__fdiv_rn(1.0f, __fmul_rn(s4, r.dec)), FLT_MAX);
    uint2 p6, p4;
    const float e6 = ink_fp4_candidate(x, s6, i6, r.outer, p6);
    const float e4 = ink_fp4_candidate(x, s4, i4, r.outer, p4);
    const bool four = e4 < e6;
    packed = four ? p4 : p6;
    return four ? b4 : b6;
}

// 16 BF16 at p (32-byte aligned) as floats.
__device__ __forceinline__ void ink_load16(const __nv_bfloat16* p, float (&x)[16]) {
    const uint4 w0 = ((const uint4*)p)[0], w1 = ((const uint4*)p)[1];
    const unsigned int w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int k = 0; k < 8; k++) {
        x[2 * k] = __uint_as_float(w[k] << 16);
        x[2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
    }
}

// The 16 softcapped SwiGLU values of block b of an interleaved (gate, up) row.
__device__ __forceinline__ void ink_situ16(const __nv_bfloat16* gu, const unsigned int b, const float cg,
                                           const float cu, float (&x)[16]) {
    float lo[16], hi[16];  // pairs 0..7 and 8..15 of the block, (gate, up) each
    ink_load16(gu + 32 * b, lo);
    ink_load16(gu + 32 * b + 16, hi);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        x[j] = ink_situ(lo[2 * j], lo[2 * j + 1], cg, cu);
        x[8 + j] = ink_situ(hi[2 * j], hi[2 * j + 1], cg, cu);
    }
}

// Rows of x [rows, K] BF16 to NVFP4: codes q [rows, K/2], E4M3 block scales
// s [rows, K/16], outer scales outer [rows]. A warp a row (two passes: the row
// max, then the blocks). grid (ceil(rows / 8)), block 256.
extern "C" __global__ void __launch_bounds__(256) inkling2_quant_fp4(
    const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    float* __restrict__ outer, const unsigned int rows, const unsigned int K) {
    atlas_pdl_enter();
    const unsigned int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x + (unsigned long long)row * K;
    float amax = 0.0f, v[16];
    for (unsigned int b = lane; b < K / 16; b += 32) {
        ink_load16(xr + 16 * b, v);
#pragma unroll
        for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
    }
    const InkFp4Row r = ink_fp4_row(ink_warp_max(amax));
    if (lane == 0) outer[row] = r.outer;
    for (unsigned int b = lane; b < K / 16; b += 32) {
        ink_load16(xr + 16 * b, v);
        uint2 p;
        s[(unsigned long long)row * (K / 16) + b] = (unsigned char)ink_fp4_block(v, r, p);
        ((uint2*)(q + (unsigned long long)row * (K / 2)))[b] = p;
    }
}

// The softcapped SwiGLU of interleaved gate_up rows gu [rows, 2 I], to NVFP4 as
// inkling2_quant_fp4 does (q [rows, I/2], s [rows, I/16], outer [rows]); a lane holds
// its blocks' values across the row max (I / 16 at most 32 * INK_SITU_BLOCKS).
// grid (ceil(rows / 8)), block 256.
#define INK_SITU_BLOCKS 3
extern "C" __global__ void __launch_bounds__(256) inkling2_situ_quant_fp4(
    const __nv_bfloat16* __restrict__ gu, unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    float* __restrict__ outer, const unsigned int rows, const unsigned int I, const float cg, const float cu) {
    atlas_pdl_enter();
    const unsigned int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const __nv_bfloat16* gr = gu + (unsigned long long)row * 2 * I;
    float amax = 0.0f, v[INK_SITU_BLOCKS][16];
#pragma unroll
    for (unsigned int i = 0; i < INK_SITU_BLOCKS; i++) {
        const unsigned int b = lane + 32 * i;
        if (b >= I / 16) break;
        ink_situ16(gr, b, cg, cu, v[i]);
#pragma unroll
        for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[i][j]));
    }
    const InkFp4Row r = ink_fp4_row(ink_warp_max(amax));
    if (lane == 0) outer[row] = r.outer;
#pragma unroll
    for (unsigned int i = 0; i < INK_SITU_BLOCKS; i++) {
        const unsigned int b = lane + 32 * i;
        if (b >= I / 16) break;
        uint2 p;
        s[(unsigned long long)row * (I / 16) + b] = (unsigned char)ink_fp4_block(v[i], r, p);
        ((uint2*)(q + (unsigned long long)row * (I / 2)))[b] = p;
    }
}

// One value of a 32-value block (the calling warp) to MXFP8: its code into q[i],
// the block's exponent into s[i / 32].
__device__ __forceinline__ void ink_store_mxfp8(const float v, unsigned char* __restrict__ q,
                                                unsigned char* __restrict__ s, const unsigned long long i) {
    unsigned int e;
    q[i] = (unsigned char)ink_mxfp8(v, ink_warp_max(fabsf(v)), e);
    if (i % 32 == 0) s[i / 32] = (unsigned char)e;
}

// x [n] BF16 (rows of a multiple of 32) to MXFP8: E4M3 codes q [n], UE8M0 scales
// s [n / 32]. A warp a block. grid (ceil(n / 256)), block 256.
extern "C" __global__ void inkling2_quant_mxfp8(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                                                unsigned char* __restrict__ s, const unsigned long long n) {
    atlas_pdl_enter();
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ink_store_mxfp8(__bfloat162float(x[i]), q, s, i);
}

// The softcapped SwiGLU of interleaved gate_up rows gu [rows, 2 I] to MXFP8 (q [rows, I],
// s [rows, I / 32]); I a multiple of 32. grid (rows, ceil(I / 256)), block 256.
extern "C" __global__ void inkling2_situ_quant_mxfp8(const __nv_bfloat16* __restrict__ gu,
                                                     unsigned char* __restrict__ q, unsigned char* __restrict__ s,
                                                     const unsigned int I, const float cg, const float cu) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = blockIdx.x;
    if (i >= I) return;
    const __nv_bfloat162 p = ((const __nv_bfloat162*)(gu + r * 2 * I))[i];
    ink_store_mxfp8(ink_situ(__low2float(p), __high2float(p), cg, cu), q, s, r * I + i);
}

#endif  // AMD

// act[r, i] = BF16(tanh(g / cg) * cg * sigmoid(g) * tanh(u / cu) * cu) with g = gu[r, 2i],
// u = gu[r, 2i + 1] (the checkpoint's interleaved w13 rows, so the grouped GEMM output is
// interleaved too), both already BF16 as the reference rounds them.
// grid (rows, ceil(I / 256)), block 256
extern "C" __global__ void inkling2_softcap_swiglu(const __nv_bfloat16* __restrict__ gu,
                                                   __nv_bfloat16* __restrict__ act, const unsigned int I,
                                                   const float cg, const float cu) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long r = blockIdx.x;
    if (i >= I) return;
    const __nv_bfloat162 p = ((const __nv_bfloat162*)(gu + r * 2 * I))[i];
    act[r * I + i] = __float2bfloat16(ink_situ(__low2float(p), __high2float(p), cg, cu));
}

// Load time: an overlay's gate_up rows in transformers' order (gate rows 0..half-1, then
// up rows) into the checkpoint's interleaved order (row 2i gate i, row 2i + 1 up i), per
// expert. src/dst [E, 2 * half, row_bytes]. grid (E * 2 * half), block 128.
extern "C" __global__ void inkling2_interleave_rows(const unsigned char* __restrict__ src,
                                                    unsigned char* __restrict__ dst, const unsigned int half,
                                                    const unsigned int row_bytes) {
    const unsigned long long d = blockIdx.x;
    const unsigned long long n = 2ull * half;
    const unsigned long long e = d / n, r = d % n;
    const unsigned long long s = e * n + (r % 2 == 0 ? r / 2 : half + r / 2);
    for (unsigned int b = threadIdx.x; b < row_bytes; b += blockDim.x) dst[d * row_bytes + b] = src[s * row_bytes + b];
}
