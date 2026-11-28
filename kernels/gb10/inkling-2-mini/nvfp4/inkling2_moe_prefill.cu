// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 routed experts for prefill rows: the pieces around the grouped GEMMs
// (moe_sort_by_expert, inkling2_moe_gemm_{w4a4,w8a8}, moe_unpermute_reduce_indexed).
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
#include "../../common/nvfp4_w4a4_tile.cuh"
#endif
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>

// The softcapped SwiGLU of one (gate, up) pair (gate_up rows interleave them, as the
// checkpoint's w13 rows do), rounded to BF16 as the reference does:
//   BF16(tanh(g / cg) * cg * sigmoid(g) * tanh(u / cu) * cu).
__device__ __forceinline__ float ink_situ(const float g, const float u, const float cg, const float cu) {
    const float v = tanhf(g / cg) * cg * (1.0f / (1.0f + __expf(-g))) * tanhf(u / cu) * cu;
    return __bfloat162float(__float2bfloat16(v));
}

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip): the activation quantizers are e2m1 / e4m3 cvt PTX. They feed only
// the block-scaled W4A4 / W8A8 prefill GEMMs, which spark-model does not take under
// cfg!(atlas_scale) (the experts take BF16 rows there; kernels/strix-hip/inkling-2-mini).
extern "C" __global__ void inkling2_quant_fp4() { atlas_pdl_enter(); __builtin_trap(); }
extern "C" __global__ void inkling2_quant_mxfp8() { atlas_pdl_enter(); __builtin_trap(); }
extern "C" __global__ void inkling2_moe_fused_gate_up_mx8() { atlas_pdl_enter(); __builtin_trap(); }
extern "C" __global__ void inkling2_moe_fused_gate_up_fp4() { atlas_pdl_enter(); __builtin_trap(); }
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

// The NVFP4 experts' gate_up GEMM with the softcapped SwiGLU and its NVFP4 quantization
// in the epilogue: what inkling2_moe_gemm_w4a4 into gu followed by a SwiGLU-and-quantize
// pass over gu (inkling2_quant_fp4's quantizer) computed, bit for bit, without gu (a BF16
// [slots, 2 I] round trip: at 8,192 rows the two took 6.93 ms, this 5.31 ms, measured
// 2026-11-26 on one GB10). A row's outer scale needs its maximum over all I SwiGLU values,
// which the N / 128 column tiles of its row tile hold between them: each tile stages its
// 64 values a row (BF16, as gu held them), publishes the rows' partial maxima, counts
// itself in done[row tile], waits for the row tile's other column tiles, then quantizes
// its four blocks of 16 a row.
// Work items w = row tile * (N / 128) + column tile are taken in order from a counter
// (done[row tiles]) by persistent CTAs. The taken items are always a prefix, so a
// waiting tile's partners are either running or will be taken by CTAs that are not
// waiting (at most N / 128 - 1 of a row tile wait); in-order taking also keeps the
// experts' weights in L2 as the hardware's own block order does (round-robin items let
// the CTAs drift apart: 25% slower). done[0 .. row tiles] is zeroed by
// inkling2_moe_tiles; partial [row tiles, N / 128, 128] F32 (tile-major: a slot-major
// [slots, N / 128] made every store and __ldcg strided, 6.4 vs 5.3 ms a visit).
// grid (any, 2 an SM resident), block 256.
#define INK_GU_PITCH 72   // BF16 a staged row: 64 values + 8 (conflict-free MMA-layout stores)
extern "C" __global__ void __launch_bounds__(256, 2) inkling2_moe_fused_gate_up_fp4(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ As, const float* __restrict__ a_outer,
    const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
    const float* __restrict__ s2,
    unsigned char* __restrict__ q, unsigned char* __restrict__ s, float* __restrict__ outer,
    float* __restrict__ partial, int* __restrict__ done,
    const int* __restrict__ offsets, const int* __restrict__ tiles, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K, const float cg, const float cu) {
    atlas_pdl_enter();
    __shared__ __align__(16) W4Smem sm;
    static_assert(sizeof(sm.a) + sizeof(sm.b) >= 128 * INK_GU_PITCH * 2 + 128 * sizeof(InkFp4Row), "staging");
    __nv_bfloat16 (*sw)[INK_GU_PITCH] = (__nv_bfloat16 (*)[INK_GU_PITCH])&sm.a[0][0][0];
    InkFp4Row* rinfo = (InkFp4Row*)((unsigned char*)&sm.a[0][0][0] + 128 * INK_GU_PITCH * 2);
    __shared__ unsigned int next;
    // a volatile read: through `const __restrict__` the count was loaded before the PDL
    // wait, so CTAs started early took the previous visit's count (and work counter)
    const unsigned int nx = N / 128, I = N / 2, row_tiles = (unsigned int)*(const volatile int*)&tiles[0];
    const unsigned int items = row_tiles * nx;
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned int qr = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
    for (;;) {
        __syncthreads();   // every thread has read the previous item's `next`
        if (tid == 0) next = atomicAdd((unsigned int*)&done[row_tiles], 1u);
        __syncthreads();
        const unsigned int w = next;
        if (w >= items) break;
        const unsigned int rt = w / nx, n0 = (w % nx) * 128;
        const unsigned int t = (unsigned int)tiles[1 + rt], e = t >> 16;
        const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
        const unsigned int m0 = (t & 0xFFFFu) * W4_BM, rows_left = min((unsigned int)m_count - m0, (unsigned int)W4_BM);
        w4_rows(sm, rows, (unsigned int)m_start + m0, rows_left);
        const unsigned char* B = codes + (unsigned long long)e * N * (K / 2);
        const unsigned char* S = scales + (unsigned long long)e * N * (K / 16);
        float acc[16][4];
        w4_tile<false>(sm, acc, A, As, K, rows_left, B, S, n0, B, S, n0 + 64);
        __syncthreads();   // the ring is free for the staging
        // gu = BF16(acc * (outer[row] * s2[e])), then the SwiGLU of each (gate, up) pair
        const float g = s2[e];
#pragma unroll
        for (unsigned int h = 0; h < 2; ++h)
#pragma unroll
            for (unsigned int mt = 0; mt < 4; ++mt) {
                const unsigned int lr = wm * 64 + mt * 16 + qr + 8 * h;
                const float rs = __fmul_rn(a_outer[sm.tok[lr]], g);
#pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    const float* a = acc[mt * 4 + j];
                    const __nv_bfloat162 p = __floats2bfloat162_rn(__fmul_rn(a[2 * h], rs), __fmul_rn(a[2 * h + 1], rs));
                    sw[lr][w4_nt(wn, j) * 4 + c] = __float2bfloat16(ink_situ(__low2float(p), __high2float(p), cg, cu));
                }
            }
        __syncthreads();
        // the rows' maxima over this tile's 64 values (rows past the expert's: unused)
        {
            const unsigned int r = tid >> 1, half = tid & 1u;
            float m = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < 32; i++) m = fmaxf(m, fabsf(__bfloat162float(sw[r][32 * half + i])));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            if (half == 0) partial[((unsigned long long)rt * nx + n0 / 128) * 128 + r] = m;
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicAdd(&done[rt], 1);
            while (*(volatile int*)&done[rt] < (int)nx) __nanosleep(32);
            __threadfence();
        }
        __syncthreads();
        if (tid < 128) {
            float amax = 0.0f;
            for (unsigned int x = 0; x < nx; x++) amax = fmaxf(amax, __ldcg(&partial[((unsigned long long)rt * nx + x) * 128 + tid]));
            rinfo[tid] = ink_fp4_row(amax);
            if (n0 == 0 && tid < rows_left) outer[m_start + m0 + tid] = rinfo[tid].outer;
        }
        __syncthreads();
#pragma unroll
        for (unsigned int k = 0; k < 2; k++) {
            const unsigned int item = tid + 256 * k, r = item >> 2, b = item & 3u;
            if (r >= rows_left) continue;
            float v[16];
#pragma unroll
            for (int i = 0; i < 16; i++) v[i] = __bfloat162float(sw[r][16 * b + i]);
            uint2 p;
            const unsigned long long grow = (unsigned long long)m_start + m0 + r;
            s[grow * (I / 16) + n0 / 32 + b] = (unsigned char)ink_fp4_block(v, rinfo[r], p);
            ((uint2*)(q + grow * (I / 2)))[n0 / 32 + b] = p;
        }
        // the next item's w4_rows synchronizes before the staging is overwritten
    }
}

// The MXFP8 experts' gate_up GEMM (inkling2_moe_gemm_w8a8's tile, C = BF16(acc)) with
// the softcapped SwiGLU quantized to MXFP8 in the epilogue (codes q [slots, I], UE8M0
// scales s [slots, I / 32]), as a SwiGLU-and-quantize pass over its BF16 output would,
// bit for bit: a 32-value block is 64 of a tile's 128 columns, so no tile waits for
// another. grid (N / 128, row tiles), block 256.
extern "C" __global__ void __launch_bounds__(256, 2) inkling2_moe_fused_gate_up_mx8(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ As,
    const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
    unsigned char* __restrict__ q, unsigned char* __restrict__ s,
    const int* __restrict__ offsets, const int* __restrict__ tiles, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K, const float cg, const float cu) {
    atlas_pdl_enter();
    __shared__ __align__(16) W4Smem sm;
    __nv_bfloat16 (*sw)[INK_GU_PITCH] = (__nv_bfloat16 (*)[INK_GU_PITCH])&sm.a[0][0][0];
    // volatile: ptxas otherwise schedules this read before the PDL wait (as in
    // inkling2_moe_fused_gate_up_fp4), and the previous visit's count would decide
    if (blockIdx.y >= (unsigned int)*(const volatile int*)&tiles[0]) return;
    const unsigned int t = (unsigned int)tiles[1 + blockIdx.y], e = t >> 16, I = N / 2;
    const int m_start = offsets[e], m_count = offsets[e + 1] - m_start;
    const unsigned int m0 = (t & 0xFFFFu) * W4_BM, n0 = blockIdx.x * 128;
    const unsigned int rows_left = min((unsigned int)m_count - m0, (unsigned int)W4_BM);
    w4_rows(sm, rows, (unsigned int)m_start + m0, rows_left);
    const unsigned long long ne = (unsigned long long)e * N;
    float acc[16][4];
    w4_tile<true>(sm, acc, A, As, K, rows_left, codes + ne * K, scales + ne * (K / 32), n0, codes + ne * K,
                  scales + ne * (K / 32), n0 + 64);
    __syncthreads();   // the ring is free for the staging
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned int qr = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
#pragma unroll
    for (unsigned int h = 0; h < 2; ++h)
#pragma unroll
        for (unsigned int mt = 0; mt < 4; ++mt) {
            const unsigned int lr = wm * 64 + mt * 16 + qr + 8 * h;
#pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                const float* a = acc[mt * 4 + j];
                const __nv_bfloat162 p = __floats2bfloat162_rn(a[2 * h], a[2 * h + 1]);
                sw[lr][w4_nt(wn, j) * 4 + c] = __float2bfloat16(ink_situ(__low2float(p), __high2float(p), cg, cu));
            }
        }
    __syncthreads();
    // a warp a (row, block of 32): 128 rows x 2 blocks over 8 warps
    for (unsigned int rb = warp; rb < 2 * rows_left; rb += 8) {
        const unsigned int r = rb >> 1, b = rb & 1u;
        const unsigned long long i = ((unsigned long long)m_start + m0 + r) * I + n0 / 2 + 32 * b + lane;
        ink_store_mxfp8(__bfloat162float(sw[r][32 * b + lane]), q, s, i);
    }
}

#endif  // AMD

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
