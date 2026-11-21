// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 routed-expert GEMVs for decode/verify rows, read straight from the
// checkpoint's stacked expert tensors. Reference: transformers PR #51050
// `Inkling2Experts`, `Inkling2SoftcappedSwiGLU` and `Inkling2LoopDequantizeExperts`
// (Apache-2.0).
//
// Formats (per layer):
//   NVFP4: codes U8 [E, N, K/2] (even element in the low nibble), E4M3 scales
//          [E, N, K/16], per-expert global s2[e] = amax[e] / outer_scale_denominator.
//   MXFP8: E4M3 values [E, N, K], E8M0 scales U8 [E, N, K/32] (x 2^(s - 127)).
// w13 rows interleave gate and up (row 2i = gate i, row 2i + 1 = up i), so one warp
// produces both halves of an activation and applies the softcapped SwiGLU
//   act = tanh(g / cg) * cg * sigmoid(g) * tanh(u / cu) * cu
// with g and u rounded to BF16 first and act rounded to BF16 after, as the reference.
//
// inkling2_moe_gate_up_*: A [rows, K] -> act [rows, top_k, I]; grid (ceil(I/8), top_k, rows), 256 threads
// inkling2_moe_down_*:    act -> y [rows, top_k, H] (BF16);    grid (ceil(H/32), top_k, rows), 256 threads
// Weight rows are read in 16-byte chunks (code rows and tensor bases 16-byte aligned) against the
// activation row, which every warp of a block reads (from cache), so a lane keeps whole chunks of
// weights in flight; E2M1 codes are decoded by cvt.rn.f16x2.e2m1x2 (exact).
// inkling2_moe_combine:   out[r, h] = BF16(sum_s y[r, s, h] * w[r, s]) in fp32.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define INK_GU_WARPS 8   // gate_up: one warp per activation (half-warps on gate and up rows)
#define INK_DN_ROWS 32   // down: 8 warps x 4 rows, 8 lanes per row
#define INK_MAX_SROW 256 // scale bytes of one weight row (K up to 4096 NVFP4 / 8192 MXFP8)

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// Standard E4M3 (1-4-3, bias 7) by bit math: the AMD compat __nv_fp8_e4m3 -> float
// decode is not the standard format (see common/dense_gemv_fp8w.cu scl_fp8).
__device__ __forceinline__ float ink_e4m3(unsigned char b) {
    const unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u) v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float ink_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}
#endif

__device__ __forceinline__ float ink_bf(unsigned int raw, int hi) {
    return __bfloat162float(__ushort_as_bfloat16((unsigned short)(hi ? raw >> 16 : raw & 0xFFFF)));
}

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip): no cvt.rn.f16x2.e2m1x2 / ld.global.nc.L1::no_allocate. E2M1 by bit math
// (exact: sign, 2-bit exponent, 1-bit mantissa; e = 0 is the 0 / 0.5 subnormal pair) and a
// plain 16-byte load.
__device__ __forceinline__ float ink_e2m1(unsigned int n) {
    const unsigned int e = (n >> 1) & 3u, m = n & 1u;
    const float v = e == 0u ? 0.5f * (float)m : __uint_as_float(((e + 126u) << 23) | (m << 22));
    return (n & 8u) ? -v : v;
}
__device__ __forceinline__ float ink_fp4x8(unsigned int w, uint4 av) {
    const unsigned int a[4] = {av.x, av.y, av.z, av.w};
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const unsigned int byte = (w >> (8 * b)) & 0xFFu;
        acc += ink_e2m1(byte & 0xFu) * ink_bf(a[b], 0) + ink_e2m1(byte >> 4) * ink_bf(a[b], 1);
    }
    return acc;
}
__device__ __forceinline__ uint4 ink_ld_stream(const unsigned char* p) { return *(const uint4*)p; }
// hipcc has no __syncwarp (see common/w4a16_gemv_fused.cu); the scale rows are shared by lanes
// of one wave, so a wave barrier with LDS ordering stands in for it.
__device__ __forceinline__ void ink_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#define __syncwarp() ink_wave_sync()
#else
// sum over 8 NVFP4 values (one code word, even element in each byte's low nibble) times
// 8 BF16 activations (one uint4); the E2M1 values come exact from the hardware converter.
__device__ __forceinline__ float ink_fp4x8(unsigned int w, uint4 av) {
    unsigned int h[4];
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(w));
    const unsigned int a[4] = {av.x, av.y, av.z, av.w};
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const float2 v = __half22float2(*(const __half2*)&h[b]);
        acc += v.x * ink_bf(a[b], 0) + v.y * ink_bf(a[b], 1);
    }
    return acc;
}

// A weight chunk, streamed past L1 (which keeps the activation row every warp re-reads).
__device__ __forceinline__ uint4 ink_ld_stream(const unsigned char* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

#endif  // AMD

// sum over 4 E4M3 values (one code word) times 4 BF16 activations (two words)
__device__ __forceinline__ float ink_fp8x4(unsigned int w, unsigned int a0, unsigned int a1) {
    return ink_e4m3(w & 0xFF) * ink_bf(a0, 0) + ink_e4m3((w >> 8) & 0xFF) * ink_bf(a0, 1)
         + ink_e4m3((w >> 16) & 0xFF) * ink_bf(a1, 0) + ink_e4m3(w >> 24) * ink_bf(a1, 1);
}

// This lane's share of dot(a[0:K], W[row, 0:K]): 16-byte code chunks (32 NVFP4 or 16
// MXFP8 values) strided over the L lanes of its group; `a` in global or shared memory.
// The row's scales are first copied into the group's `srow` in 4-byte words (one wide
// request for the row instead of a 2-byte load per chunk, which cost a quarter of the
// kernel's time); scale rows and bases must be 4-byte aligned.
template <bool MX, int L>
__device__ __forceinline__ float ink_dot(const unsigned char* __restrict__ codes,
                                         const unsigned char* __restrict__ scales, float s2,
                                         const __nv_bfloat16* a, unsigned int K, unsigned int g,
                                         unsigned char* srow) {
    constexpr unsigned int V = MX ? 16 : 32;
    for (unsigned int w = g; w < K / (MX ? 32 : 16) / 4; w += L)
        ((unsigned int*)srow)[w] = ((const unsigned int*)scales)[w];
    __syncwarp();
    float acc = 0.0f;
    // unrolled so a lane's code chunks are in flight together (same summation order)
#pragma unroll 4
    for (unsigned int c = g; c < K / V; c += L) {
        const uint4 cv = ink_ld_stream(codes + c * 16);
        const uint4* ap = (const uint4*)(a + c * V);
        if (MX) {
            const uint4 a0 = ap[0], a1 = ap[1];
            const float p = ink_fp8x4(cv.x, a0.x, a0.y) + ink_fp8x4(cv.y, a0.z, a0.w)
                          + ink_fp8x4(cv.z, a1.x, a1.y) + ink_fp8x4(cv.w, a1.z, a1.w);
            acc += p * exp2f((float)srow[c / 2] - 127.0f);
        } else {
            const unsigned short sv = *(const unsigned short*)(srow + c * 2);
            const float lo = ink_fp4x8(cv.x, ap[0]) + ink_fp4x8(cv.y, ap[1]);
            const float hi = ink_fp4x8(cv.z, ap[2]) + ink_fp4x8(cv.w, ap[3]);
            acc += lo * (ink_e4m3(sv & 0xFF) * s2) + hi * (ink_e4m3(sv >> 8) * s2);
        }
    }
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    return acc;
}

template <bool MX>
__device__ __forceinline__ void ink_gate_up(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
                                            const unsigned char* __restrict__ scales, const float* __restrict__ s2,
                                            const int* __restrict__ experts, __nv_bfloat16* __restrict__ act,
                                            unsigned int I, unsigned int K, unsigned int top_k, float cg, float cu) {
    const unsigned int slot = blockIdx.y, r = blockIdx.z;
    // every warp reads the same activation row: the cache serves it
    const __nv_bfloat16* a = A + (unsigned long long)r * K;
    const unsigned int i = blockIdx.x * INK_GU_WARPS + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (i >= I) return;
    const unsigned long long e = (unsigned long long)experts[r * top_k + slot];
    // half-warp 0 on the gate row 2i, half-warp 1 on the up row 2i + 1
    const unsigned long long row = e * 2ull * I + 2ull * i + lane / 16;
    const unsigned char* c = codes + row * (MX ? K : K / 2);
    const unsigned char* sc = scales + row * (K / (MX ? 32 : 16));
    __shared__ __align__(16) unsigned char srow[2 * INK_GU_WARPS][INK_MAX_SROW];
    const float d = ink_dot<MX, 16>(c, sc, MX ? 1.0f : s2[e], a, K, lane % 16, srow[threadIdx.x / 16]);
    const float other = __shfl_sync(0xffffffff, d, 16);
    if (lane == 0) {
        const float g = __bfloat162float(__float2bfloat16(d));
        const float u = __bfloat162float(__float2bfloat16(other));
        const float v = tanhf(g / cg) * cg * (1.0f / (1.0f + __expf(-g))) * tanhf(u / cu) * cu;
        act[((unsigned long long)r * top_k + slot) * I + i] = __float2bfloat16(v);
    }
}

template <bool MX>
__device__ __forceinline__ void ink_down(const __nv_bfloat16* __restrict__ act, const unsigned char* __restrict__ codes,
                                         const unsigned char* __restrict__ scales, const float* __restrict__ s2,
                                         const int* __restrict__ experts, __nv_bfloat16* __restrict__ y,
                                         unsigned int H, unsigned int I, unsigned int top_k) {
    const unsigned int slot = blockIdx.y, r = blockIdx.z;
    const __nv_bfloat16* a = act + ((unsigned long long)r * top_k + slot) * I;
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int n = blockIdx.x * INK_DN_ROWS + (threadIdx.x / 32) * 4 + lane / 8;
    const unsigned long long e = (unsigned long long)experts[r * top_k + slot];
    const unsigned long long row = e * H + (n < H ? n : 0);
    const unsigned char* c = codes + row * (MX ? I : I / 2);
    const unsigned char* sc = scales + row * (I / (MX ? 32 : 16));
    __shared__ __align__(16) unsigned char srow[INK_DN_ROWS][INK_MAX_SROW];
    const float d = ink_dot<MX, 8>(c, sc, MX ? 1.0f : s2[e], a, I, lane % 8, srow[threadIdx.x / 8]);
    if (lane % 8 == 0 && n < H) y[((unsigned long long)r * top_k + slot) * H + n] = __float2bfloat16(d);
}

#define INK_GATE_UP_ARGS                                                                                   \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ codes,                         \
        const unsigned char *__restrict__ scales, const float *__restrict__ s2, const int *__restrict__ experts, \
        __nv_bfloat16 *__restrict__ act, unsigned int I, unsigned int K, unsigned int top_k, float cg, float cu
#define INK_DOWN_ARGS                                                                                      \
    const __nv_bfloat16 *__restrict__ act, const unsigned char *__restrict__ codes,                       \
        const unsigned char *__restrict__ scales, const float *__restrict__ s2, const int *__restrict__ experts, \
        __nv_bfloat16 *__restrict__ y, unsigned int H, unsigned int I, unsigned int top_k

extern "C" __global__ void inkling2_moe_gate_up_nvfp4(INK_GATE_UP_ARGS) {
    atlas_pdl_enter();
    ink_gate_up<false>(A, codes, scales, s2, experts, act, I, K, top_k, cg, cu);
}
extern "C" __global__ void inkling2_moe_gate_up_mxfp8(INK_GATE_UP_ARGS) {
    atlas_pdl_enter();
    ink_gate_up<true>(A, codes, scales, s2, experts, act, I, K, top_k, cg, cu);
}
extern "C" __global__ void inkling2_moe_down_nvfp4(INK_DOWN_ARGS) {
    atlas_pdl_enter();
    ink_down<false>(act, codes, scales, s2, experts, y, H, I, top_k);
}
extern "C" __global__ void inkling2_moe_down_mxfp8(INK_DOWN_ARGS) {
    atlas_pdl_enter();
    ink_down<true>(act, codes, scales, s2, experts, y, H, I, top_k);
}

// out[r, h] = BF16(sum_s float(y[r, s, h]) * w[r, s]); grid (ceil(H/256), rows)
extern "C" __global__ void inkling2_moe_combine(const __nv_bfloat16* __restrict__ y, const float* __restrict__ w,
                                                __nv_bfloat16* __restrict__ out, unsigned int H, unsigned int top_k,
                                                unsigned int out_stride) {
    atlas_pdl_enter();
    const unsigned int h = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (h >= H) return;
    float acc = 0.0f;
    for (unsigned int s = 0; s < top_k; s++)
        acc += __bfloat162float(y[((unsigned long long)r * top_k + s) * H + h]) * w[r * top_k + s];
    out[(unsigned long long)r * out_stride + h] = __float2bfloat16(acc);
}

// ---------------------------------------------------------------------------------------
// Decode batches of a few rows: the rows' expert slots sorted by expert (moe_sort_by_expert),
// each expert's weights read once and applied to every row that chose it (up to INK_GR rows
// staged per pass), outputs in sorted-slot order for moe_unpermute_reduce_indexed.
// A block stages an expert's rows once for INK_GR_TILES output tiles (the staging was more
// traffic than the weights at one tile a block).
//   _grouped_gate_up: grid (ceil(I/(8*INK_GR_TILES)), E), 256 threads; act[slot, i]
//   _grouped_down:    grid (ceil(H/(32*INK_GR_TILES)), E), 256 threads; y[slot, n]
#define INK_GR 8
#define INK_GR_MAX_K 2304
#define INK_GR_TILES 4

// R dots of one weight row against R staged activation rows, this lane's share.
template <bool MX, int L>
__device__ __forceinline__ void ink_dot_rows(const unsigned char* __restrict__ codes,
                                             const unsigned char* __restrict__ scales, float s2,
                                             const __nv_bfloat16* a, unsigned int K, unsigned int g, int nr,
                                             float* acc, unsigned char* srow) {
    constexpr unsigned int V = MX ? 16 : 32;
    // the row's scales into the group's srow, as in ink_dot (the group's previous row is
    // done with it: its reads came before the reduction's shuffles)
    __syncwarp();
    for (unsigned int w = g; w < K / (MX ? 32 : 16) / 4; w += L)
        ((unsigned int*)srow)[w] = ((const unsigned int*)scales)[w];
    __syncwarp();
#pragma unroll
    for (int r = 0; r < INK_GR; r++) acc[r] = 0.0f;
    for (unsigned int c = g; c < K / V; c += L) {
        const uint4 cv = *(const uint4*)(codes + c * 16);
        float sa = 1.0f, sb = 1.0f;
        if (MX) {
            sa = exp2f((float)srow[c / 2] - 127.0f);
        } else {
            const unsigned short sv = *(const unsigned short*)(srow + c * 2);
            sa = ink_e4m3(sv & 0xFF) * s2;
            sb = ink_e4m3(sv >> 8) * s2;
        }
#pragma unroll
        for (int r = 0; r < INK_GR; r++) {
            if (r >= nr) break;
            const uint4* ap = (const uint4*)(a + (unsigned long long)r * K + c * V);
            if (MX) {
                const uint4 a0 = ap[0], a1 = ap[1];
                acc[r] += sa * (ink_fp8x4(cv.x, a0.x, a0.y) + ink_fp8x4(cv.y, a0.z, a0.w)
                              + ink_fp8x4(cv.z, a1.x, a1.y) + ink_fp8x4(cv.w, a1.z, a1.w));
            } else {
                acc[r] += sa * (ink_fp4x8(cv.x, ap[0]) + ink_fp4x8(cv.y, ap[1]))
                        + sb * (ink_fp4x8(cv.z, ap[2]) + ink_fp4x8(cv.w, ap[3]));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < INK_GR; r++)
#pragma unroll
        for (int off = L / 2; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffff, acc[r], off);
}

// stage rows src_row(s0 + base + r) (r < nr) of X [*, K] into smem a [nr][K]
__device__ __forceinline__ void ink_stage_rows(const __nv_bfloat16* __restrict__ X, const int* __restrict__ rows,
                                               int first, int nr, unsigned int K, __nv_bfloat16* a) {
    const unsigned int chunks = K / 8;
    for (unsigned int i = threadIdx.x; i < nr * chunks; i += blockDim.x) {
        const int r = i / chunks;
        const unsigned long long src = rows ? (unsigned long long)rows[first + r] : (unsigned long long)(first + r);
        ((uint4*)(a + (unsigned long long)r * K))[i % chunks] = ((const uint4*)(X + src * K))[i % chunks];
    }
}

template <bool MX>
__device__ __forceinline__ void ink_grouped_gate_up(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
                                                    const unsigned char* __restrict__ scales, const float* __restrict__ s2,
                                                    const int* __restrict__ rows, const int* __restrict__ offsets,
                                                    __nv_bfloat16* __restrict__ act, unsigned int I, unsigned int K,
                                                    float cg, float cu) {
    const unsigned int e = blockIdx.y;
    const int s0 = offsets[e], cnt = offsets[e + 1] - s0;
    if (cnt <= 0) return;
    __shared__ __align__(16) __nv_bfloat16 a[INK_GR * INK_GR_MAX_K];
    __shared__ __align__(16) unsigned char srow[2 * INK_GU_WARPS][INK_MAX_SROW];
    const unsigned int lane = threadIdx.x % 32;
    for (int base = 0; base < cnt; base += INK_GR) {
        const int nr = min(INK_GR, cnt - base);
        __syncthreads();
        ink_stage_rows(A, rows, s0 + base, nr, K, a);
        __syncthreads();
        for (unsigned int t = 0; t < INK_GR_TILES; t++) {
            const unsigned int i = (blockIdx.x * INK_GR_TILES + t) * INK_GU_WARPS + threadIdx.x / 32;
            if (i >= I) break;
            const unsigned long long row = (unsigned long long)e * 2ull * I + 2ull * i + lane / 16;
            const unsigned char* c = codes + row * (MX ? K : K / 2);
            const unsigned char* sc = scales + row * (K / (MX ? 32 : 16));
            float d[INK_GR];
            ink_dot_rows<MX, 16>(c, sc, MX ? 1.0f : s2[e], a, K, lane % 16, nr, d, srow[threadIdx.x / 16]);
#pragma unroll
            for (int r = 0; r < INK_GR; r++) {
                if (r >= nr) break;
                const float other = __shfl_sync(0xffffffff, d[r], 16);
                if (lane == 0) {
                    const float gg = __bfloat162float(__float2bfloat16(d[r]));
                    const float u = __bfloat162float(__float2bfloat16(other));
                    const float v = tanhf(gg / cg) * cg * (1.0f / (1.0f + __expf(-gg))) * tanhf(u / cu) * cu;
                    act[(unsigned long long)(s0 + base + r) * I + i] = __float2bfloat16(v);
                }
            }
        }
    }
}

template <bool MX>
__device__ __forceinline__ void ink_grouped_down(const __nv_bfloat16* __restrict__ act, const unsigned char* __restrict__ codes,
                                                 const unsigned char* __restrict__ scales, const float* __restrict__ s2,
                                                 const int* __restrict__ offsets, __nv_bfloat16* __restrict__ y,
                                                 unsigned int H, unsigned int I) {
    const unsigned int e = blockIdx.y;
    const int s0 = offsets[e], cnt = offsets[e + 1] - s0;
    if (cnt <= 0) return;
    __shared__ __align__(16) __nv_bfloat16 a[INK_GR * INK_GR_MAX_K];
    __shared__ __align__(16) unsigned char srow[INK_DN_ROWS][INK_MAX_SROW];
    const unsigned int lane = threadIdx.x % 32;
    for (int base = 0; base < cnt; base += INK_GR) {
        const int nr = min(INK_GR, cnt - base);
        __syncthreads();
        ink_stage_rows(act, nullptr, s0 + base, nr, I, a);
        __syncthreads();
        for (unsigned int t = 0; t < INK_GR_TILES; t++) {
            const unsigned int n = (blockIdx.x * INK_GR_TILES + t) * INK_DN_ROWS + (threadIdx.x / 32) * 4 + lane / 8;
            if ((blockIdx.x * INK_GR_TILES + t) * INK_DN_ROWS >= H) break;
            const unsigned long long row = (unsigned long long)e * H + (n < H ? n : 0);
            const unsigned char* c = codes + row * (MX ? I : I / 2);
            const unsigned char* sc = scales + row * (I / (MX ? 32 : 16));
            float d[INK_GR];
            ink_dot_rows<MX, 8>(c, sc, MX ? 1.0f : s2[e], a, I, lane % 8, nr, d, srow[threadIdx.x / 8]);
            if (lane % 8 == 0 && n < H) {
#pragma unroll
                for (int r = 0; r < INK_GR; r++) {
                    if (r >= nr) break;
                    y[(unsigned long long)(s0 + base + r) * H + n] = __float2bfloat16(d[r]);
                }
            }
        }
    }
}

#define INK_GGU_ARGS                                                                                         \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ codes,                           \
        const unsigned char *__restrict__ scales, const float *__restrict__ s2, const int *__restrict__ rows, \
        const int *__restrict__ offsets, __nv_bfloat16 *__restrict__ act, unsigned int I, unsigned int K,     \
        float cg, float cu
#define INK_GDN_ARGS                                                                                         \
    const __nv_bfloat16 *__restrict__ act, const unsigned char *__restrict__ codes,                         \
        const unsigned char *__restrict__ scales, const float *__restrict__ s2, const int *__restrict__ offsets, \
        __nv_bfloat16 *__restrict__ y, unsigned int H, unsigned int I

extern "C" __global__ void inkling2_moe_grouped_gate_up_nvfp4(INK_GGU_ARGS) {
    atlas_pdl_enter();
    ink_grouped_gate_up<false>(A, codes, scales, s2, rows, offsets, act, I, K, cg, cu);
}
extern "C" __global__ void inkling2_moe_grouped_gate_up_mxfp8(INK_GGU_ARGS) {
    atlas_pdl_enter();
    ink_grouped_gate_up<true>(A, codes, scales, s2, rows, offsets, act, I, K, cg, cu);
}
extern "C" __global__ void inkling2_moe_grouped_down_nvfp4(INK_GDN_ARGS) {
    atlas_pdl_enter();
    ink_grouped_down<false>(act, codes, scales, s2, offsets, y, H, I);
}
extern "C" __global__ void inkling2_moe_grouped_down_mxfp8(INK_GDN_ARGS) {
    atlas_pdl_enter();
    ink_grouped_down<true>(act, codes, scales, s2, offsets, y, H, I);
}
