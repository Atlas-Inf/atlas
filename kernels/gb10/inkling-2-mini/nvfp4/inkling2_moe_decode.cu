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
// activation row staged in shared memory, so a lane keeps whole chunks of weights in flight.
// inkling2_moe_combine:   out[r, h] = BF16(sum_s y[r, s, h] * w[r, s]) in fp32.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define INK_MAX_K 4096   // shared activation row (hidden and intermediate sizes)
#define INK_GU_WARPS 8   // gate_up: one warp per activation (half-warps on gate and up rows)
#define INK_DN_ROWS 32   // down: 8 warps x 4 rows, 8 lanes per row

__device__ __constant__ float INK_E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float ink_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}

__device__ __forceinline__ float ink_bf(unsigned int raw, int hi) {
    return __bfloat162float(__ushort_as_bfloat16((unsigned short)(hi ? raw >> 16 : raw & 0xFFFF)));
}

// Byte -> (E2M1 low nibble, E2M1 high nibble), staged in shared memory: lanes index
// the table independently, which __constant__ memory would serialize (~14 transactions
// a read, per the note in common/w4a16_gemv.cu).
__device__ __forceinline__ void ink_stage_lut(float2* lut) {
    for (unsigned int b = threadIdx.x; b < 256; b += blockDim.x) lut[b] = make_float2(INK_E2M1[b & 0xF], INK_E2M1[b >> 4]);
}

// sum over 8 NVFP4 values (one code word) times 8 BF16 activations (one uint4)
__device__ __forceinline__ float ink_fp4x8(const float2* lut, unsigned int w, uint4 av) {
    const unsigned int a[4] = {av.x, av.y, av.z, av.w};
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const float2 v = lut[(w >> (8 * b)) & 0xFF];
        acc += v.x * ink_bf(a[b], 0) + v.y * ink_bf(a[b], 1);
    }
    return acc;
}

// sum over 4 E4M3 values (one code word) times 4 BF16 activations (two words)
__device__ __forceinline__ float ink_fp8x4(unsigned int w, unsigned int a0, unsigned int a1) {
    return ink_e4m3(w & 0xFF) * ink_bf(a0, 0) + ink_e4m3((w >> 8) & 0xFF) * ink_bf(a0, 1)
         + ink_e4m3((w >> 16) & 0xFF) * ink_bf(a1, 0) + ink_e4m3(w >> 24) * ink_bf(a1, 1);
}

// This lane's share of dot(a[0:K], W[row, 0:K]): 16-byte code chunks (32 NVFP4 or 16
// MXFP8 values) strided over the L lanes of its group; `a` is in shared memory.
template <bool MX, int L>
__device__ __forceinline__ float ink_dot(const float2* lut, const unsigned char* __restrict__ codes,
                                         const unsigned char* __restrict__ scales, float s2,
                                         const __nv_bfloat16* a, unsigned int K, unsigned int g) {
    constexpr unsigned int V = MX ? 16 : 32;
    float acc = 0.0f;
    for (unsigned int c = g; c < K / V; c += L) {
        const uint4 cv = *(const uint4*)(codes + c * 16);
        const uint4* ap = (const uint4*)(a + c * V);
        if (MX) {
            const uint4 a0 = ap[0], a1 = ap[1];
            const float p = ink_fp8x4(cv.x, a0.x, a0.y) + ink_fp8x4(cv.y, a0.z, a0.w)
                          + ink_fp8x4(cv.z, a1.x, a1.y) + ink_fp8x4(cv.w, a1.z, a1.w);
            acc += p * exp2f((float)scales[c / 2] - 127.0f);
        } else {
            const unsigned short sv = *(const unsigned short*)(scales + c * 2);
            const float lo = ink_fp4x8(lut, cv.x, ap[0]) + ink_fp4x8(lut, cv.y, ap[1]);
            const float hi = ink_fp4x8(lut, cv.z, ap[2]) + ink_fp4x8(lut, cv.w, ap[3]);
            acc += lo * (ink_e4m3(sv & 0xFF) * s2) + hi * (ink_e4m3(sv >> 8) * s2);
        }
    }
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    return acc;
}

__device__ __forceinline__ void ink_stage(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* dst, unsigned int n) {
    for (unsigned int i = threadIdx.x; i < n / 8; i += blockDim.x) ((uint4*)dst)[i] = ((const uint4*)src)[i];
    __syncthreads();
}

template <bool MX>
__device__ __forceinline__ void ink_gate_up(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
                                            const unsigned char* __restrict__ scales, const float* __restrict__ s2,
                                            const int* __restrict__ experts, __nv_bfloat16* __restrict__ act,
                                            unsigned int I, unsigned int K, unsigned int top_k, float cg, float cu) {
    __shared__ __align__(16) __nv_bfloat16 a[INK_MAX_K];
    __shared__ float2 lut[256];
    const unsigned int slot = blockIdx.y, r = blockIdx.z;
    if (!MX) ink_stage_lut(lut);
    ink_stage(A + (unsigned long long)r * K, a, K);
    const unsigned int i = blockIdx.x * INK_GU_WARPS + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (i >= I) return;
    const unsigned long long e = (unsigned long long)experts[r * top_k + slot];
    // half-warp 0 on the gate row 2i, half-warp 1 on the up row 2i + 1
    const unsigned long long row = e * 2ull * I + 2ull * i + lane / 16;
    const unsigned char* c = codes + row * (MX ? K : K / 2);
    const unsigned char* sc = scales + row * (K / (MX ? 32 : 16));
    const float d = ink_dot<MX, 16>(lut, c, sc, MX ? 1.0f : s2[e], a, K, lane % 16);
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
    __shared__ __align__(16) __nv_bfloat16 a[INK_MAX_K];
    __shared__ float2 lut[256];
    const unsigned int slot = blockIdx.y, r = blockIdx.z;
    if (!MX) ink_stage_lut(lut);
    ink_stage(act + ((unsigned long long)r * top_k + slot) * I, a, I);
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int n = blockIdx.x * INK_DN_ROWS + (threadIdx.x / 32) * 4 + lane / 8;
    const unsigned long long e = (unsigned long long)experts[r * top_k + slot];
    const unsigned long long row = e * H + (n < H ? n : 0);
    const unsigned char* c = codes + row * (MX ? I : I / 2);
    const unsigned char* sc = scales + row * (I / (MX ? 32 : 16));
    const float d = ink_dot<MX, 8>(lut, c, sc, MX ? 1.0f : s2[e], a, I, lane % 8);
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
