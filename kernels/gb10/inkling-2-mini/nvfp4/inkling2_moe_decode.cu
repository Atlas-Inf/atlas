// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 routed-expert GEMVs for decode/verify rows, read straight from the
// checkpoint's stacked expert tensors. Reference: transformers PR #51050
// `Inkling2Experts`, `Inkling2SoftcappedSwiGLU` and `Inkling2LoopDequantizeExperts`
// (Apache-2.0); the GEMV structure follows common/moe_expert_gemv_fused.cu.
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
// inkling2_moe_gate_up_*: A [rows, K] -> act [rows, top_k, I]; grid (ceil(I/4), top_k, rows)
// inkling2_moe_down_*:    act -> y [rows, top_k, H] (BF16);    grid (ceil(H/4), top_k, rows)
// inkling2_moe_combine:   out[r, h] = BF16(sum_s y[r, s, h] * w[r, s]) in fp32.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define INK_BLOCK 128
#define INK_OUT_PER_BLOCK 4
#define INK_TPO (INK_BLOCK / INK_OUT_PER_BLOCK)   // 32: one warp per output

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

// dot(A[0:K], W[row, 0:K]) for one weight row, lanes striding 8-element chunks.
template <bool MX>
__device__ __forceinline__ float ink_row_dot(const unsigned char* __restrict__ codes,
                                             const unsigned char* __restrict__ scales, float s2,
                                             const __nv_bfloat16* __restrict__ a, unsigned long long row,
                                             unsigned int K, unsigned int lane) {
    float acc = 0.0f;
    for (unsigned int k8 = lane; k8 < K / 8; k8 += INK_TPO) {
        const uint4 av = ((const uint4*)a)[k8];
        const unsigned int ar[4] = {av.x, av.y, av.z, av.w};
        float w[8];
        if (MX) {
            const uint2 cv = *(const uint2*)(codes + row * K + k8 * 8);
            const float s = exp2f((float)scales[row * (K / 32) + (k8 * 8) / 32] - 127.0f);
            const unsigned int cr[2] = {cv.x, cv.y};
#pragma unroll
            for (int i = 0; i < 8; i++) w[i] = ink_e4m3((cr[i / 4] >> (8 * (i % 4))) & 0xFF) * s;
        } else {
            const unsigned int p = *(const unsigned int*)(codes + row * (K / 2) + k8 * 4);
            const float s = ink_e4m3(scales[row * (K / 16) + (k8 * 8) / 16]) * s2;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const unsigned int byte = (p >> (8 * b)) & 0xFF;
                w[2 * b] = INK_E2M1[byte & 0xF] * s;
                w[2 * b + 1] = INK_E2M1[byte >> 4] * s;
            }
        }
#pragma unroll
        for (int b = 0; b < 4; b++) acc += ink_bf(ar[b], 0) * w[2 * b] + ink_bf(ar[b], 1) * w[2 * b + 1];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    return acc;
}

template <bool MX>
__device__ __forceinline__ void ink_gate_up(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ codes,
                                            const unsigned char* __restrict__ scales, const float* __restrict__ s2,
                                            const int* __restrict__ experts, __nv_bfloat16* __restrict__ act,
                                            unsigned int I, unsigned int K, unsigned int top_k, float cg, float cu) {
    const unsigned int slot = blockIdx.y, r = blockIdx.z;
    const unsigned int i = blockIdx.x * INK_OUT_PER_BLOCK + threadIdx.x / INK_TPO;
    const unsigned int lane = threadIdx.x % INK_TPO;
    if (i >= I) return;
    const unsigned long long e = (unsigned long long)experts[r * top_k + slot];
    const unsigned long long rows_per_expert = 2ull * I;
    const unsigned long long row_g = e * rows_per_expert + 2ull * i;
    const __nv_bfloat16* a = A + (unsigned long long)r * K;
    const float sc = MX ? 1.0f : s2[e];
    const float g = __bfloat162float(__float2bfloat16(ink_row_dot<MX>(codes, scales, sc, a, row_g, K, lane)));
    const float u = __bfloat162float(__float2bfloat16(ink_row_dot<MX>(codes, scales, sc, a, row_g + 1, K, lane)));
    if (lane == 0) {
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
    const unsigned int n = blockIdx.x * INK_OUT_PER_BLOCK + threadIdx.x / INK_TPO;
    const unsigned int lane = threadIdx.x % INK_TPO;
    if (n >= H) return;
    const unsigned long long e = (unsigned long long)experts[r * top_k + slot];
    const __nv_bfloat16* a = act + ((unsigned long long)r * top_k + slot) * I;
    const float d = ink_row_dot<MX>(codes, scales, MX ? 1.0f : s2[e], a, e * H + n, I, lane);
    if (lane == 0) y[((unsigned long long)r * top_k + slot) * H + n] = __float2bfloat16(d);
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
