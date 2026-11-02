// SPDX-License-Identifier: AGPL-3.0-only

// Qwen3.8-Flash-Next: the glm-5.3-flash `mxfp8_gemv.cu` kernels verbatim.
// The FP4-Local checkpoint (nvidia/Qwen3.8-Flash-Next-FP4-Local) ships the
// GDN, QSA attention and shared-expert projections as MXFP8, E4M3 [n, k] +
// UE8M0 [n, k/32], which is exactly the row-major layout these tiers read, so
// decode reads the checkpoint's own bytes (`weight_map::mxfp8_load`).
// Including (not copying) keeps one source for both targets, the precedent
// glm-5.3-flash's `w4a16_gemm.cu` set for deepseek-v4-flash's.
//
// Plus this target's MXFP8 shared expert (`layers/moe/shared_mxfp8.rs`):
//
//   mxfp8_gemv_tc{8,16,32}_swiglu: gate and up of [gate; up] ([2N, K]) and
//     SiLU(gate) * up (`mx_swiglu`) into C [M, N]: decode's gate/up, the
//     plain tiers' sums. Grid: (ceil(N/8),1,1) Block: (256,1,1).
//   mxfp8_swiglu_quant_act: prefill's: the W8A8 gate/up GEMM's BF16 rows
//     [M, 2I] (gate | up) -> `mx_swiglu` -> E4M3 [M, I] + tiled UE8M0, the
//     bytes `mxfp8_quant_act_bf16` makes of the BF16 SiLU rows (the down
//     GEMM's operand). One thread per 8 outputs, a quad per 32-value block.
//     I % 32 == 0. Grid: (ceil(M*I/8 / 256),1,1) Block: 256.
#include "../../glm-5.3-flash/nvfp4/mxfp8_gemv.cu"
#include "../../common/mxfp8_act_quant.cuh"

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_swiglu(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_cta<1, false, false, true>(A, W, S, C, M, N, K, K, out_stride, blockIdx.x);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_swiglu(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_cta<2, false, false, true>(A, W, S, C, M, N, K, K, out_stride, blockIdx.x);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32_swiglu(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_cta<4, false, false, true>(A, W, S, C, M, N, K, K, out_stride, blockIdx.x);
}

extern "C" __global__ void __launch_bounds__(256) mxfp8_swiglu_quant_act(
    const __nv_bfloat16* __restrict__ gu,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf,
    const unsigned int M,
    const unsigned int I,
    const unsigned int inner_pad
) {
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_row = I / 8u;
    const unsigned long long total = (unsigned long long)M * per_row;
    // Lanes past the end still join the quad shuffle (on zeros), then stop.
    const bool live = t < total;
    const unsigned int r = live ? (unsigned int)(t / per_row) : 0u;
    const unsigned int c8 = live ? (unsigned int)(t % per_row) : 0u;
    uint4 graw = make_uint4(0u, 0u, 0u, 0u), uraw = graw;
    if (live) {
        const __nv_bfloat16* row = gu + (size_t)r * 2u * I + (size_t)c8 * 8u;
        graw = *reinterpret_cast<const uint4*>(row);
        uraw = *reinterpret_cast<const uint4*>(row + I);
    }
    const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&graw);
    const __nv_bfloat162* uh = reinterpret_cast<const __nv_bfloat162*>(&uraw);
    float v[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 g = __bfloat1622float2(gh[j]), u = __bfloat1622float2(uh[j]);
        v[2 * j] = __bfloat162float(mx_swiglu(g.x, u.x));
        v[2 * j + 1] = __bfloat162float(mx_swiglu(g.y, u.y));
    }
    uint2 packed;
    const unsigned char s = mx_quant8_quad(v, packed);
    if (!live) return;
    *reinterpret_cast<uint2*>(q + (size_t)r * I + (size_t)c8 * 8u) = packed;
    if ((c8 & 3u) == 0u) {
        const unsigned int blk = c8 / 4u;
        sf[mx_sf_offset(r, blk, inner_pad)] = s;
        if (blk + 1u == I / 32u) mx_sf_zero_pad(sf, r, blk + 1u, inner_pad);
    }
}
