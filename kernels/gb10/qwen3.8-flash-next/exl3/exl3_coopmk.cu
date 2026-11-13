// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 CoopMK — per-expert runtime-K cooperative decode MoE for mixed-K
// packs, reading the packed trellis directly (no reconstruct, no requant).
//
// Device code vendored from vcruz305/exllamav3 @ 047ce72 (9aa5329 CoopMK +
// turboderp 58d4d73 coop kernels) into exl3_vendor/coopmk/ (see
// exl3_vendor/NOTICE.md). This file only instantiates it: Atlas resolves
// kernels by unmangled name (cuModuleGetFunction), so each template instance
// is wrapped in an extern "C" __global__ with the upstream launch bounds.
//
// Instances (mul1 codebook cb = 2, narrow tile WIDE = false — the upstream
// host rule `pick_wide` picks narrow on Blackwell for Flash-Next decode:
// A kslices = Hi/16 = 160 < 256 with 11 slots, B kslices = I/16 = 40):
//
//   exl3_coopmk_{a,b}_all     KMASK = K1..8        MINB 1   (plan 1)
//   exl3_coopmk_{a,b}_reg     KMASK = K2,3,4       MINB 1   (plan 2, default)
//   exl3_coopmk_{a,b}_stg     KMASK = K1,5,6,7,8   MINB 1   (plan 2 / 4)
//   exl3_coopmk_{a,b}_allmb2  KMASK = K1..8        MINB 2   (plan 3)
//   exl3_coopmk_{a,b}_regmb2  KMASK = K2,3,4       MINB 2   (plan 4)
//
// Launch (upstream CoopMK::run, bsz 1): block 512 threads; A grid =
// slots * 2 * (I / 32), dynamic smem Hi*2 + 16384 + 8192; B grid =
// slots * (Ho / 32), dynamic smem 16384 + 8192 + 8192. Args: MoeCoopParams by
// value (344 bytes, layout pinned by static_asserts in moe_coop.cuh and the
// Rust mirror) and the int32 k_tab [K_gate; K_up; K_down] device table.
//
// Plus the Atlas glue kernels (bf16 activations, u32/f32 top-k):
//   exl3_coopmk_prep       x bf16 -> fp16, top-k ids u32 -> int64, weights f32 -> fp16,
//                          shared expert appended as slot top_k with weight sigmoid(x . gate)
//   exl3_coopmk_prep_rows  the same for M rows (one block per row, same arithmetic per row)
//   exl3_coopmk_rot        bsz > 1 rotation pre-kernel + run table (upstream coopmk_rot_kernel)
//   exl3_coopmk_f32_to_bf16

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_vendor/coopmk/moe_coopmk_kernel.cuh"

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_a_all(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, false, exl3_coopmk_ns::KM_ALL, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_b_all(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, false, exl3_coopmk_ns::KM_ALL, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_a_reg(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, false, exl3_coopmk_ns::KM_REG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_b_reg(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, false, exl3_coopmk_ns::KM_REG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_a_stg(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, false, exl3_coopmk_ns::KM_STG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_b_stg(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, false, exl3_coopmk_ns::KM_STG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_a_allmb2(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, false, exl3_coopmk_ns::KM_ALL, 2>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_b_allmb2(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, false, exl3_coopmk_ns::KM_ALL, 2>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_a_regmb2(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, false, exl3_coopmk_ns::KM_REG, 2>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_b_regmb2(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, false, exl3_coopmk_ns::KM_REG, 2>(p, k_tab);
}

// Wide tile (WIDE = true, 128-column block tiles): same variants, suffix _w.
extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_a_all_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, true, exl3_coopmk_ns::KM_ALL, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_b_all_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, true, exl3_coopmk_ns::KM_ALL, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_a_reg_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, true, exl3_coopmk_ns::KM_REG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_b_reg_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, true, exl3_coopmk_ns::KM_REG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_a_stg_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, true, exl3_coopmk_ns::KM_STG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 1) void exl3_coopmk_b_stg_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, true, exl3_coopmk_ns::KM_STG, 1>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_a_allmb2_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, true, exl3_coopmk_ns::KM_ALL, 2>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_b_allmb2_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, true, exl3_coopmk_ns::KM_ALL, 2>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_a_regmb2_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_a_kernel<2, true, exl3_coopmk_ns::KM_REG, 2>(p, k_tab);
}

extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS, 2) void exl3_coopmk_b_regmb2_w(
    const MoeCoopParams p, const int* __restrict__ k_tab)
{
    exl3_coopmk_ns::coopmk_b_kernel<2, true, exl3_coopmk_ns::KM_REG, 2>(p, k_tab);
}

// Routing glue for one token (bsz 1):
//   xh[i]   = fp16(x[i])                                   (grid-stride, all blocks)
//   sel[k]  = idx[k] (u32 -> int64),  rw[k] = fp16(w[k])   for k < topk (block 0)
//   sel[topk] = shared_local, rw[topk] = fp16(sigmoid(x . gate_w))   (block 0)
// The shared expert rides as one more slot of the same A/B launch (its local
// index is the last entry of the pointer tables). gate_w NULL -> weight 1.
// The gate dot is fp32 over bf16 inputs, the same arithmetic as
// moe_weighted_sum_blend's gate phase.
extern "C" __global__ __launch_bounds__(256) void exl3_coopmk_prep(
    const __nv_bfloat16* __restrict__ x,
    half* __restrict__ xh,
    int n,
    const unsigned int* __restrict__ idx,
    const float* __restrict__ w,
    long long* __restrict__ sel,
    half* __restrict__ rw,
    int topk,
    const __nv_bfloat16* __restrict__ gate_w,
    int shared_local)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        xh[i] = __float2half_rn(__bfloat162float(x[i]));
    if (blockIdx.x != 0) return;
    if ((int) threadIdx.x < topk)
    {
        sel[threadIdx.x] = (long long) idx[threadIdx.x];
        rw[threadIdx.x] = __float2half_rn(w[threadIdx.x]);
    }
    __shared__ float s_part[8];
    float dot = 0.0f;
    if (gate_w != nullptr)
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            dot += __bfloat162float(x[i]) * __bfloat162float(gate_w[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if ((threadIdx.x & 31) == 0) s_part[threadIdx.x >> 5] = dot;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float g = 1.0f;
        if (gate_w != nullptr)
        {
            float t = 0.0f;
            for (int k = 0; k < (int) (blockDim.x >> 5); ++k) t += s_part[k];
            g = 1.0f / (1.0f + __expf(-t));
        }
        sel[topk] = (long long) shared_local;
        rw[topk] = __float2half_rn(g);
    }
}

// fp32 [n] -> bf16 [n], grid-stride.
extern "C" __global__ __launch_bounds__(256) void exl3_coopmk_f32_to_bf16(
    const float* __restrict__ in, __nv_bfloat16* __restrict__ out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = __float2bfloat16(in[i]);
}

// bsz > 1 (M verify rows in one A/B pair): the upstream rotation pre-kernel. Grid
// ceil(slots * (Hi / 128) * nproj / 16), block 512; block 0 builds the run table.
extern "C" __global__ __launch_bounds__(MOE_COOP_THREADS) void exl3_coopmk_rot(const MoeCoopParams p)
{
    exl3_coopmk_ns::coopmk_rot_body(p);
}

// Routing glue for M rows: grid (M, 1, 1), block 256, row r = blockIdx.x. Per row the
// arithmetic is exl3_coopmk_prep's block 0 (same element conversions, same fp32 gate dot in the
// same thread/warp order), so every row gets the bits the bsz-1 path gives it.
//   xh[r][i] = fp16(x[r][i]);  sel[r][k] = idx[r][k], rw[r][k] = fp16(w[r][k])  (k < topk)
//   sel[r][topk] = shared_local, rw[r][topk] = fp16(sigmoid(x[r] . gate_w))
extern "C" __global__ __launch_bounds__(256) void exl3_coopmk_prep_rows(
    const __nv_bfloat16* __restrict__ x,
    half* __restrict__ xh,
    int n,
    const unsigned int* __restrict__ idx,
    const float* __restrict__ w,
    long long* __restrict__ sel,
    half* __restrict__ rw,
    int topk,
    const __nv_bfloat16* __restrict__ gate_w,
    int shared_local)
{
    const int r = blockIdx.x;
    const int slots = topk + 1;
    x += (size_t) r * n;
    xh += (size_t) r * n;
    idx += (size_t) r * topk;
    w += (size_t) r * topk;
    sel += (size_t) r * slots;
    rw += (size_t) r * slots;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        xh[i] = __float2half_rn(__bfloat162float(x[i]));
    if ((int) threadIdx.x < topk)
    {
        sel[threadIdx.x] = (long long) idx[threadIdx.x];
        rw[threadIdx.x] = __float2half_rn(w[threadIdx.x]);
    }
    __shared__ float s_part[8];
    float dot = 0.0f;
    if (gate_w != nullptr)
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            dot += __bfloat162float(x[i]) * __bfloat162float(gate_w[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if ((threadIdx.x & 31) == 0) s_part[threadIdx.x >> 5] = dot;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float g = 1.0f;
        if (gate_w != nullptr)
        {
            float t = 0.0f;
            for (int k = 0; k < (int) (blockDim.x >> 5); ++k) t += s_part[k];
            g = 1.0f / (1.0f + __expf(-t));
        }
        sel[topk] = (long long) shared_local;
        rw[topk] = __float2half_rn(g);
    }
}
