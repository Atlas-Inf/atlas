// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 reconstruct — expands an ExLlamaV3 linear's packed 16x-K trellis into
// its inner weight matrix W_inner (fp16, row-major [in_features, out_cols]).
//
// The device code is vendored from turboderp-org/exllamav3 @ 6b84a21
// into exl3_vendor/ (NOTICE.md; each header lists its minimal edits); this file only instantiates it. The
// trellis holds shuffled codebook indices (3INST procedural codebook, mul1 =
// cb 2 here); the shuffle is undone by the tensor-core fragment layout plus
// the shfl/smem round trip in reconstruct_tile — no dequant scale reaches
// W_inner, so this is a pure index -> half expansion.
//
// Launch geometry (from upstream reconstruct_slice, reconstruct.cu):
//   block: 256 threads.
//   grid:  (out_cols / 128, in_features / 16) — `out_cols` is the number of
//          output columns written, a multiple of 128; each block writes one
//          16-row x 128-column tile.
//   packed_blocks_n = out_features / 16 of the FULL tensor (tile columns of
//          the packed trellis), packed_n_offset = first tile column to read
//          (slicing; a multiple of 8).
//   out:   fp16 row-major [in_features, out_cols].
//
// K = bits per weight (upstream derives it from trellis.shape[-1] / 16). Only
// the integer-bit-rate mul1 entries the EXL3 checkpoints in scope need are
// instantiated here (k3..k6); the half-integer rates (1.5/2.5/3.5, the
// HALF = true template arm) are a later item.

#include <cuda_bf16.h>

#include "exl3_vendor/hadamard_inner.cuh"
#include "exl3_vendor/reconstruct_tile.cuh"

// One entry per integer bit rate; mul1 codebook = template arg cb = 2.
extern "C" __global__ __launch_bounds__(256) void exl3_reconstruct_mul1_k3(
    half* __restrict__ out, const uint16_t* __restrict__ packed, int packed_blocks_n, int packed_n_offset)
{
    reconstruct_tile<3, 2, false>(out, packed, packed_blocks_n, packed_n_offset);
}

extern "C" __global__ __launch_bounds__(256) void exl3_reconstruct_mul1_k4(
    half* __restrict__ out, const uint16_t* __restrict__ packed, int packed_blocks_n, int packed_n_offset)
{
    reconstruct_tile<4, 2, false>(out, packed, packed_blocks_n, packed_n_offset);
}

extern "C" __global__ __launch_bounds__(256) void exl3_reconstruct_mul1_k5(
    half* __restrict__ out, const uint16_t* __restrict__ packed, int packed_blocks_n, int packed_n_offset)
{
    reconstruct_tile<5, 2, false>(out, packed, packed_blocks_n, packed_n_offset);
}

extern "C" __global__ __launch_bounds__(256) void exl3_reconstruct_mul1_k6(
    half* __restrict__ out, const uint16_t* __restrict__ packed, int packed_blocks_n, int packed_n_offset)
{
    reconstruct_tile<6, 2, false>(out, packed, packed_blocks_n, packed_n_offset);
}

// ---------------------------------------------------------------------------
// Hadamard / conversion / GEMM for the EXL3 linear path:
//   y = x * W  with  W = diag(suh) * H * W_inner * H * diag(svh),
//   H = normalized 128-wide Sylvester Hadamard (block diagonal).
// Upstream evaluates it as xh = had_r_128(x * suh), y = xh * W_inner (fp16
// GEMM), y = had_r_128(y) * svh — the three entry points below are upstream
// had_hf_r_128_kernel<pre, post> (hadamard.cu lines 9-22), one per scale mode.
//
// Launch geometry (all three):
//   block: 32 threads (one warp transforms one 128-element row block).
//   grid:  (rows, cols / 128) — `cols % 128 == 0`; blockIdx.x picks the row,
//          blockIdx.y the 128-column block (had_hf_r_128_inner reads the scale
//          vector at blockIdx.y * 32 + lane, so `scale` holds `cols` fp16
//          entries: suh for pre, svh for post).
//   in-place is allowed (in == out).
//   r_scale = scale_factor / sqrt(128); pass 0.088388347648f for a plain,
//   unscaled H.

// xh = had_r_128(x * suh): pre-scale arm, had_hf_r_128_kernel<true, false>.
extern "C" __global__ __launch_bounds__(32) void exl3_had_r128_pre(
    const half* __restrict__ in, half* __restrict__ out, const half* __restrict__ scale, float r_scale)
{
    in += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    out += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<true, false>(in, out, scale, r_scale);
}

// y = had_r_128(y) * svh: post-scale arm, had_hf_r_128_kernel<false, true>.
extern "C" __global__ __launch_bounds__(32) void exl3_had_r128_post(
    const half* __restrict__ in, half* __restrict__ out, const half* __restrict__ scale, float r_scale)
{
    in += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    out += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<false, true>(in, out, scale, r_scale);
}

// Plain transform, no scale vector: had_hf_r_128_kernel<false, false>; `scale`
// is ignored (upstream passes nullptr) but keeps the launch signature uniform.
extern "C" __global__ __launch_bounds__(32) void exl3_had_r128_plain(
    const half* __restrict__ in, half* __restrict__ out, const half* __restrict__ scale, float r_scale)
{
    in += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    out += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<false, false>(in, out, scale, r_scale);
}

// ---------------------------------------------------------------------------
// Atlas activations are bf16; the EXL3 linear path above is fp16 end to end.
// These two conversions bridge the two worlds (n = number of elements,
// block 256, grid-stride).

extern "C" __global__ void exl3_bf16_to_f16(const __nv_bfloat16* __restrict__ in, half* __restrict__ out, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = __float2half_rn(__bfloat162float(in[i]));
}

extern "C" __global__ void exl3_f16_to_bf16(const half* __restrict__ in, __nv_bfloat16* __restrict__ out, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = __float2bfloat16_rn(__half2float(in[i]));
}

// ---------------------------------------------------------------------------
// c[m, n] = a[m, k] * b[k, n], all row-major fp16, fp32 accumulation, one
// rounding at the store. Correctness-first: a plain 16x16 shared-memory-tiled
// sgemm-style kernel — the EXL3 path needs SOME fp16 GEMM (Atlas has none and
// its bf16 GEMMs want the weight as [N, K]); speed comes in a later milestone.
// block: (16, 16). grid: (ceil(n / 16), ceil(m / 16)). m, n, k need not be
// multiples of 16 — every load/store is guarded. --fmad=false keeps the
// accumulation order deterministic (plain += of products, no fma intrinsics).

#define EXL3_HGEMM_TILE 16

extern "C" __global__ __launch_bounds__(256) void exl3_hgemm_f16(
    const half* __restrict__ a, const half* __restrict__ b, half* __restrict__ c,
    unsigned m, unsigned n, unsigned k)
{
    __shared__ half sa[EXL3_HGEMM_TILE][EXL3_HGEMM_TILE];
    __shared__ half sb[EXL3_HGEMM_TILE][EXL3_HGEMM_TILE];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned row = blockIdx.y * EXL3_HGEMM_TILE + ty;
    const unsigned col = blockIdx.x * EXL3_HGEMM_TILE + tx;

    // One element of each tile per thread: sa holds a[row block, k0 .. k0+16),
    // sb holds b[k0 .. k0+16, col block]; out-of-range elements load as zero.
    float acc = 0.0f;
    for (unsigned k0 = 0; k0 < k; k0 += EXL3_HGEMM_TILE)
    {
        const unsigned ka = k0 + tx;
        const unsigned kb = k0 + ty;
        sa[ty][tx] = (row < m && ka < k) ? a[(size_t) row * k + ka] : __float2half(0.0f);
        sb[ty][tx] = (kb < k && col < n) ? b[(size_t) kb * n + col] : __float2half(0.0f);
        __syncthreads();

        #pragma unroll
        for (unsigned i = 0; i < EXL3_HGEMM_TILE; i++)
            acc += __half2float(sa[ty][i]) * __half2float(sb[i][tx]);
        __syncthreads();
    }

    if (row < m && col < n)
        c[(size_t) row * n + col] = __float2half_rn(acc);
}
