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
// Fused dense rebuild (Atlas code): W^T [out, in] bf16 from W_inner [in, out]
// fp16, i.e. steps 2-5 of exl3_dense_bf16_nk in one kernel:
//   had_r128_post over the rows of [in, out] with svh, transpose,
//   had_r128_post over the rows of [out, in] with suh, fp16 -> bf16.
// The 128-wide Hadamard only mixes within 128-element blocks, so each 128x128
// output tile depends on one 128x128 tile of W_inner plus a 128-slice of svh
// and of suh; the whole chain runs on that tile in shared memory. Every fp16
// rounding point of the five-kernel chain is kept (the tile is stored as fp16
// after each Hadamard pass), so the result is bitwise identical to it.

// Twin of the vendored had_hf_r_128_inner<false, true> (exl3_vendor/
// hadamard_inner.cuh): identical load, butterfly, shuffle_had_f4x32, rounding,
// post-scale and store. The only change: the scale slice is passed in
// (pointing at this block's 128 scales) instead of being indexed by
// blockIdx.y, so it can run from any grid.
__device__ __forceinline__ void exl3_had_row_post(
    const half* __restrict__ input_ptr, half* __restrict__ output_ptr,
    const half* __restrict__ scale_slice, const float r_scale)
{
    int t = threadIdx.x & 31;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    // Post scale
    half4 scales = ((half4*) scale_slice)[t];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);

    // Store
    ((half4*) output_ptr)[t] = v;
}

// Padded row stride (halves): keeps half4 alignment and staggers banks.
#define EXL3_H2T_STRIDE 136

// Grid (out_f / 128, in_f / 128), block 128 (4 warps), dynamic shared memory
// 2 * 128 * EXL3_H2T_STRIDE * sizeof(half) = 69,632 bytes.
extern "C" __global__ __launch_bounds__(128) void exl3_had2_transpose_bf16(
    const half* __restrict__ w, __nv_bfloat16* __restrict__ out, const half* __restrict__ suh,
    const half* __restrict__ svh, unsigned in_f, unsigned out_f, float r_scale)
{
    extern __shared__ __align__(16) unsigned char exl3_h2t_smem[];
    half* a = (half*) exl3_h2t_smem;              // [128 in rows][128 out cols]
    half* b = a + 128 * EXL3_H2T_STRIDE;          // [128 out rows][128 in cols]
    const unsigned r0 = blockIdx.y * 128;         // first in row
    const unsigned c0 = blockIdx.x * 128;         // first out col
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;

    // Load the W_inner tile (coalesced half2 along the out columns).
    for (unsigned idx = tid; idx < 128 * 64; idx += 128)
    {
        const unsigned i = idx >> 6, j2 = idx & 63;
        ((half2*) (a + i * EXL3_H2T_STRIDE))[j2] =
            ((const half2*) (w + (size_t) (r0 + i) * out_f + c0))[j2];
    }
    __syncthreads();

    // Right Hadamard with svh: one warp per 128-wide row, as exl3_had_r128_post
    // does over the rows of [in, out] (its blockIdx.y == c0 / 128 -> svh + c0).
    for (unsigned i = warp; i < 128; i += 4)
        exl3_had_row_post(a + i * EXL3_H2T_STRIDE, a + i * EXL3_H2T_STRIDE, svh + c0, r_scale);
    __syncthreads();

    // Transpose the tile.
    for (unsigned idx = tid; idx < 128 * 128; idx += 128)
    {
        const unsigned i = idx & 127, j = idx >> 7;
        b[j * EXL3_H2T_STRIDE + i] = a[i * EXL3_H2T_STRIDE + j];
    }
    __syncthreads();

    // Left Hadamard with suh over the rows of [out, in] (blockIdx.y there is
    // r0 / 128 -> suh + r0).
    for (unsigned j = warp; j < 128; j += 4)
        exl3_had_row_post(b + j * EXL3_H2T_STRIDE, b + j * EXL3_H2T_STRIDE, suh + r0, r_scale);
    __syncthreads();

    // fp16 -> bf16 (as exl3_f16_to_bf16), coalesced along the in columns.
    for (unsigned idx = tid; idx < 128 * 64; idx += 128)
    {
        const unsigned j = idx >> 6, i2 = idx & 63;
        const half2 h = ((const half2*) (b + j * EXL3_H2T_STRIDE))[i2];
        __nv_bfloat162 o;
        o.x = __float2bfloat16_rn(__half2float(__low2half(h)));
        o.y = __float2bfloat16_rn(__half2float(__high2half(h)));
        ((__nv_bfloat162*) (out + (size_t) (c0 + j) * in_f + r0))[i2] = o;
    }
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
// out[c * rows + r] = in[r * cols + c]: the fp16 transpose, used by the
// load-time dequant (exl3_dense_bf16_nk) to walk W [in, out] to W^T [out, in].
// Atlas code — the same 32x32 shared-memory tile (+1 padded, so both the read
// and the write are coalesced within warps) as common/transpose_u8.cu.
// Grid: (ceil(cols / 32), ceil(rows / 32))  Block: (32, 8), each thread moving
// four elements. rows and cols need not be multiples of 32: every access is
// guarded.

#define EXL3_TRANSPOSE_TILE 32

extern "C" __global__ __launch_bounds__(256) void exl3_transpose_f16(
    const half* __restrict__ in, half* __restrict__ out, unsigned rows, unsigned cols)
{
    __shared__ half tile[EXL3_TRANSPOSE_TILE][EXL3_TRANSPOSE_TILE + 1];

    const unsigned ix = blockIdx.x * EXL3_TRANSPOSE_TILE + threadIdx.x;  // col of in
    const unsigned iy_base = blockIdx.y * EXL3_TRANSPOSE_TILE + threadIdx.y;

    #pragma unroll
    for (unsigned j = 0; j < EXL3_TRANSPOSE_TILE; j += 8)
    {
        const unsigned r = iy_base + j;
        if (r < rows && ix < cols)
            tile[threadIdx.y + j][threadIdx.x] = in[(size_t) r * cols + ix];
    }
    __syncthreads();

    const unsigned ox = blockIdx.y * EXL3_TRANSPOSE_TILE + threadIdx.x;  // row of out
    const unsigned oy_base = blockIdx.x * EXL3_TRANSPOSE_TILE + threadIdx.y;

    #pragma unroll
    for (unsigned j = 0; j < EXL3_TRANSPOSE_TILE; j += 8)
    {
        const unsigned c = oy_base + j;
        if (c < cols && ox < rows)
            out[(size_t) c * rows + ox] = tile[threadIdx.x][threadIdx.y + j];
    }
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
