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
