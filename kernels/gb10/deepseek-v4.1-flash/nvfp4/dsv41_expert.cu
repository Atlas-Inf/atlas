// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the experts of one token in three launches.
//
// A token runs seven experts: the six the router chose, FP4, and the shared
// one, FP8. Each is two projections in (gate and up), an activation, and one
// projection out. Launched one projection at a time that is 21 projections
// and 14 small kernels per part, and at one token the launches cost more
// than the arithmetic. Here every projection in is a row of one grid, the
// activations are one block, and every projection out is a row of another.
//
// A block computes what the same block of `dsv41_{fp4,fp8}_gemv_tc8[_f32]`
// computes, from staged weights (`dsv41_tc.cuh`), and the activation is
// `dsv41_swiglu` then `dsv41_fp8_quant_dequant` (`dsv41_common.cuh`), so the
// bytes are those of the launches these replace.
//
// A slot is an expert: the chosen in ascending expert order (`chosen`, as
// `dsv41_moe_pick` leaves them with `by_expert`), then the shared expert.
// The routed experts' weights are one part of the layer's slab: expert e at
// `slab + e * stride`, as the part's rows of w1, of w3 and of w2, each
// `data` bytes followed by `scales` bytes of E8M0 scales.
//
// A part is a quarter of an expert's hidden rows, and a quarter of its
// output rows. The projections in leave a quarter of every slot's hidden
// activation, the ranks gather the quarters, and the projection out reads
// them all: it is in pieces (`f4_at` in `dsv41_tc.cuh`), a piece what one
// part computed.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "dsv41_common.cuh"
#include "dsv41_tc.cuh"

// Projection `which` (0 w1, 1 w3, 2 w2) of the expert in routed slot `slot`.
__device__ __forceinline__ const unsigned char* dsv41_expert_weights(
    const unsigned char* slab, const unsigned int* chosen, unsigned int slot,
    unsigned long long stride, unsigned int data, unsigned int scales, unsigned int which) {
    return slab + (unsigned long long)chosen[slot] * stride +
           (unsigned long long)which * (data + scales);
}

// gate_up[slot][0] = BF16(w1 . x), gate_up[slot][1] = BF16(w3 . x): the
// part's `rows` rows of each. x [K] BF16, FP8-rounded. shared_w / shared_s
// are the shared expert's w1 then w3 (two pointers each).
// Grid (ceil(rows / 16), 2 * (routed + 1)), block 256.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_expert_in(
    const __nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ slab,
    const unsigned int* __restrict__ chosen,
    const unsigned char* __restrict__ shared_w1,
    const unsigned char* __restrict__ shared_s1,
    const unsigned char* __restrict__ shared_w3,
    const unsigned char* __restrict__ shared_s3,
    __nv_bfloat16* __restrict__ gate_up,
    unsigned int routed,
    unsigned int rows,
    unsigned int K,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned int slot = blockIdx.y / 2, which = blockIdx.y & 1u;
    __nv_bfloat16* out = gate_up + (size_t)blockIdx.y * rows;
    if (slot < routed) {
        const unsigned char* w =
            dsv41_expert_weights(slab, chosen, slot, stride, data, scales, which);
        dsv41_gemv_staged_impl<1, true, false>(sh, staged, x, w, w + data, out, 1, rows, K, rows,
                                               blockIdx.x, K);
    } else {
        dsv41_gemv_staged_impl<1, false, false>(
            sh, staged, x, which ? shared_w3 : shared_w1, which ? shared_s3 : shared_s1, out, 1,
            rows, K, rows, blockIdx.x, K);
    }
}

// inner[slot] = fp8(BF16(weight * silu(min(gate, limit)) * clamp(up, limit))),
// the routed slots under their routing `weights`, the shared one under 1.
// One thread per 32 values; rows is a multiple of 32.
extern "C" __global__ void dsv41_expert_act(
    const __nv_bfloat16* __restrict__ gate_up,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ inner,
    unsigned int routed,
    unsigned int slots,
    unsigned int rows,
    float limit) {
    atlas_pdl_enter();
    const unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_slot = rows / 32;
    if (b >= slots * per_slot) return;
    const unsigned int slot = b / per_slot, at = (b % per_slot) * 32;
    const float weight = slot < routed ? weights[slot] : 1.0f;
    const __nv_bfloat16* gate = gate_up + (size_t)(2 * slot) * rows + at;
    const __nv_bfloat16* up = gate + rows;
    float v[32];
    for (int i = 0; i < 32; ++i)
        v[i] = dsv41_swiglu_one(__bfloat162float(gate[i]), __bfloat162float(up[i]), weight, limit);
    dsv41_fp8_block(v, inner + (size_t)slot * rows + at);
}

// out[slot] = BF16(w2 . inner[slot]) for the part's N rows of w2: [slots, N].
// Slot s of piece p of `inner` is at inner[p * piece_stride + s * rows].
// Grid (ceil(N / 16), routed + 1), block 256.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_expert_out(
    const __nv_bfloat16* __restrict__ inner,
    const unsigned char* __restrict__ slab,
    const unsigned int* __restrict__ chosen,
    const unsigned char* __restrict__ shared_w2,
    const unsigned char* __restrict__ shared_s2,
    __nv_bfloat16* __restrict__ out,
    unsigned int routed,
    unsigned int N,
    unsigned int rows,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales,
    unsigned long long piece_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned int slot = blockIdx.y, K = rows * DSV41_PARTS;
    const __nv_bfloat16* a = inner + (size_t)slot * rows;
    __nv_bfloat16* c = out + (size_t)slot * N;
    if (slot < routed) {
        const unsigned char* w = dsv41_expert_weights(slab, chosen, slot, stride, data, scales, 2);
        dsv41_gemv_staged_impl<1, true, false>(sh, staged, a, w, w + data, c, 1, N, K, N, blockIdx.x,
                                               rows, nullptr, nullptr, rows, piece_stride);
    } else {
        dsv41_gemv_staged_impl<1, false, false>(sh, staged, a, shared_w2, shared_s2, c, 1, N, K, N,
                                                blockIdx.x, rows, nullptr, nullptr, rows,
                                                piece_stride);
    }
}

// The routed experts of a batch of rows, grouped. Rows that chose the same
// expert share a launch's read of its weights: a tile is up to 32 of them,
// `first` on in the order the host sorted (row, slot) pairs by expert. A
// row's bytes are those of `dsv41_expert_in` / `_act` / `_out` on it alone,
// for a block computes a row from that row.
struct Dsv41Tile {
    unsigned int expert;
    unsigned int first;
    unsigned int count;
};

// The (row, slot) pairs of a few rows in expert order, as the host puts a
// batch's (`moe_grouped::Plan`), without leaving the device: a step that
// waits on the host for this, forty times, is a step the ranks fall out of
// with one another. chosen [rows, top_k] u32 and weights [rows, top_k] FP32
// as the router leaves them, a pair `row * top_k + slot`. A pair's place is
// the pairs of lesser experts and the earlier pairs of its own. tiles
// [pairs]: a tile an expert chosen, in expert order, and none (count 0)
// after them; row_of and out_row [pairs] i32 and sorted [pairs] FP32 by
// place. One block, a thread a pair: at most 1,024 of them, and no expert
// with more than a tile's.
extern "C" __global__ void dsv41_experts_plan(
    const unsigned int* __restrict__ chosen,
    const float* __restrict__ weights,
    Dsv41Tile* __restrict__ tiles,
    int* __restrict__ row_of,
    int* __restrict__ out_row,
    float* __restrict__ sorted,
    unsigned int top_k,
    unsigned int slots) {
    atlas_pdl_enter();
    __shared__ unsigned int expert[1024];
    __shared__ unsigned char leads[1024];
    const unsigned int pairs = blockDim.x, p = threadIdx.x;
    expert[p] = chosen[p];
    __syncthreads();
    const unsigned int e = expert[p];
    unsigned int lesser = 0, same = 0, earlier = 0;
    for (unsigned int q = 0; q < pairs; ++q) {
        lesser += expert[q] < e;
        same += expert[q] == e;
        earlier += expert[q] == e && q < p;
    }
    leads[p] = earlier == 0;
    __syncthreads();
    // The experts chosen, and those before this pair's.
    unsigned int all = 0, tile = 0;
    for (unsigned int q = 0; q < pairs; ++q) {
        all += leads[q];
        tile += leads[q] && expert[q] < e;
    }
    const unsigned int at = lesser + earlier;
    row_of[at] = (int)(p / top_k);
    out_row[at] = (int)((p / top_k) * slots + p % top_k);
    sorted[at] = weights[p];
    if (leads[p]) tiles[tile] = Dsv41Tile{e, lesser, same};
    if (p >= all) tiles[p] = Dsv41Tile{0u, 0u, 0u};
}

// gate_up[pair][0] = BF16(w1 . x[row_of[pair]]), [pair][1] the same of w3.
// x [T, K] FP8-rounded BF16; gate_up [pairs, 2, rows]. A tile of no pairs is
// none. Grid (ceil(rows / 16), 2 * tiles), block 256.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_experts_in(
    const __nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ slab,
    const Dsv41Tile* __restrict__ tiles,
    const int* __restrict__ row_of,
    __nv_bfloat16* __restrict__ gate_up,
    unsigned int rows,
    unsigned int K,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales) {
    atlas_pdl_enter();
    __shared__ F4Shared<4> sh;
    __shared__ F4Staged staged;
    const Dsv41Tile tile = tiles[blockIdx.y / 2];
    if (tile.count == 0u) return;
    const unsigned int which = blockIdx.y & 1u;
    const unsigned char* w = slab + (unsigned long long)tile.expert * stride +
                             (unsigned long long)which * (data + scales);
    dsv41_gemv_staged_impl<4, true, false>(
        sh, staged, x, w, w + data, gate_up + ((size_t)tile.first * 2 + which) * rows, tile.count,
        rows, K, 2 * rows, blockIdx.x, K, row_of + tile.first);
}

// inner[r] = fp8(BF16(weight[r] * silu(min(gate, limit)) * clamp(up, limit)))
// for each of `count` rows; with no `weights`, under 1 (the shared expert).
// Row r's gate values are at gate_up[r * row_stride] and its up values
// `up_offset` after them: [count, 2, rows] has a row_stride of 2 * rows and
// an up_offset of rows. One thread per 32 values.
extern "C" __global__ void dsv41_experts_act(
    const __nv_bfloat16* __restrict__ gate_up,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ inner,
    unsigned int count,
    unsigned int rows,
    float limit,
    unsigned long long up_offset,
    unsigned long long row_stride) {
    atlas_pdl_enter();
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_row = rows / 32;
    if (b >= (size_t)count * per_row) return;
    const size_t r = b / per_row;
    const unsigned int at = (unsigned int)(b % per_row) * 32;
    const float weight = weights ? weights[r] : 1.0f;
    const __nv_bfloat16* gate = gate_up + r * row_stride + at;
    const __nv_bfloat16* up = gate + up_offset;
    float v[32];
    for (int i = 0; i < 32; ++i)
        v[i] = dsv41_swiglu_one(__bfloat162float(gate[i]), __bfloat162float(up[i]), weight, limit);
    dsv41_fp8_block(v, inner + r * rows + at);
}

// The same activations as the ranks send them and the tensor cores read
// them (`dsv41_mxg.cu`): codes[r] the E4M3 codes and sf[r] the E8M0 scale of
// each 32, half the bytes of the values. A code times its scale is the
// value `dsv41_experts_act` leaves.
extern "C" __global__ void dsv41_experts_act_codes(
    const __nv_bfloat16* __restrict__ gate_up,
    const float* __restrict__ weights,
    unsigned char* __restrict__ codes,
    unsigned char* __restrict__ sf,
    unsigned int count,
    unsigned int rows,
    float limit,
    unsigned long long up_offset,
    unsigned long long row_stride) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_row = rows / 32;
    if (b >= (size_t)count * per_row) return;
    const size_t r = b / per_row;
    const unsigned int at = (unsigned int)(b % per_row) * 32;
    const float weight = weights ? weights[r] : 1.0f;
    const __nv_bfloat16* gate = gate_up + r * row_stride + at;
    const __nv_bfloat16* up = gate + up_offset;
    float v[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        v[i] = dsv41_swiglu_one(__bfloat162float(gate[i]), __bfloat162float(up[i]), weight, limit);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float scale = dsv41_fp8_scale(amax);
    unsigned char* dst = codes + r * rows + at;
    for (int i = 0; i < 32; ++i) {
        const float scaled = fminf(fmaxf(v[i] / scale, -448.0f), 448.0f);
        dst[i] = (unsigned char)__nv_cvt_float_to_fp8(dsv41_round_e4m3(scaled), __NV_SATFINITE,
                                                      __NV_E4M3);
    }
    sf[b] = (unsigned char)(dsv41_exponent(scale) + 127);
}

// out[out_row[pair]] = BF16(w2 . inner[pair]) for the part's N rows of w2.
// Pair q of piece p of `inner` is at inner[p * piece_stride + q * rows].
// Grid (ceil(N / 16), tiles), block 256.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_experts_out(
    const __nv_bfloat16* __restrict__ inner,
    const unsigned char* __restrict__ slab,
    const Dsv41Tile* __restrict__ tiles,
    const int* __restrict__ out_row,
    __nv_bfloat16* __restrict__ out,
    unsigned int N,
    unsigned int rows,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales,
    unsigned long long piece_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<4> sh;
    __shared__ F4Staged staged;
    const Dsv41Tile tile = tiles[blockIdx.y];
    if (tile.count == 0u) return;
    const unsigned char* w =
        slab + (unsigned long long)tile.expert * stride + 2ull * (data + scales);
    dsv41_gemv_staged_impl<4, true, false>(
        sh, staged, inner + (size_t)tile.first * rows, w, w + data, out, tile.count, N,
        rows * DSV41_PARTS, N, blockIdx.x, rows, nullptr, out_row + tile.first, rows,
        piece_stride);
}
