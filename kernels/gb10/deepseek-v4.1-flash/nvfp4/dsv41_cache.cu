// SPDX-License-Identifier: AGPL-3.0-only
//
// What DeepSeek-V4.1-Flash writes into its attention caches: rotary
// positions on the last values of a vector, and the reference's FP8 and FP4
// quantize-then-dequantize, which leave BF16 values that are exactly what
// the quantized form would decode to. The host statement of each rule, which
// these kernels are tested against byte for byte, is
// `spark-model/src/layers/deepseek_v41/cache_recipe.rs`.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

// Rotate the last `rope` values of every vector to its row's position.
// x [rows, vectors, dim] BF16, positions [rows], freqs [rope / 2] radians per
// position. `inverse` turns the other way. One thread per value pair.
extern "C" __global__ void dsv41_rotate(
    __nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rows,
    unsigned int vectors,
    unsigned int dim,
    unsigned int rope,
    unsigned int inverse) {
    atlas_pdl_enter();
    const unsigned int pairs = rope / 2;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * vectors * pairs) return;
    const unsigned int pair = (unsigned int)(i % pairs);
    const size_t vector = i / pairs;
    const float angle = (float)positions[vector / vectors] * freqs[pair];
    __nv_bfloat16* p = x + vector * dim + (dim - rope) + 2 * pair;
    float re, im;
    dsv41_rotate_pair(__bfloat162float(p[0]), __bfloat162float(p[1]), angle, inverse != 0u, re, im);
    p[0] = __float2bfloat16(re);
    p[1] = __float2bfloat16(im);
}

// FP8 (E4M3) under one power-of-two scale per 32 values, written as BF16:
// `blocks` blocks of 32 from `src` into `dst`, which may be the same buffer.
// One thread per two values, which it reads and writes as a word: the 32 of
// a scale block are the pairs of 16 neighbouring lanes, which find the
// block's largest by shuffles (`dsv41_fp8_block`'s bytes). A thread a block
// took 32 loads and roundings in a row, 7 us of a launch three times a layer
// of a decode step. Block a multiple of 32 threads.
extern "C" __global__ void dsv41_fp8_quant_dequant(
    const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int blocks) {
    atlas_pdl_enter();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const bool mine = i < (size_t)blocks * 16;
    float v0 = 0.0f, v1 = 0.0f;
    if (mine) dsv41_unpack2(((const unsigned int*)src)[i], v0, v1);
    float amax = fmaxf(fabsf(v0), fabsf(v1));
    #pragma unroll
    for (unsigned int o = 8; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (!mine) return;
    const float scale = dsv41_fp8_scale(amax);
    ((unsigned int*)dst)[i] = dsv41_pack2(dsv41_fp8_value(v0, scale), dsv41_fp8_value(v1, scale));
}

// FP4 (E2M1) written back as BF16, one thread per group. `e4m3_scale` = 0:
// groups of 32 under a power-of-two scale (index keys). 1: groups of 16 under
// an E4M3 scale (compressed keys).
extern "C" __global__ void dsv41_fp4_quant_dequant(
    __nv_bfloat16* __restrict__ x, unsigned int groups, unsigned int e4m3_scale) {
    atlas_pdl_enter();
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int width = e4m3_scale ? 16 : 32;
    __nv_bfloat16* p = x + g * width;
    float amax = 0.0f;
    for (int i = 0; i < width; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(p[i])));
    const float scale = e4m3_scale
        ? dsv41_round_e4m3(fmaxf(amax, 6.0f * 0x1p-9f) / 6.0f)
        : dsv41_pow2_ceil(fmaxf(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    for (int i = 0; i < width; ++i) {
        const float scaled = fminf(fmaxf(__bfloat162float(p[i]) / scale, -6.0f), 6.0f);
        p[i] = __float2bfloat16(dsv41_round_fp4(scaled) * scale);
    }
}

// The window ring: [128, 512] BF16, slot p % 128 the key of position p. A
// key is 64 words of 16 bytes, and a thread moves one.
#define DSV41_RING 128u
#define DSV41_KEY_QUADS 64u

// The window of a batch of `rows` rows from `position`, in `run` [128 +
// rows, 512]: the ring unrolled into position order (entry e is position
// `position` - 128 + e), ahead of the batch's own keys, which the launch
// before this one wrote there. A row then reads its 128 by position, and no
// row's key displaces one an earlier row still reads. The batch's last 128
// keys go on into the ring, and with `keep` the keys its first `keep` rows
// displace into `kept` (a verify's rows may be taken back:
// `dsv41_window_restore`). One launch for what was five copies, which a
// stream runs one at a time with nothing beside them. Grid (128 * 64 / 256).
extern "C" __global__ void dsv41_window_unroll(
    uint4* __restrict__ ring,
    uint4* __restrict__ run,
    uint4* __restrict__ kept,
    unsigned int position,
    unsigned int rows,
    unsigned int keep) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DSV41_RING * DSV41_KEY_QUADS) return;
    const unsigned int e = i / DSV41_KEY_QUADS, q = i % DSV41_KEY_QUADS;
    const unsigned int slot = (position + e) % DSV41_RING;
    const uint4 old = ring[slot * DSV41_KEY_QUADS + q];
    run[e * DSV41_KEY_QUADS + q] = old;
    if (e < keep) kept[e * DSV41_KEY_QUADS + q] = old;
    // The one of the batch's last 128 rows whose position has this slot.
    const unsigned int first = rows > DSV41_RING ? rows - DSV41_RING : 0u;
    const unsigned int row =
        first + (slot + DSV41_RING - (position + first) % DSV41_RING) % DSV41_RING;
    if (row < rows) ring[slot * DSV41_KEY_QUADS + q] = run[(DSV41_RING + row) * DSV41_KEY_QUADS + q];
}

// The ring gets back the keys that rows `from`.. of a batch of `rows` from
// `position` displaced: a verify's refused rows (`kept` as
// `dsv41_window_unroll` left it). Grid (ceil((rows - from) * 64 / 256)).
extern "C" __global__ void dsv41_window_restore(
    const uint4* __restrict__ kept,
    uint4* __restrict__ ring,
    unsigned int position,
    unsigned int from,
    unsigned int rows) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (rows - from) * DSV41_KEY_QUADS) return;
    const unsigned int row = from + i / DSV41_KEY_QUADS, q = i % DSV41_KEY_QUADS;
    ring[((position + row) % DSV41_RING) * DSV41_KEY_QUADS + q] = kept[row * DSV41_KEY_QUADS + q];
}
