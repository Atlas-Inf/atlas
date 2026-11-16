// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the routed experts of a batch on the block-scaled
// tensor cores.
//
// The checkpoint's experts are E2M1 values under one E8M0 scale per 32, and
// the model rounds what they multiply to E4M3 under one power-of-two scale
// per 32. That pair of formats is one tensor-core instruction on this
// hardware (`mma ... kind::mxf8f6f4.block_scale`, m16n8k32: sixteen rows of
// 32 E4M3 codes against eight rows of 32 E2M1 codes, each row with its
// block's scale), so a batch's experts need no widening and no repacking of
// values: the weights stream from where the loader put them.
//
// The products are the row-at-a-time kernels' (`dsv41_expert.cu`); the FP32
// sum of a row's products is in the instruction's order, one chain over K.
//
// A block is a tile of up to 128 of one expert's pairs against 128 rows of
// its weights, two halves of 64 that may be of two projections (the gate's
// and the up's, which read the same activations). Its warps go in twos, a
// half each, over a group of 64 pairs: a warp multiplies the group's four
// sets of 16 pairs by its half's eight sets of 8 rows, 32 instructions a
// step of 32 K values.
//
// What bounds it is how the bytes arrive, not the arithmetic. An
// instruction that copies from 32 rows, a lane a row, costs four times one
// whose lanes copy along a few rows (measured here, from L2: 216 against
// 1,900 GB/s; from memory 64 against 120), and a row of weights, of codes
// or of scales is a stream of its own. So nothing is fetched a row a
// thread. A block stages four steps of its weights at once, a warp's 32
// copies of one instruction being four rows' 64 bytes each; the pairs'
// codes two steps at a time, eight pairs an instruction; and the scales 32
// steps at a time, eight rows an instruction. Everything is copied into
// shared memory as it lies in global (`cp.async`) while the steps before
// are multiplied, and reaches the registers by the matrix loads made for
// the two operands: one takes 16 pairs, and one 8 weight rows, spreading
// the packed E2M1 codes to a code a byte.
//
// A block's size is its shared memory, and half of what an SM has lets two
// run on it, one multiplying while the other waits for memory: 128 pairs
// and four steps. (256 and eight, one block an SM, measured a tenth
// slower.) Each expert's weights are read once for every 128 pairs that
// chose it, from L2 after the first. With the weights cold a launch cannot
// be quicker than memory gives them along 128 rows at once, some 110 GB/s:
// a layer's 1.1 GB of gate and up rows a rank in 10 ms.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#ifndef MXG_THREADS
#define MXG_THREADS 128u
#endif
// Pairs a tile, 64 for every two warps; and weight rows a block.
#define MXG_ROWS MXG_THREADS
#define MXG_COLS 128u
// Steps of 32 K values a stage of weights, a stage of pairs, and a block of
// scales.
#ifndef MXG_STAGE
#define MXG_STAGE 4u
#endif
#define MXG_SUB 2u
#define MXG_SCALES 32u
// Shared memory: two sets of a stage of weights, [step][row][half], a half
// being 8 bytes of packed codes and 8 the matrix load does not read; two of
// a stage of pairs, [step][pair][32 codes]; and each pair's row of the
// codes. The unread bytes of a set of weights hold a block of scales, the
// two sets a block each in turn: a row's 32 in the first halves of its
// first four steps, and a pair's 32 in the second halves of the row of its
// number (or, with 256 pairs and eight steps, in the 256 halves of each of
// the last four steps).
#define MXG_W_STEP (MXG_COLS * 32u)
#define MXG_W_SET (MXG_STAGE * MXG_W_STEP)
#define MXG_A_STEP (MXG_ROWS * 32u)
#define MXG_A_SET (MXG_SUB * MXG_A_STEP)
#define MXG_A_AT (2u * MXG_W_SET)
// What a launch asks for (`moe_gemm.rs` has the same number).
#define MXG_ROW_AT (MXG_A_AT + 2u * MXG_A_SET)
#define MXG_SHARED (MXG_ROW_AT + MXG_ROWS * 4u)
// Where pair `q`'s scales of step-of-eight `at` of a block are, in a set.
#define MXG_PAIR_SCALES(q, at)                                                         \
    (MXG_ROWS == 256u ? (4u + (at)) * MXG_W_STEP + (q) * 16u + 8u                      \
                      : (at) * MXG_W_STEP + (q) * 32u + 24u)

// One tile: `count` (at most MXG_ROWS) pairs of `expert`, from pair `first`.
struct MxgTile {
    unsigned int expert;
    unsigned int first;
    unsigned int count;
};

// `bytes` (8 or 16) from global memory into shared, no register between.
static __device__ __forceinline__ void mxg_copy16(unsigned int to, const void* from) {
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" : : "r"(to), "l"(from));
}
static __device__ __forceinline__ void mxg_copy8(unsigned int to, const void* from) {
    asm volatile("cp.async.ca.shared.global [%0],[%1],8;\n" : : "r"(to), "l"(from));
}

// Sixteen rows of 32 E4M3 codes as the multiply's first operand: `at` is
// this lane's row of the four 8-row, 16-byte matrices.
static __device__ __forceinline__ void mxg_load_pairs(unsigned int (&a)[4], unsigned int at) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(at));
}

// Eight rows of 32 packed E2M1 codes as its second, a code a byte in the
// byte's bits 2 to 5: `at` is this lane's row of the two 8-row matrices, 16
// codes in 8 bytes and 8 bytes unread.
static __device__ __forceinline__ void mxg_load_weights(unsigned int& b0, unsigned int& b1,
                                                        unsigned int at) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(at));
    b0 <<= 2;
    b1 <<= 2;
}

// acc[16 pairs, 8 rows] += a[16, 32] (E4M3) . b[8, 32] (E2M1), each row
// under its block's E8M0 scale: byte `a_byte` of `sfa` and `b_byte` of
// `sfb`. A pair's scale is read from the lane that holds its codes, and a
// weight row's from the first lane of the four that hold its.
static __device__ __forceinline__ void mxg_mma(
    float (&acc)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1,
    unsigned int sfa, unsigned short a_byte, unsigned int sfb, unsigned short b_byte) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1"
        ".f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},"
        "{%14,%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(a_byte),
          "h"((unsigned short)0), "r"(sfb), "h"(b_byte), "h"((unsigned short)0));
}

// out{0,1}[first + m, col] = BF16(sum_k W[row, k] * a[pair m, k]) for every
// tile, where block x of the grid takes weight rows
// x * row_step + half * half_row + (0..64) of the projection at `mat{half}`
// bytes into an expert (its scales at `sc{half}`), and `col` is that row.
//
// codes [.., K] E4M3 bytes and sfa [.., K / 32] E8M0, a pair's row being
// a_rows[pair], or the pair with no `a_rows`. Codes the ranks gathered are
// in runs of `run` K values a row, a row's next run `run_stride` bytes
// after the same row's last; codes whole have a `run` of 0. A run is a
// multiple of 64. An expert is `stride` bytes
// after the last in `slab`; a projection's rows are K / 2 bytes, its scales
// K / 32. out rows are `out_stride` values apart. K % (32 * MXG_STAGE) == 0.
// Grid (weight rows / row_step, tiles), block MXG_THREADS, MXG_SHARED
// bytes of shared memory.
extern "C" __global__ void __launch_bounds__(MXG_THREADS) dsv41_mxg(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ sfa,
    const int* __restrict__ a_rows,
    const unsigned char* __restrict__ slab,
    const MxgTile* __restrict__ tiles,
    __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1,
    unsigned long long stride,
    unsigned int mat0,
    unsigned int mat1,
    unsigned int sc0,
    unsigned int sc1,
    unsigned int row_step,
    unsigned int half_row,
    unsigned int K,
    unsigned int out_stride,
    unsigned int run,
    unsigned long long run_stride) {
    extern __shared__ __align__(16) unsigned char sm[];
    const unsigned int sm_at = (unsigned int)__cvta_generic_to_shared(sm);

    const MxgTile tile = tiles[blockIdx.y];
    const unsigned int i = threadIdx.x, warp = i >> 5, lane = i & 31u;
    const unsigned int g = lane >> 2, t = lane & 3u;
    const unsigned int half = warp & 1u, group = warp >> 1;
    const unsigned int steps = K / 32u, stages = steps / MXG_STAGE;
    const unsigned char* expert = slab + (unsigned long long)tile.expert * stride;
    const unsigned long long block_row = (unsigned long long)blockIdx.x * row_step;
    // Weight row `row` of the block's 128, and its scales.
    auto weights = [&](unsigned int row) {
        const unsigned int h = row >> 6;
        return expert + (h ? mat1 : mat0) + (block_row + h * half_row + (row & 63u)) * (K / 2u);
    };
    auto weight_scales = [&](unsigned int row) {
        const unsigned int h = row >> 6;
        return expert + (h ? sc1 : sc0) + (block_row + h * half_row + (row & 63u)) * steps;
    };

    // Where pair `q`'s row of the codes is kept.
    auto pair_row = [&](unsigned int q) {
        return (unsigned int*)(sm + MXG_ROW_AT + q * 4u);
    };
    // This thread's pair: its row of the codes for the copies; and where
    // the tile has none, zeros for its codes and for its scales.
    if (i < tile.count) {
        const unsigned long long pair = tile.first + i;
        *pair_row(i) = a_rows ? (unsigned int)a_rows[pair] : (unsigned int)pair;
    } else {
        #pragma unroll
        for (unsigned int z = 0; z < 2u * MXG_SUB; ++z) {
            uint4* row = (uint4*)(sm + MXG_A_AT + z * MXG_A_STEP + i * 32u);
            row[0] = make_uint4(0u, 0u, 0u, 0u);
            row[1] = make_uint4(0u, 0u, 0u, 0u);
        }
        #pragma unroll
        for (unsigned int z = 0; z < 8u; ++z)
            *(unsigned long long*)(sm + (z >> 2) * MXG_W_SET + MXG_PAIR_SCALES(i, z & 3u)) =
                0ull;
    }
    __syncthreads();

    // A quarter of stage `st` of the weights into its set: 8 bytes a piece,
    // 8 pieces a row, four rows a warp's instruction.
    auto weights_in = [&](unsigned int st, unsigned int quarter) {
        const unsigned int to = sm_at + (st & 1u) * MXG_W_SET;
        #pragma unroll
        for (unsigned int j = 0; j < 512u / MXG_THREADS; ++j) {
            const unsigned int piece = (quarter * (512u / MXG_THREADS) + j) * MXG_THREADS + i;
            const unsigned int row = piece / (2u * MXG_STAGE), at = piece % (2u * MXG_STAGE);
            mxg_copy8(to + (at >> 1) * MXG_W_STEP + row * 32u + (at & 1u) * 16u,
                      weights(row) + (unsigned long long)st * (MXG_STAGE * 16u) + at * 8u);
        }
    };
    // Sub-stage `u` of the pairs' codes: 16 bytes a piece, 4 pieces a pair.
    auto pairs_in = [&](unsigned int u) {
        const unsigned int to = sm_at + MXG_A_AT + (u & 1u) * MXG_A_SET;
        #pragma unroll
        for (unsigned int j = 0; j < MXG_ROWS * MXG_SUB * 2u / MXG_THREADS; ++j) {
            const unsigned int piece = j * MXG_THREADS + i, pair = piece >> 2, at = piece & 3u;
            if (pair < tile.count) {
                const unsigned long long row = *pair_row(pair);
                const unsigned int k = u * (MXG_SUB * 32u);
                const unsigned char* from =
                    run ? codes + (unsigned long long)(k / run) * run_stride + row * run + k % run
                        : codes + row * K + k;
                mxg_copy16(to + (at >> 1) * MXG_A_STEP + pair * 32u + (at & 1u) * 16u,
                           from + at * 16u);
            }
        }
    };
    // Block `sb` of the scales into the unread bytes of its set: 8 bytes a
    // piece, 4 pieces a row or a pair. A piece past the last step is not
    // read, here or in memory.
    auto scales_in = [&](unsigned int sb) {
        const unsigned int to = sm_at + (sb & 1u) * MXG_W_SET + 8u;
        #pragma unroll
        for (unsigned int j = 0; j < (MXG_COLS + MXG_ROWS) * 4u / MXG_THREADS; ++j) {
            const unsigned int piece = j * MXG_THREADS + i, at = piece & 3u;
            const unsigned int step = sb * MXG_SCALES + at * 8u;
            if (step >= steps) continue;
            if (piece < MXG_COLS * 4u) {
                const unsigned int row = piece >> 2;
                mxg_copy8(to + at * MXG_W_STEP + row * 32u, weight_scales(row) + step);
            } else {
                const unsigned int pair = (piece >> 2) - MXG_COLS;
                if (pair < tile.count)
                    mxg_copy8(to - 8u + MXG_PAIR_SCALES(pair, at),
                              sfa + (unsigned long long)*pair_row(pair) * steps + step);
            }
        }
    };

    // What it multiplies: its lane's rows of the matrices.
    const unsigned int here = tile.count > group * 64u ? tile.count - group * 64u : 0u;
    const unsigned int sets = min((here + 15u) / 16u, 4u);
    const unsigned int a_from = sm_at + MXG_A_AT +
                                (group * 64u + (lane & 7u) + 8u * ((lane >> 3) & 1u)) * 32u +
                                16u * (lane >> 4);
    const unsigned int w_from =
        sm_at + (half * 64u + (lane & 7u)) * 32u + 16u * ((lane >> 3) & 1u);
    // Its scales: of rows g of its half's eight sets, and of pair g, or
    // 8 + g on a lane that holds a second pair's, of the group's four.
    const unsigned int ws_from = (half * 64u + g) * 32u + 8u;
    const unsigned int as_pair = group * 64u + g + 8u * (lane & 1u);

    float acc[4][8][4];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            acc[p][c][0] = 0.0f; acc[p][c][1] = 0.0f; acc[p][c][2] = 0.0f; acc[p][c][3] = 0.0f;
        }
    }

    const unsigned int subs = MXG_STAGE / MXG_SUB, total = stages * subs;
    const unsigned int blocks = (steps + MXG_SCALES - 1u) / MXG_SCALES;
    #pragma unroll
    for (unsigned int q = 0; q < subs; ++q) weights_in(0u, q);
    pairs_in(0u);
    scales_in(0u);
    asm volatile("cp.async.commit_group;\n" ::);
    for (unsigned int u = 0; u < total; ++u) {
        const unsigned int st = u / subs, sub = u % subs;
        // Ask for what comes next, into the sets these steps do not read.
        if (st + 1u < stages) weights_in(st + 1u, sub);
        if (u + 1u < total) pairs_in(u + 1u);
        const unsigned int sb = (u * MXG_SUB) / MXG_SCALES;
        if ((u * MXG_SUB) % MXG_SCALES == 0u && sb + 1u < blocks) scales_in(sb + 1u);
        asm volatile("cp.async.commit_group;\n" ::);
        // All asked for before that has arrived.
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        const unsigned int w_set = (st & 1u) * MXG_W_SET, a_set = (u & 1u) * MXG_A_SET;
        const unsigned int s_set = (sb & 1u) * MXG_W_SET;
        #pragma unroll
        for (unsigned int k2 = 0; k2 < MXG_SUB && sets > 0u; ++k2) {
            const unsigned int k = sub * MXG_SUB + k2;
            // Where the step's scales are in a row's 32: which of its four
            // steps of unread bytes, which word of the eight, which byte.
            const unsigned int in_block = (u * MXG_SUB + k2) % MXG_SCALES;
            const unsigned int scale = s_set + (in_block >> 3) * MXG_W_STEP + (in_block & 4u);
            const unsigned short byte = (unsigned short)(in_block & 3u);
            unsigned int b[8][2], sb_[8];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                mxg_load_weights(b[c][0], b[c][1], w_from + w_set + k * MXG_W_STEP + c * 8 * 32);
                sb_[c] = *(const unsigned int*)(sm + scale + ws_from + c * 8 * 32);
            }
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                if ((unsigned int)p < sets) {
                    unsigned int a[4];
                    mxg_load_pairs(a, a_from + a_set + k2 * MXG_A_STEP + p * 16 * 32);
                    const unsigned int sa = *(const unsigned int*)(
                        sm + s_set + MXG_PAIR_SCALES(as_pair + p * 16, in_block >> 3) +
                        (in_block & 4u));
                    #pragma unroll
                    for (int c = 0; c < 8; ++c)
                        mxg_mma(acc[p][c], a, b[c][0], b[c][1], sa, byte, sb_[c], byte);
                }
            }
        }
        // Every warp is past the sets the next steps are copied into.
        __syncthreads();
    }
    if (sets == 0u) return;
    __nv_bfloat16* out = half ? out1 : out0;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const unsigned int r0 = group * 64u + (unsigned int)p * 16u + g, r1 = r0 + 8u;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const unsigned long long col =
                block_row + half * half_row + (unsigned int)c * 8u + 2u * t;
            if (r0 < tile.count)
                *(__nv_bfloat162*)(out + (unsigned long long)(tile.first + r0) * out_stride +
                                   col) = __floats2bfloat162_rn(acc[p][c][0], acc[p][c][1]);
            if (r1 < tile.count)
                *(__nv_bfloat162*)(out + (unsigned long long)(tile.first + r1) * out_stride +
                                   col) = __floats2bfloat162_rn(acc[p][c][2], acc[p][c][3]);
        }
    }
}
