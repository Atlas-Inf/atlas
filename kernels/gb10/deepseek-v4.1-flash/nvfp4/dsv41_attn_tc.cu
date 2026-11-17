// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash attention over gathered keys on tensor cores: the
// first and third launches of the shared form of `dsv41_attn.cu`, with
// `dsv41_attn_schedule` between them as there, for a few queries; and the
// three in one launch for a batch (`dsv41_attn_batch_tc`).
//
// The schedule is that file's: keys 64 at a time in the order of the index
// row, a running maximum that rescales what has been summed, weights rounded
// to BF16 where they multiply a key. What changes is the order of two FP32
// sums. A logit's 512 products are added by 32 MMA steps, not one after
// another; and the 64 keys of a step are added to an output value four MMA
// steps of 16 keys at a time, the sum so far rescaled between steps of 64
// as before. Every product is the same one: q, the keys and the weights are
// BF16. So a query's bytes differ from the row-at-a-time form's as the
// reference's own differ between batch shapes, and a prefill runs this
// where a decode step runs that.
//
// A batch's block is one query and 16 heads. A few queries leave most of
// the GPU idle that way, so theirs are spread:
// the logits over the blocks of a grid row, each taking every
// `gridDim.x`-th run of keys, and the output over sixteen blocks of 32
// values (`dsv41_attn_output_tc_few`). No sum is cut, so the bytes are the
// same: a decode step, a verify's rows and a prefill's batch are one
// arithmetic.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "../../common/mxfp8_act_quant.cuh"
#include "dsv41_common.cuh"

#define DSV41_DIM 512
#define DSV41_MAX_KEYS 640
#define DSV41_KEYS_PER_STEP 64
#define DSV41_MAX_STEPS (DSV41_MAX_KEYS / DSV41_KEYS_PER_STEP)
#define DSV41_GROUP 16
#define DSV41_RECORD_SHRINK DSV41_MAX_KEYS
#define DSV41_RECORD_DENOMINATOR (DSV41_MAX_KEYS + DSV41_MAX_STEPS)
#define DSV41_RECORD 656

static __device__ __forceinline__ unsigned int dsv41_tc_steps(unsigned int k) {
    return (k + DSV41_KEYS_PER_STEP - 1) / DSV41_KEYS_PER_STEP;
}

// Key `j` of a query's index row `ids`: its row of 512 values, or none.
static __device__ __forceinline__ const __nv_bfloat16* dsv41_tc_key(
    const __nv_bfloat16* keys_a, const __nv_bfloat16* keys_b, const int* ids, unsigned int j,
    unsigned int k, unsigned int split) {
    const int id = j < k ? ids[j] : -1;
    return id < 0 ? nullptr : (j < split ? keys_a : keys_b) + (size_t)id * DSV41_DIM;
}

// One m16n8k16 BF16 MMA step into `acc` (FP32).
static __device__ __forceinline__ void dsv41_tc_mma(
    float (&acc)[4], const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// records [m, h, DSV41_RECORD] FP32: a logit a key. With rope > 0, q is
// turned to positions [m] (u32) by freqs [rope / 2] first, as
// `dsv41_attn_logits` turns it. q [m, h, 512] BF16; h a multiple of 16.
//
// The block stages its 16 heads of q in shared memory, turned. A warp takes
// 16 keys at a time, a lane loading 16 consecutive values of two of them,
// and multiplies them with the heads eight at a time: keys by heads is the
// MMA's 16 by 8, and the 16 values go to the same k-slots of both operands
// (`dsv41_tc.cu`).
extern "C" __global__ void __launch_bounds__(256) dsv41_attn_logits_tc(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys_a,
    const __nv_bfloat16* __restrict__ keys_b,
    const int* __restrict__ idx,
    float* __restrict__ records,
    unsigned int h,
    unsigned int k,
    unsigned int split,
    float scale,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rope) {
    atlas_pdl_enter();
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int query = blockIdx.y / groups, first = (blockIdx.y % groups) * DSV41_GROUP;
    const unsigned int t = threadIdx.x;
    __shared__ uint4 qs[DSV41_GROUP * DSV41_DIM / 8];

    const uint4* qv = (const uint4*)(q + ((size_t)query * h + first) * DSV41_DIM);
    for (unsigned int i = t; i < DSV41_GROUP * DSV41_DIM / 8; i += 256) qs[i] = qv[i];
    __syncthreads();
    // A thread per rotary pair of a head, as many of them as it takes.
    __nv_bfloat16* qb = (__nv_bfloat16*)qs;
    const unsigned int pairs = rope / 2, turned = DSV41_DIM - rope;
    for (unsigned int i = t; i < DSV41_GROUP * pairs; i += 256) {
        __nv_bfloat16* pair = qb + (size_t)(i / pairs) * DSV41_DIM + turned + 2 * (i % pairs);
        float re, im;
        dsv41_rotate_pair(__bfloat162float(pair[0]), __bfloat162float(pair[1]),
                          (float)positions[query] * freqs[i % pairs], false, re, im);
        pair[0] = __float2bfloat16(re);
        pair[1] = __float2bfloat16(im);
    }
    __syncthreads();

    const unsigned int warp = t / 32, lane = t % 32;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const int* ids = idx + (size_t)query * k;
    const unsigned int key_groups = dsv41_tc_steps(k) * (DSV41_KEYS_PER_STEP / 16);
    float* record = records + ((size_t)query * h + first) * DSV41_RECORD;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int group = warp + 8u * blockIdx.x; group < key_groups; group += 8u * gridDim.x) {
        const unsigned int j[2] = {group * 16u + g, group * 16u + g + 8u};
        const __nv_bfloat16* key[2] = {dsv41_tc_key(keys_a, keys_b, ids, j[0], k, split),
                                       dsv41_tc_key(keys_a, keys_b, ids, j[1], k, split)};
        // Two sets of eight heads.
        float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        #pragma unroll 2
        for (unsigned int chunk = 0; chunk < DSV41_DIM / 64; ++chunk) {
            const unsigned int at = chunk * 64u + 16u * c;
            uint4 w[2][2];
            #pragma unroll
            for (int which = 0; which < 2; ++which) {
                w[which][0] = zero;
                w[which][1] = zero;
                if (key[which]) {
                    const uint4* kp = (const uint4*)(key[which] + at);
                    w[which][0] = kp[0];
                    w[which][1] = kp[1];
                }
            }
            const unsigned int w0[8] = {w[0][0].x, w[0][0].y, w[0][0].z, w[0][0].w,
                                        w[0][1].x, w[0][1].y, w[0][1].z, w[0][1].w};
            const unsigned int w1[8] = {w[1][0].x, w[1][0].y, w[1][0].z, w[1][0].w,
                                        w[1][1].x, w[1][1].y, w[1][1].z, w[1][1].w};
            #pragma unroll
            for (int set = 0; set < 2; ++set) {
                const uint4* col = qs + (size_t)(set * 8 + g) * (DSV41_DIM / 8) + at / 8;
                const uint4 lo = col[0], hi = col[1];
                const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                #pragma unroll
                for (int step = 0; step < 4; ++step) {
                    const unsigned int a[4] = {w0[2 * step], w1[2 * step], w0[2 * step + 1],
                                               w1[2 * step + 1]};
                    dsv41_tc_mma(acc[set], a, xw[2 * step], xw[2 * step + 1]);
                }
            }
        }
        // The lane's sums: its two keys, by heads 2c and 2c + 1 of each set.
        #pragma unroll
        for (int set = 0; set < 2; ++set) {
            #pragma unroll
            for (int which = 0; which < 2; ++which) {
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const unsigned int head = set * 8 + 2u * c + p;
                    record[(size_t)head * DSV41_RECORD + j[which]] =
                        key[which] ? acc[set][which * 2 + p] * scale : -INFINITY;
                }
            }
        }
    }
}

// The output for a few queries, from the records the schedule left:
// out [m, h, 512] BF16, its last `rope` values turned back, as
// `dsv41_attn_output` leaves it. Heads by output values is the MMA's 16 by
// 8, and its 16 k-slots are 16 keys: a lane packs the weights of two heads
// for four keys and those four keys' values; the sums are rescaled before
// each step of 64 keys. A lane that fetches a key's values and a head's
// weights as it reaches them waits out one load behind another, forty times
// over. Here a (query, 16 heads) is sixteen
// blocks of 32 output values, and a block first stages what it will read:
// its 32 values of every key and its heads' weights as the BF16 the multiply
// takes, every load at once. Its four warps then take a tile of eight
// values each, with the same operands in the same k-slots, the same steps
// and the same rescaling between them.
//
// Grid (16, g), block 128, dynamic shared DSV41_FEW_SHARED bytes: a block
// takes every g-th of the `units` (query, 16 heads), one after another. The
// GPU holds one such block a multiprocessor, so a verify's queries as a
// grid of their own were not all on it at once, and what is queued behind
// this launch started only as it ended (`dsv41_attn_schedule`).
#define DSV41_FEW_VALUES 32
#define DSV41_FEW_THREADS 128
#define DSV41_FEW_SHARED \
    ((DSV41_MAX_KEYS * DSV41_FEW_VALUES + DSV41_GROUP * DSV41_MAX_KEYS) * 2)

extern "C" __global__ void __launch_bounds__(DSV41_FEW_THREADS) dsv41_attn_output_tc_few(
    const __nv_bfloat16* __restrict__ keys_a,
    const __nv_bfloat16* __restrict__ keys_b,
    const int* __restrict__ idx,
    const float* __restrict__ records,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int k,
    unsigned int split,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rope,
    unsigned int units) {
    atlas_pdl_enter();
    extern __shared__ __align__(16) unsigned char dsv41_few_shared[];
    // [slots, 32] key values, then [16, slots] weights.
    unsigned short* skeys = (unsigned short*)dsv41_few_shared;
    unsigned short* sweights = skeys + DSV41_MAX_KEYS * DSV41_FEW_VALUES;
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int base = blockIdx.x * DSV41_FEW_VALUES;
    const unsigned int t = threadIdx.x;
    const unsigned int steps = dsv41_tc_steps(k), slots = steps * DSV41_KEYS_PER_STEP;
    for (unsigned int unit = blockIdx.y; unit < units; unit += gridDim.y) {
    const unsigned int query = unit / groups, first = (unit % groups) * DSV41_GROUP;
    const int* ids = idx + (size_t)query * k;
    const float* heads = records + ((size_t)query * h + first) * DSV41_RECORD;
    const __nv_bfloat16* first_key = dsv41_tc_key(keys_a, keys_b, ids, 0, k, split);
    const __nv_bfloat16* any = first_key ? first_key : keys_a;
    // A thread a key: the block's 32 values of it.
    for (unsigned int j = t; j < slots; j += DSV41_FEW_THREADS) {
        const __nv_bfloat16* row = dsv41_tc_key(keys_a, keys_b, ids, j, k, split);
        const uint4* from = (const uint4*)((row ? row : any) + base);
        uint4* to = (uint4*)(skeys + (size_t)j * DSV41_FEW_VALUES);
        #pragma unroll
        for (int i = 0; i < DSV41_FEW_VALUES / 8; ++i) to[i] = from[i];
    }
    // A thread a (head, key): the weight as the multiply reads it.
    for (unsigned int i = t; i < DSV41_GROUP * slots; i += DSV41_FEW_THREADS) {
        const unsigned int head = i / slots, j = i % slots;
        sweights[(size_t)head * DSV41_MAX_KEYS + j] =
            __bfloat16_as_ushort(__float2bfloat16(heads[(size_t)head * DSV41_RECORD + j]));
    }
    __syncthreads();

    // A warp a tile of eight output values; a lane's two heads.
    const unsigned int warp = t / 32, lane = t % 32;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const float* record[2] = {heads + (size_t)g * DSV41_RECORD,
                              heads + (size_t)(g + 8u) * DSV41_RECORD};
    const unsigned short* weight[2] = {sweights + (size_t)g * DSV41_MAX_KEYS,
                                       sweights + (size_t)(g + 8u) * DSV41_MAX_KEYS};
    const unsigned int at = warp * 8u + g;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int chunk = 0; chunk < steps * (DSV41_KEYS_PER_STEP / 16); ++chunk) {
        if (chunk % (DSV41_KEYS_PER_STEP / 16) == 0) {
            const unsigned int step = chunk / (DSV41_KEYS_PER_STEP / 16);
            const float s0 = record[0][DSV41_RECORD_SHRINK + step];
            const float s1 = record[1][DSV41_RECORD_SHRINK + step];
            acc[0] *= s0;
            acc[1] *= s0;
            acc[2] *= s1;
            acc[3] *= s1;
        }
        // The lane's k-slots: keys 2c, 2c + 1, 2c + 8 and 2c + 9 of the 16.
        const unsigned int j0 = chunk * 16u + 2u * c;
        const unsigned int slot[4] = {j0, j0 + 1u, j0 + 8u, j0 + 9u};
        auto pair = [&](const unsigned short* of, unsigned int lo, unsigned int hi) {
            return (unsigned int)of[lo] | ((unsigned int)of[hi] << 16);
        };
        const unsigned int a[4] = {pair(weight[0], slot[0], slot[1]), pair(weight[1], slot[0], slot[1]),
                                   pair(weight[0], slot[2], slot[3]), pair(weight[1], slot[2], slot[3])};
        auto value = [&](unsigned int j) {
            return (unsigned int)skeys[(size_t)j * DSV41_FEW_VALUES + at];
        };
        const unsigned int b0 = value(slot[0]) | (value(slot[1]) << 16);
        const unsigned int b1 = value(slot[2]) | (value(slot[3]) << 16);
        dsv41_tc_mma(acc, a, b0, b1);
    }
    // The lane's sums: its two heads, by output values 2c and 2c + 1 of the
    // tile, which are a rotary pair where the output is turned.
    const unsigned int turned = DSV41_DIM - rope;
    const unsigned int d = base + warp * 8u + 2u * c;
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
        const float denominator = record[which][DSV41_RECORD_DENOMINATOR];
        __nv_bfloat16* o = out + ((size_t)query * h + first + g + 8u * which) * DSV41_DIM;
        float v0 = dsv41_bf16(acc[which * 2] / denominator);
        float v1 = dsv41_bf16(acc[which * 2 + 1] / denominator);
        if (d >= turned)
            dsv41_rotate_pair(v0, v1, (float)positions[query] * freqs[(d - turned) / 2], true, v0,
                              v1);
        o[d] = __float2bfloat16(v0);
        o[d + 1] = __float2bfloat16(v1);
    }
    // The next unit's keys and weights replace what the warps have read.
    __syncthreads();
    }
}

// Four 8-by-8 matrices of 16-bit values from shared memory as a tensor-core
// operand: lane `l` names row `l % 8` of matrix `l / 8`, 16 bytes, and gets
// of each matrix its row `l / 4`'s values `2 (l % 4)` and the one after
// (the first operand's rows), or turned, its rows `2 (l % 4)` and the one
// after at value `l / 4` (the second's columns).
static __device__ __forceinline__ void dsv41_attn_rows(unsigned int (&r)[4], const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned int)__cvta_generic_to_shared(row)));
}
static __device__ __forceinline__ void dsv41_attn_columns(unsigned int (&r)[4], const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned int)__cvta_generic_to_shared(row)));
}

// A batch's attention in one launch and one pass over a query's keys: the
// logits, the schedule and the output of `dsv41_attn_logits_tc`,
// `dsv41_attn_schedule` and `dsv41_attn_output_tc_few`, with their bytes.
//
// The schedule is a running one: a step of 64 keys needs the largest logit
// of the steps before it and nothing of the steps after. So a block takes a
// (query, 16 heads) a step at a time: the step's logits, its weights from
// them, and the step's keys into the output while they are still in the
// cache. As three launches a layer wrote a logit a (query, head, key) to
// memory, read and rewrote them for the weights and read them again for the
// output, 344 MB a rank at 8,192 rows each time, and read every key twice
// from memory: 12.3 ms of a layer's 75, and 9.9 so.
//
// Every sum is the one the three launches made, in their order: a logit's
// products by the same MMA steps (a warp takes eight heads of 16 keys where
// `dsv41_attn_logits_tc`'s took sixteen: a head's sums do not meet
// another's), a step's weights added in key order, the output's keys 16 at
// a time with the sum rescaled before each step.
//
// q [m, h, 512] BF16, sink [h] FP32, out [m, h, 512] BF16; h a multiple of
// 16. Grid (1, m * h / 16), block 256.
//
// What bounds it is the keys' bytes arriving, and after that the registers.
// The multiplies are free (every one made twice costs nothing) and so is
// the schedule; half the launch is the output's reads of the keys. A
// block's shared memory was 39 KB, two blocks a multiprocessor: the heads
// of q, a step's logits and weights, and 16 KB for a warp's 64 values of
// the 16 keys it is adding. Those 64 are staged a quarter at a time, in the
// memory the step's logits no longer need, which leaves 22.7 KB a block.
// That is room for four blocks, but four leave a thread 64 registers, and
// what does not fit is kept in memory and read back in the loops: three
// blocks, 85 registers, are quicker. 16 heads of 16,384 queries over 640
// keys take 17.1 ms with three blocks, 21.2 with four and 17.8 with two
// (which need no memory for registers at all), and a 32,000-token prefill
// 4.35 s where four took 4.50. (A step's 64 keys staged whole, one block a
// multiprocessor, took twice as long. A warp taking a key once for all
// sixteen heads in the logits, half their reads, was slower. Reading the
// four quarters' pieces ahead of the first changed nothing.)
extern "C" __global__ void __launch_bounds__(256, 3) dsv41_attn_batch_tc(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys_a,
    const __nv_bfloat16* __restrict__ keys_b,
    const int* __restrict__ idx,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int k,
    unsigned int split,
    float scale,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rope,
    unsigned char* __restrict__ codes,
    unsigned long long code_stride,
    unsigned char* __restrict__ sfs,
    unsigned long long sf_stride,
    unsigned int group_heads,
    unsigned int inner_pad) {
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int query = blockIdx.y / groups, first = (blockIdx.y % groups) * DSV41_GROUP;
    const unsigned int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const unsigned int g = lane >> 2, c = lane & 3u;
    __shared__ uint4 qs[DSV41_GROUP * DSV41_DIM / 8];
    // A step's logits, then its weights unrounded; the weights as BF16.
    __shared__ __align__(16) float logit[DSV41_GROUP][DSV41_KEYS_PER_STEP];
    __shared__ __align__(16) unsigned short weight[DSV41_GROUP][DSV41_KEYS_PER_STEP];
    // A head's running maximum, what the step rescales its sums by, and its
    // denominator so far.
    __shared__ float top[DSV41_GROUP];
    __shared__ float shrink[DSV41_GROUP];
    __shared__ float total[DSV41_GROUP];
    // A warp's 64 output values of the 16 keys it is adding: fetched 16
    // bytes at a time and read from here.
    // A quarter of a warp's 64 values of 16 keys, in the logits' memory:
    // the logits are done with when the keys are staged.
    unsigned short (*staged)[16][16] = (unsigned short (*)[16][16])logit;

    // The block's heads of q, turned to the query's position.
    const uint4* qv = (const uint4*)(q + ((size_t)query * h + first) * DSV41_DIM);
    for (unsigned int i = t; i < DSV41_GROUP * DSV41_DIM / 8; i += 256) qs[i] = qv[i];
    if (t < DSV41_GROUP) {
        top[t] = -1e30f;
        total[t] = 0.0f;
    }
    __syncthreads();
    __nv_bfloat16* qb = (__nv_bfloat16*)qs;
    const unsigned int pairs = rope / 2, turned = DSV41_DIM - rope;
    for (unsigned int i = t; i < DSV41_GROUP * pairs; i += 256) {
        __nv_bfloat16* pair = qb + (size_t)(i / pairs) * DSV41_DIM + turned + 2 * (i % pairs);
        float re, im;
        dsv41_rotate_pair(__bfloat162float(pair[0]), __bfloat162float(pair[1]),
                          (float)positions[query] * freqs[i % pairs], false, re, im);
        pair[0] = __float2bfloat16(re);
        pair[1] = __float2bfloat16(im);
    }
    __syncthreads();

    const int* ids = idx + (size_t)query * k;
    const unsigned int steps = dsv41_tc_steps(k);
    // A key a query does not have weighs nothing; its place is taken by the
    // query's first key, which every query has.
    const __nv_bfloat16* first_key = dsv41_tc_key(keys_a, keys_b, ids, 0, k, split);
    const __nv_bfloat16* any = first_key ? first_key : keys_a;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    // The output: eight tiles of eight values, 64 * warp on, of the lane's
    // two heads.
    float acc[8][4];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) acc[tile][0] = acc[tile][1] = acc[tile][2] = acc[tile][3] = 0.0f;

    for (unsigned int step = 0; step < steps; ++step) {
        // The step's logits: a pair of warps 16 keys, a warp eight heads.
        {
            const unsigned int base = (warp / 2u) * 16u, set = warp & 1u;
            const unsigned int j[2] = {base + g, base + g + 8u};
            const unsigned int at_key = step * DSV41_KEYS_PER_STEP;
            const __nv_bfloat16* key[2] = {
                dsv41_tc_key(keys_a, keys_b, ids, at_key + j[0], k, split),
                dsv41_tc_key(keys_a, keys_b, ids, at_key + j[1], k, split)};
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll 2
            for (unsigned int chunk = 0; chunk < DSV41_DIM / 64; ++chunk) {
                const unsigned int at = chunk * 64u + 16u * c;
                uint4 w[2][2];
                #pragma unroll
                for (int which = 0; which < 2; ++which) {
                    w[which][0] = zero;
                    w[which][1] = zero;
                    if (key[which]) {
                        const uint4* kp = (const uint4*)(key[which] + at);
                        w[which][0] = kp[0];
                        w[which][1] = kp[1];
                    }
                }
                const unsigned int w0[8] = {w[0][0].x, w[0][0].y, w[0][0].z, w[0][0].w,
                                            w[0][1].x, w[0][1].y, w[0][1].z, w[0][1].w};
                const unsigned int w1[8] = {w[1][0].x, w[1][0].y, w[1][0].z, w[1][0].w,
                                            w[1][1].x, w[1][1].y, w[1][1].z, w[1][1].w};
                const uint4* col = qs + (size_t)(set * 8 + g) * (DSV41_DIM / 8) + at / 8;
                const uint4 lo = col[0], hi = col[1];
                const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const unsigned int a[4] = {w0[2 * s], w1[2 * s], w0[2 * s + 1], w1[2 * s + 1]};
                    dsv41_tc_mma(sum, a, xw[2 * s], xw[2 * s + 1]);
                }
            }
            // The lane's sums: its two keys, by heads 2c and 2c + 1 of the set.
            #pragma unroll
            for (int which = 0; which < 2; ++which) {
                #pragma unroll
                for (int p = 0; p < 2; ++p)
                    logit[set * 8 + 2u * c + p][j[which]] =
                        key[which] ? sum[which * 2 + p] * scale : -INFINITY;
            }
        }
        __syncthreads();
        // A thread a head: the step's largest logit, the running maximum up
        // to and including it, and what it rescales the sums so far by.
        if (t < DSV41_GROUP) {
            float peak = -INFINITY;
            for (unsigned int i = 0; i < DSV41_KEYS_PER_STEP; ++i) peak = fmaxf(peak, logit[t][i]);
            const float before = top[t], now = fmaxf(before, peak);
            shrink[t] = expf(before - now);
            top[t] = now;
        }
        __syncthreads();
        for (unsigned int i = t; i < DSV41_GROUP * DSV41_KEYS_PER_STEP; i += 256) {
            const unsigned int head = i / DSV41_KEYS_PER_STEP, key = i % DSV41_KEYS_PER_STEP;
            const float e = expf(logit[head][key] - top[head]);
            logit[head][key] = e;
            weight[head][key] = __bfloat16_as_ushort(__float2bfloat16(e));
        }
        __syncthreads();
        // A thread a head: the step's weights added in key order, unrounded,
        // onto the denominator so far.
        if (t < DSV41_GROUP) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < DSV41_KEYS_PER_STEP; ++i) sum += logit[t][i];
            total[t] = total[t] * shrink[t] + sum;
        }
        // The staged keys take the logits' memory from here.
        __syncthreads();
        // The step's keys into the output, 16 at a time.
        {
            const float s0 = shrink[g], s1 = shrink[g + 8u];
            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                acc[tile][0] *= s0;
                acc[tile][1] *= s0;
                acc[tile][2] *= s1;
                acc[tile][3] *= s1;
            }
            for (unsigned int chunk = 0; chunk < DSV41_KEYS_PER_STEP / 16; ++chunk) {
                // The lane's k-slots: keys 2c, 2c + 1, 2c + 8 and 2c + 9 of the
                // 16, of heads g and g + 8: the weights' four matrices are
                // heads 0-7 and 8-15 of keys 0-7, then of keys 8-15.
                const unsigned int matrix = lane >> 3, line = lane & 7u;
                unsigned int a[4];
                dsv41_attn_rows(
                    a, &weight[(matrix & 1u) * 8u + line][chunk * 16u + (matrix >> 1) * 8u]);
                // The warp's 64 values of each key in four quarters of 16, a
                // quarter staged and added at a time: two lanes a key, eight
                // values each. (All 64 at once was 12 KB more of shared
                // memory a block, and two blocks filled a multiprocessor.)
                // A tile's sum is its own either way.
                const __nv_bfloat16* row = dsv41_tc_key(
                    keys_a, keys_b, ids, step * DSV41_KEYS_PER_STEP + chunk * 16u + (lane >> 1), k,
                    split);
                #pragma unroll
                for (unsigned int quarter = 0; quarter < 4u; ++quarter) {
                    {
                        const unsigned int of = lane >> 1, part = (lane & 1u) * 8u;
                        const uint4* from =
                            (const uint4*)((row ? row : any) + warp * 64u + quarter * 16u + part);
                        *(uint4*)&staged[warp][of][part] = from[0];
                    }
                    __syncwarp();
                    // The staged keys' four matrices: keys 0-7 and 8-15 of
                    // values 0-7, then of values 8-15; a lane's are its
                    // keys' at value g of each.
                    unsigned int b[4];
                    dsv41_attn_columns(
                        b, &staged[warp][(matrix & 1u) * 8u + line][(matrix >> 1) * 8u]);
                    dsv41_tc_mma(acc[quarter * 2u], a, b[0], b[1]);
                    dsv41_tc_mma(acc[quarter * 2u + 1u], a, b[2], b[3]);
                    // The next quarter, or the next 16 keys, replace these.
                    __syncwarp();
                }
            }
        }
        // The next step's logits and weights replace these.
        __syncthreads();
    }
    // A head's denominator: its weights, and the sink's under the last
    // running maximum.
    if (t < DSV41_GROUP) total[t] = total[t] + expf(sink[first + t] - top[t]);
    __syncthreads();
    // With `codes`, the output goes out as the projection after it reads
    // it: the E4M3 codes of the BF16 values below under one scale per 32,
    // by groups of `group_heads` heads, codes[group][query, group_heads *
    // 512] and the scales tiled (`dsv41_mx_quant` of out, a group at a
    // time). A head's 32 values are four tiles of the four lanes that share
    // its g.
    if (codes != nullptr) {
        #pragma unroll
        for (int which = 0; which < 2; ++which) {
            const unsigned int head = first + g + 8u * which;
            const unsigned int group = head / group_heads, in_group = head % group_heads;
            const float denominator = total[g + 8u * which];
            unsigned char* row =
                codes + group * code_stride + ((size_t)query * group_heads + in_group) * DSV41_DIM;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                float x[8];
                float amax = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int tile = half * 4 + i;
                    const unsigned int d = (warp * 8u + tile) * 8u + 2u * c;
                    float v0 = dsv41_bf16(acc[tile][which * 2] / denominator);
                    float v1 = dsv41_bf16(acc[tile][which * 2 + 1] / denominator);
                    if (d >= turned)
                        dsv41_rotate_pair(v0, v1, (float)positions[query] * freqs[(d - turned) / 2],
                                          true, v0, v1);
                    x[2 * i] = __bfloat162float(__float2bfloat16(v0));
                    x[2 * i + 1] = __bfloat162float(__float2bfloat16(v1));
                    amax = fmaxf(amax, fmaxf(fabsf(x[2 * i]), fabsf(x[2 * i + 1])));
                }
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                const float scale = dsv41_fp8_scale(amax);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const unsigned int d = (warp * 8u + half * 4 + i) * 8u + 2u * c;
                    unsigned int pair = 0u;
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float scaled = fminf(fmaxf(x[2 * i + j] / scale, -448.0f), 448.0f);
                        pair |= (unsigned int)(unsigned char)__nv_cvt_float_to_fp8(
                                    dsv41_round_e4m3(scaled), __NV_SATFINITE, __NV_E4M3)
                                << (8 * j);
                    }
                    *(unsigned short*)(row + d) = (unsigned short)pair;
                }
                if (c == 0u) {
                    const unsigned int blocks = group_heads * (DSV41_DIM / 32u);
                    const unsigned int block = in_group * (DSV41_DIM / 32u) + warp * 2u + half;
                    unsigned char* sf = sfs + group * sf_stride;
                    sf[mx_sf_offset(query, block, inner_pad)] =
                        (unsigned char)(dsv41_exponent(scale) + 127);
                    if (block + 1u == blocks) mx_sf_zero_pad(sf, query, blocks, inner_pad);
                }
            }
        }
        return;
    }
    // The lane's sums: its two heads, by output values 2c and 2c + 1 of each
    // tile, which are a rotary pair where the output is turned.
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
        const float denominator = total[g + 8u * which];
        __nv_bfloat16* o = out + ((size_t)query * h + first + g + 8u * which) * DSV41_DIM;
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const unsigned int d = (warp * 8u + tile) * 8u + 2u * c;
            float v0 = dsv41_bf16(acc[tile][which * 2] / denominator);
            float v1 = dsv41_bf16(acc[tile][which * 2 + 1] / denominator);
            if (d >= turned)
                dsv41_rotate_pair(v0, v1, (float)positions[query] * freqs[(d - turned) / 2], true,
                                  v0, v1);
            o[d] = __float2bfloat16(v0);
            o[d + 1] = __float2bfloat16(v1);
        }
    }
}
