// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash attention over gathered keys, for a batch of queries,
// on tensor cores: the first and third launches of the shared form of
// `dsv41_attn.cu`, with `dsv41_attn_schedule` between them as there.
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
// A block is one query and 16 heads. Grid (1, m * h / 16), block 256.
//
// A few queries leave most of the GPU idle that way, so theirs are spread:
// the logits over the blocks of a grid row, each taking every
// `gridDim.x`-th run of keys, and the output over sixteen blocks of 32
// values (`dsv41_attn_output_tc_few`). No sum is cut, so the bytes are the
// same: a decode step, a verify's rows and a prefill's batch are one
// arithmetic.

#include <cuda_bf16.h>

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

static __device__ __forceinline__ unsigned int dsv41_tc_pair(float lo, float hi) {
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(lo)) |
           ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
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

// out [m, h, 512] BF16 from the records the schedule left and the keys, its
// last `rope` values turned back, as `dsv41_attn_output` leaves it.
//
// Heads by output values is the MMA's 16 by 8, and its 16 k-slots are 16
// keys: a lane packs the weights of two heads for four keys, and, for each
// of the 64 output values its warp holds, those four keys' values. The sums
// are rescaled before each step of 64 keys.
extern "C" __global__ void __launch_bounds__(256) dsv41_attn_output_tc(
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
    unsigned int rope) {
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int query = blockIdx.y / groups, first = (blockIdx.y % groups) * DSV41_GROUP;
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int steps = dsv41_tc_steps(k);
    const int* ids = idx + (size_t)query * k;
    // The lane's two heads.
    const float* record[2] = {records + ((size_t)query * h + first + g) * DSV41_RECORD,
                              records + ((size_t)query * h + first + g + 8u) * DSV41_RECORD};
    // A key a query does not have weighs nothing; its place is taken by the
    // query's first key, which every query has.
    const __nv_bfloat16* first_key = dsv41_tc_key(keys_a, keys_b, ids, 0, k, split);
    const __nv_bfloat16* any = first_key ? first_key : keys_a;
    // Eight tiles of eight output values: 64 * warp on.
    float acc[8][4];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) acc[tile][0] = acc[tile][1] = acc[tile][2] = acc[tile][3] = 0.0f;
    for (unsigned int chunk = 0; chunk < steps * (DSV41_KEYS_PER_STEP / 16); ++chunk) {
        if (chunk % (DSV41_KEYS_PER_STEP / 16) == 0) {
            const unsigned int step = chunk / (DSV41_KEYS_PER_STEP / 16);
            const float s0 = record[0][DSV41_RECORD_SHRINK + step];
            const float s1 = record[1][DSV41_RECORD_SHRINK + step];
            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                acc[tile][0] *= s0;
                acc[tile][1] *= s0;
                acc[tile][2] *= s1;
                acc[tile][3] *= s1;
            }
        }
        // The lane's k-slots: keys 2c, 2c + 1, 2c + 8 and 2c + 9 of the 16.
        const unsigned int j0 = chunk * 16u + 2u * c;
        const unsigned int slot[4] = {j0, j0 + 1u, j0 + 8u, j0 + 9u};
        const unsigned int a[4] = {dsv41_tc_pair(record[0][slot[0]], record[0][slot[1]]),
                                   dsv41_tc_pair(record[1][slot[0]], record[1][slot[1]]),
                                   dsv41_tc_pair(record[0][slot[2]], record[0][slot[3]]),
                                   dsv41_tc_pair(record[1][slot[2]], record[1][slot[3]])};
        const unsigned short* key[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat16* row = dsv41_tc_key(keys_a, keys_b, ids, slot[i], k, split);
            key[i] = (const unsigned short*)(row ? row : any);
        }
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const unsigned int d = (warp * 8u + tile) * 8u + g;
            const unsigned int b0 = (unsigned int)key[0][d] | ((unsigned int)key[1][d] << 16);
            const unsigned int b1 = (unsigned int)key[2][d] | ((unsigned int)key[3][d] << 16);
            dsv41_tc_mma(acc[tile], a, b0, b1);
        }
    }
    // The lane's sums: its two heads, by output values 2c and 2c + 1 of each
    // tile, which are a rotary pair where the output is turned.
    const unsigned int turned = DSV41_DIM - rope;
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
        const float denominator = record[which][DSV41_RECORD_DENOMINATOR];
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

// `dsv41_attn_output_tc` for a few queries, with its bytes. That kernel
// fetches a key's values and a head's weights as it reaches them, one load
// behind another, which a batch hides behind its other blocks and a query
// alone waits out: forty times over. Here a (query, 16 heads) is sixteen
// blocks of 32 output values, and a block first stages what it will read:
// its 32 values of every key and its heads' weights as the BF16 the multiply
// takes, every load at once. Its four warps then take a tile of eight
// values each, with the same operands in the same k-slots, the same steps
// and the same rescaling between them.
//
// Grid (16, m * h / 16), block 128, dynamic shared DSV41_FEW_SHARED bytes.
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
    unsigned int rope) {
    atlas_pdl_enter();
    extern __shared__ __align__(16) unsigned char dsv41_few_shared[];
    // [slots, 32] key values, then [16, slots] weights.
    unsigned short* skeys = (unsigned short*)dsv41_few_shared;
    unsigned short* sweights = skeys + DSV41_MAX_KEYS * DSV41_FEW_VALUES;
    const unsigned int groups = h / DSV41_GROUP;
    const unsigned int query = blockIdx.y / groups, first = (blockIdx.y % groups) * DSV41_GROUP;
    const unsigned int base = blockIdx.x * DSV41_FEW_VALUES;
    const unsigned int t = threadIdx.x;
    const unsigned int steps = dsv41_tc_steps(k), slots = steps * DSV41_KEYS_PER_STEP;
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
}
