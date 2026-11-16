// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: which keys each query of a batch attends to, without
// leaving the device.
//
//   dsv41_index_score_tc  the indexer's scores (`dsv41_index.cu`) for a
//                         batch of queries, on tensor cores.
//   dsv41_index_choose    the `topk` best compressed positions of each
//                         query, earliest first among equals, ascending.
//   dsv41_attn_index      a query's index row: its window, then its chosen.
//
// A row at a time these are a launch, a wait and a read back for every
// query and layer, and the choice is the host's. A prefill has thousands of
// queries a layer, each scoring every compressed position before it.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_CHOOSE_THREADS 256
#define DSV41_WINDOW 128
// Queries a block scores together, and the index heads and head width the
// block has room for.
#define DSV41_SCORE_ROWS 4
#define DSV41_SCORE_HEADS 32
#define DSV41_SCORE_WIDTH 128
#define DSV41_SCORE_KEYS 128

static __device__ __forceinline__ float dsv41_choose_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// One m16n8k16 BF16 MMA step into `acc` (FP32).
static __device__ __forceinline__ void dsv41_score_mma(
    float (&acc)[4], const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// partial[y, t] = sum_h BF16(relu(BF16(q[y, h] . key[t])) * w[y, h]) in FP32,
// the heads in order: what `dsv41_index_score` leaves, with each dot product
// an FP32 sum on tensor cores. Both operands are FP4 values under
// power-of-two scales, so the sum is the exact one unless a scale block of
// the 128 values is some 2^11 times smaller than another, and then it
// differs below what the BF16 rounding keeps but for a tie.
//
// q [rows, heads, 128] BF16, weights [rows, heads] BF16, keys [groups, 128]
// BF16, partial [rows, stride] FP32. Query y is the token at position + y and
// sees (position + y + 1) / ratio keys; the rest of its row is not written.
// heads is even and at most 32.
//
// A block is 4 queries by 128 keys. It stages its queries' heads in shared
// memory; a warp holds 16 keys in registers and takes the heads two at a
// time, 4 queries by 2 heads being the 8 columns of an MMA. A lane loads 16
// consecutive values of its keys and of its column, and feeds them to the
// same k-slots of both operands (`dsv41_tc.cu`). Grid (ceil(stride / 128),
// ceil(rows / 4)), block 256.
extern "C" __global__ void __launch_bounds__(256) dsv41_index_score_tc(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ weights,
    float* __restrict__ partial,
    unsigned int heads,
    unsigned int stride,
    float weight_scale,
    unsigned int position,
    unsigned int ratio,
    unsigned int rows) {
    __shared__ uint4 qs[DSV41_SCORE_ROWS * DSV41_SCORE_HEADS * DSV41_SCORE_WIDTH / 8];
    __shared__ float ws[DSV41_SCORE_ROWS][DSV41_SCORE_HEADS];
    const unsigned int tid = threadIdx.x, row0 = blockIdx.y * DSV41_SCORE_ROWS;
    const unsigned int per_row = heads * (DSV41_SCORE_WIDTH / 8);
    const uint4* q4 = (const uint4*)(q + (size_t)row0 * heads * DSV41_SCORE_WIDTH);
    for (unsigned int i = tid; i < DSV41_SCORE_ROWS * per_row; i += 256)
        qs[i] = row0 + i / per_row < rows ? q4[i] : make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int i = tid; i < DSV41_SCORE_ROWS * heads; i += 256) {
        const unsigned int r = i / heads, h = i % heads;
        ws[r][h] = row0 + r < rows
                       ? dsv41_choose_bf16(
                             __bfloat162float(weights[(size_t)(row0 + r) * heads + h]) * weight_scale)
                       : 0.0f;
    }
    __syncthreads();

    const unsigned int warp = tid / 32, lane = tid % 32;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int key0 = blockIdx.x * DSV41_SCORE_KEYS + warp * 16u;
    // The lane's keys, g and g + 8 of the warp's 16: values 16c.. of each
    // half of the 128, as the a operand of eight steps.
    unsigned int a[8][4];
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        uint4 lo[2], hi[2];
        #pragma unroll
        for (int which = 0; which < 2; ++which) {
            const unsigned int key = key0 + g + 8u * which;
            const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
            lo[which] = zero;
            hi[which] = zero;
            if (key < stride) {
                const uint4* kp = (const uint4*)(keys + (size_t)key * DSV41_SCORE_WIDTH +
                                                 half * 64 + 16u * c);
                lo[which] = kp[0];
                hi[which] = kp[1];
            }
        }
        const unsigned int w0[8] = {lo[0].x, lo[0].y, lo[0].z, lo[0].w,
                                    hi[0].x, hi[0].y, hi[0].z, hi[0].w};
        const unsigned int w1[8] = {lo[1].x, lo[1].y, lo[1].z, lo[1].w,
                                    hi[1].x, hi[1].y, hi[1].z, hi[1].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            a[half * 4 + j][0] = w0[2 * j];
            a[half * 4 + j][1] = w1[2 * j];
            a[half * 4 + j][2] = w0[2 * j + 1];
            a[half * 4 + j][3] = w1[2 * j + 1];
        }
    }
    // The lane's column of the b operand is query g / 2, head 2 * pair +
    // g % 2; its sums are keys g and g + 8 for query c, heads 2 * pair and
    // the one after.
    float total[2] = {0.0f, 0.0f};
    for (unsigned int pair = 0; pair < heads / 2u; ++pair) {
        const uint4* col = qs + (size_t)(g >> 1) * per_row +
                           (size_t)(2u * pair + (g & 1u)) * (DSV41_SCORE_WIDTH / 8);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint4 lo = col[half * 8 + 2u * c], hi = col[half * 8 + 2u * c + 1u];
            const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                dsv41_score_mma(acc, a[half * 4 + j], xw[2 * j], xw[2 * j + 1]);
        }
        #pragma unroll
        for (int which = 0; which < 2; ++which) {
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const float head = fmaxf(dsv41_choose_bf16(acc[which * 2 + p]), 0.0f);
                total[which] += dsv41_choose_bf16(head * ws[c][2u * pair + p]);
            }
        }
    }
    const unsigned int row = row0 + c;
    if (row >= rows) return;
    const unsigned int visible = (position + row + 1u) / ratio;
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
        const unsigned int key = key0 + g + 8u * which;
        if (key < visible && key < stride) partial[(size_t)row * stride + key] = total[which];
    }
}

// An index score as 16 bits that order as the score does.
static __device__ __forceinline__ unsigned int dsv41_score_key(float partial) {
    unsigned int u = __bfloat16_as_ushort(__float2bfloat16(partial));
    if (u == 0x8000u) u = 0u;
    return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}

// How many of the bins of `hist`, from the top, it takes to hold `left`
// values: the bin the cut falls in, and how many of that bin's are taken.
static __device__ __forceinline__ void dsv41_cut(
    const unsigned int* hist, unsigned int left, unsigned int* cut) {
    unsigned int bin = 255u;
    while (hist[bin] < left) {
        left -= hist[bin];
        --bin;
    }
    cut[0] = bin;
    cut[1] = left;
}

// chosen[y] = the `topk` best positions of query y's scores, ascending;
// among equal scores the earliest are taken, as the reference's top-k does.
// A query that sees no more than `topk` positions takes them all, and the
// rest of its row is -1. A score is BF16(partial); partial [.., stride]
// FP32, row y of the grid at partial + y * stride (not read for a query that
// takes all). Query y is the token at position + y.
//
// A block a query: the 16-bit keys are cut by two histograms of 8 bits, and
// the positions kept are written in order by 256 threads that each own a run
// of positions. Grid (1, queries), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_CHOOSE_THREADS) dsv41_index_choose(
    const float* __restrict__ partial,
    int* __restrict__ chosen,
    unsigned int stride,
    unsigned int topk,
    unsigned int position,
    unsigned int ratio) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int cut[2];
    __shared__ unsigned int above[DSV41_CHOOSE_THREADS], equal[DSV41_CHOOSE_THREADS];
    const unsigned int tid = threadIdx.x, y = blockIdx.y;
    const unsigned int n = (position + y + 1u) / ratio;
    int* out = chosen + (size_t)y * topk;
    if (n <= topk) {
        for (unsigned int i = tid; i < topk; i += DSV41_CHOOSE_THREADS) out[i] = i < n ? (int)i : -1;
        return;
    }
    const float* s = partial + (size_t)y * stride;
    hist[tid] = 0u;
    __syncthreads();
    for (unsigned int i = tid; i < n; i += DSV41_CHOOSE_THREADS)
        atomicAdd(&hist[dsv41_score_key(s[i]) >> 8], 1u);
    __syncthreads();
    if (tid == 0) dsv41_cut(hist, topk, cut);
    __syncthreads();
    const unsigned int high = cut[0];
    hist[tid] = 0u;
    __syncthreads();
    for (unsigned int i = tid; i < n; i += DSV41_CHOOSE_THREADS) {
        const unsigned int key = dsv41_score_key(s[i]);
        if ((key >> 8) == high) atomicAdd(&hist[key & 255u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        dsv41_cut(hist, cut[1], cut);
        cut[0] |= high << 8;
    }
    __syncthreads();
    // Every position above the cut is kept, and the first `need` at it.
    const unsigned int at_cut = cut[0], need = cut[1];
    const unsigned int run = (n + DSV41_CHOOSE_THREADS - 1u) / DSV41_CHOOSE_THREADS;
    const unsigned int first = min(tid * run, n), end = min(first + run, n);
    unsigned int mine_above = 0u, mine_equal = 0u;
    for (unsigned int i = first; i < end; ++i) {
        const unsigned int key = dsv41_score_key(s[i]);
        mine_above += key > at_cut;
        mine_equal += key == at_cut;
    }
    above[tid] = mine_above;
    equal[tid] = mine_equal;
    __syncthreads();
    unsigned int before_above = 0u, before_equal = 0u;
    for (unsigned int t = 0; t < tid; ++t) {
        before_above += above[t];
        before_equal += equal[t];
    }
    unsigned int at = before_above + min(before_equal, need);
    for (unsigned int i = first; i < end; ++i) {
        const unsigned int key = dsv41_score_key(s[i]);
        if (key > at_cut) {
            out[at++] = (int)i;
        } else if (key == at_cut) {
            if (before_equal < need) out[at++] = (int)i;
            ++before_equal;
        }
    }
}

// idx[r] = the keys query r of a batch attends to, [rows, keys] i32: its
// window, then the compressed positions chosen for it. The query at
// position + r reads the positions of its window in order, of which
// position q is entry q - position + 128 of the batch's unrolled window
// keys, then -1 up to 128 entries. With `ring`, for a query alone, position
// q is where the window's ring holds it, slot q % 128. `chosen` is [rows,
// topk], and null on a layer that attends over its window only. One thread
// an entry.
extern "C" __global__ void dsv41_attn_index(
    const int* __restrict__ chosen,
    int* __restrict__ idx,
    unsigned int keys,
    unsigned int topk,
    unsigned int position,
    unsigned int rows,
    unsigned int ring) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * keys) return;
    const unsigned int r = (unsigned int)(i / keys), j = (unsigned int)(i % keys);
    const unsigned int at = position + r;
    int entry = -1;
    if (j < DSV41_WINDOW) {
        const unsigned int first = at + 1u > DSV41_WINDOW ? at + 1u - DSV41_WINDOW : 0u;
        if (first + j <= at)
            entry = ring ? (int)((first + j) % DSV41_WINDOW)
                         : (int)(first + j + DSV41_WINDOW - position);
    } else if (chosen) {
        entry = chosen[(size_t)r * topk + (j - DSV41_WINDOW)];
    }
    idx[i] = entry;
}
