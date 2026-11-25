// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: what a token's attention does between its first two
// projections and the heads, in one launch.
//
// The low-rank query and the window key leave their projections as BF16
// (`dsv41_fp8_gemv_two`). The query is normed, and its FP8 form is what the
// heads' projection and the indexer read (they round it on the way in). The
// key is normed, turned to the token's position and written into the
// window in FP8. That was five launches of a block each, and at one token a
// launch costs more than these 1,792 values.
//
// A block of 1,024 threads a token. Each norm is `rms_norm_vanilla` as it runs on
// its own block (1,024 threads for the query's 1,280 values, 512 for the
// key's 512): the same threads add the same pairs, the warps reduce by the
// same shuffles, and the warps' sums meet the same way. The turn is
// `dsv41_rotate` and the rounding `dsv41_fp8_quant_dequant`
// (`dsv41_common.cuh`). So the bytes are those of the five launches.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_FRONT_THREADS 1024
#define DSV41_FRONT_WARPS (DSV41_FRONT_THREADS / 32)
#define DSV41_KEY 512

// The sum of squares of `n` BF16 values as a block of `threads` threads of
// `rms_norm_vanilla` adds it: thread `tid` (of the first `threads`) takes
// every `threads`-th pair, a warp reduces by shuffles into `warp_sums`, and
// after a barrier the first warp reduces those. Every thread calls both
// halves; the total is in warp_sums[0] after a second barrier.
__device__ __forceinline__ void dsv41_squares_warps(
    const __nv_bfloat16* x, unsigned int n, unsigned int threads, float* warp_sums) {
    const unsigned int tid = threadIdx.x;
    const unsigned int* x32 = (const unsigned int*)x;
    float sum_sq = 0.0f;
    if (tid < threads) {
        for (unsigned int i = tid; i < n / 2; i += threads) {
            float v0, v1;
            dsv41_unpack2(x32[i], v0, v1);
            sum_sq += v0 * v0 + v1 * v1;
        }
    }
    #pragma unroll
    for (unsigned int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, o);
    if (tid % 32 == 0) warp_sums[tid / 32] = sum_sq;
}

__device__ __forceinline__ void dsv41_squares_block(unsigned int threads, float* warp_sums) {
    const unsigned int tid = threadIdx.x;
    if (tid >= 32) return;
    float total = tid < threads / 32 ? warp_sums[tid] : 0.0f;
    #pragma unroll
    for (unsigned int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
    if (tid == 0) warp_sums[0] = total;
}

// q_low [T, q_rank] and key [T, 512] BF16, as projected. q_normed [T, q_rank]
// = fp8(the norm of q_low under q_weight): every projection that reads the
// query reads it in FP8, and a launch of its own to round it was 7 us of a
// decode step in every layer. slot [T, 512] = fp8(turn(norm of key under
// key_weight, position[T])), the turn over the last `rope` values by freqs
// [rope / 2] radians a position. q_rank is a multiple of 64, so that whole
// warps hold its pairs, and at most 2,048. Grid (T).
extern "C" __global__ void __launch_bounds__(DSV41_FRONT_THREADS) dsv41_attn_front(
    const __nv_bfloat16* __restrict__ q_low,
    const __nv_bfloat16* __restrict__ q_weight,
    __nv_bfloat16* __restrict__ q_normed,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ key_weight,
    __nv_bfloat16* __restrict__ slot,
    const unsigned int* __restrict__ position,
    const float* __restrict__ freqs,
    unsigned int q_rank,
    unsigned int rope,
    float eps) {
    atlas_pdl_enter();
    const unsigned int tid = threadIdx.x, t = blockIdx.x;
    q_low += (size_t)t * q_rank;
    q_normed += (size_t)t * q_rank;
    key += (size_t)t * DSV41_KEY;
    slot += (size_t)t * DSV41_KEY;
    __shared__ float q_squares[DSV41_FRONT_WARPS];
    __shared__ float key_squares[DSV41_FRONT_WARPS];
    __shared__ float normed[DSV41_KEY];

    dsv41_squares_warps(q_low, q_rank, DSV41_FRONT_THREADS, q_squares);
    dsv41_squares_warps(key, DSV41_KEY, DSV41_KEY, key_squares);
    __syncthreads();
    dsv41_squares_block(DSV41_FRONT_THREADS, q_squares);
    dsv41_squares_block(DSV41_KEY, key_squares);
    __syncthreads();

    const float q_rms = rsqrtf(q_squares[0] / (float)q_rank + eps);
    const unsigned int* x32 = (const unsigned int*)q_low;
    const unsigned int* w32 = (const unsigned int*)q_weight;
    unsigned int* out32 = (unsigned int*)q_normed;
    for (unsigned int i = tid; i < q_rank / 2; i += DSV41_FRONT_THREADS) {
        float x0, x1, w0, w1;
        dsv41_unpack2(x32[i], x0, x1);
        dsv41_unpack2(w32[i], w0, w1);
        // Normed and rounded to BF16, then to FP8: a scale block's 32
        // values are the pairs of 16 neighbouring lanes.
        float v0, v1;
        dsv41_unpack2(dsv41_pack2(x0 * q_rms * w0, x1 * q_rms * w1), v0, v1);
        float amax = fmaxf(fabsf(v0), fabsf(v1));
        #pragma unroll
        for (unsigned int o = 8; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float scale = dsv41_fp8_scale(amax);
        out32[i] = dsv41_pack2(dsv41_fp8_value(v0, scale), dsv41_fp8_value(v1, scale));
    }
    const float key_rms = rsqrtf(key_squares[0] / (float)DSV41_KEY + eps);
    if (tid < DSV41_KEY / 2) {
        float x0, x1, w0, w1;
        dsv41_unpack2(((const unsigned int*)key)[tid], x0, x1);
        dsv41_unpack2(((const unsigned int*)key_weight)[tid], w0, w1);
        normed[2 * tid] = dsv41_bf16(x0 * key_rms * w0);
        normed[2 * tid + 1] = dsv41_bf16(x1 * key_rms * w1);
    }
    __syncthreads();
    if (tid < rope / 2) {
        float* pair = normed + (DSV41_KEY - rope) + 2 * tid;
        dsv41_rotate_pair(pair[0], pair[1], (float)position[t] * freqs[tid], false, pair[0],
                          pair[1]);
    }
    __syncthreads();
    if (tid < DSV41_KEY / 32) dsv41_fp8_block(normed + tid * 32, slot + tid * 32);
}
