// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Per-row bodies of the GLM semantic-index cache update (glm_indexer.cu's
// layernorm, raw tail write and four-token pool finalize), on `nth` threads
// `tid` of a CTA. glm_indexer.cu runs each on its own launch (`tid` =
// threadIdx.x, `nth` = blockDim.x = 128, the CTA's barrier); glm_mla_fuse.cu
// runs them on 128-thread groups of one CTA with a named barrier per group.
// Same statements in the same order either way, so the same bytes.
#include <cuda_bf16.h>
#include <math.h>

#include "glm_barrier.cuh"

// LayerNorm of one `dim`-wide row `x` in place; `reduce` holds `nth` floats
// (`nth` a power of two).
template <class Bar>
__device__ __forceinline__ void glm_index_layernorm_row(
    __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias, unsigned int dim,
    float eps, float* reduce, unsigned int tid, unsigned int nth, Bar bar) {
    float sum = 0.0f;
    for (unsigned int d = tid; d < dim; d += nth) {
        sum += __bfloat162float(x[d]);
    }
    reduce[tid] = sum;
    bar();
    for (unsigned int stride = nth / 2; stride; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        bar();
    }
    const float mean = reduce[0] / (float)dim;
    // Every thread must hold the mean before thread 0 reuses reduce[0] for
    // its squared sum: without this barrier a warp scheduled late reads the
    // squared sum as the row sum and normalizes its lanes with a wrong mean.
    bar();
    float sq = 0.0f;
    for (unsigned int d = tid; d < dim; d += nth) {
        const float centered = __bfloat162float(x[d]) - mean;
        sq += centered * centered;
    }
    reduce[tid] = sq;
    bar();
    for (unsigned int stride = nth / 2; stride; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        bar();
    }
    const float inv_std = rsqrtf(reduce[0] / (float)dim + eps);
    for (unsigned int d = tid; d < dim; d += nth) {
        const float y = (__bfloat162float(x[d]) - mean) * inv_std
                      * __bfloat162float(weight[d]) + __bfloat162float(bias[d]);
        x[d] = __float2bfloat16(y);
    }
}

// Raw tails are indexed by physical block, or through `tail_map` (block ->
// lent tail slot) when the cache slot-maps them; an unmapped block is skipped.
__device__ __forceinline__ bool glm_index_tail_index(
    const unsigned int* __restrict__ tail_map,
    unsigned int physical_block,
    unsigned int* tail_index) {
    *tail_index = tail_map ? tail_map[physical_block] : physical_block;
    return *tail_index != 0xFFFFFFFFu;
}

// Token `token`'s raw key and gate into the tail of the physical block that
// owns it.
__device__ __forceinline__ void glm_index_tail_write_row(
    const __nv_bfloat16* keys, const __nv_bfloat16* gates, __nv_bfloat16* tail,
    const unsigned int* tail_map, const long long* slots, unsigned int token,
    unsigned int block_size, unsigned int head_dim, unsigned long long tail_block_stride_bytes,
    unsigned int tid, unsigned int nth) {
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned int physical_block = (unsigned int)(slot / block_size);
    const unsigned int raw_offset = (unsigned int)(slot % block_size);
    unsigned int tail_index;
    if (!glm_index_tail_index(tail_map, physical_block, &tail_index)) return;
    __nv_bfloat16* block_tail = (__nv_bfloat16*)((char*)tail
        + (unsigned long long)tail_index * tail_block_stride_bytes);
    __nv_bfloat16* key_dst = block_tail + (unsigned long long)raw_offset * head_dim;
    __nv_bfloat16* gate_dst = block_tail
        + (unsigned long long)block_size * head_dim
        + (unsigned long long)raw_offset * head_dim;
    const __nv_bfloat16* key_src = keys + (unsigned long long)token * head_dim;
    const __nv_bfloat16* gate_src = gates + (unsigned long long)token * head_dim;
    for (unsigned int d = tid; d < head_dim; d += nth) {
        key_dst[d] = key_src[d];
        gate_dst[d] = gate_src[d];
    }
}

// The pooled key of token `token` when it ends a pool, from the four raw
// keys and gates its block's tail holds. `tail` may have been written by the
// same launch (glm_mla_fuse.cu): plain loads.
__device__ __forceinline__ void glm_index_kpool_finalize_row(
    const __nv_bfloat16* tail, const unsigned int* tail_map, const __nv_bfloat16* ape,
    __nv_bfloat16* cache, const long long* slots, unsigned int token, unsigned int block_size,
    unsigned int pool_size, unsigned int head_dim, unsigned long long tail_block_stride_bytes,
    unsigned long long values_block_stride_bytes, unsigned int tid, unsigned int nth) {
    if (pool_size != 4) return;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned int physical_block = (unsigned int)(slot / block_size);
    const unsigned int raw_offset = (unsigned int)(slot % block_size);
    if (raw_offset % pool_size != pool_size - 1) return;
    unsigned int tail_index;
    if (!glm_index_tail_index(tail_map, physical_block, &tail_index)) return;
    const __nv_bfloat16* block_tail = (const __nv_bfloat16*)((const char*)tail
        + (unsigned long long)tail_index * tail_block_stride_bytes);
    const unsigned int first = raw_offset - (pool_size - 1);
    const __nv_bfloat16* tail_keys = block_tail
        + (unsigned long long)first * head_dim;
    const __nv_bfloat16* tail_gates = block_tail
        + (unsigned long long)block_size * head_dim
        + (unsigned long long)first * head_dim;
    __nv_bfloat16* dst = (__nv_bfloat16*)((char*)cache
        + (unsigned long long)physical_block * values_block_stride_bytes)
        + (unsigned long long)(raw_offset / pool_size) * head_dim;
    for (unsigned int d = tid; d < head_dim; d += nth) {
        float score[4];
        float max_score = -INFINITY;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            score[i] = __bfloat162float(tail_gates[(unsigned long long)i * head_dim + d])
                     + __bfloat162float(ape[(unsigned long long)i * head_dim + d]);
            max_score = fmaxf(max_score, score[i]);
        }
        float denom = 0.0f;
        float weighted = 0.0f;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            const float p = expf(score[i] - max_score);
            denom += p;
            weighted += p * __bfloat162float(
                tail_keys[(unsigned long long)i * head_dim + d]);
        }
        dst[d] = __float2bfloat16(weighted / denom);
    }
}
