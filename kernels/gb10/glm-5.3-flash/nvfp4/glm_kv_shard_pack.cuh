// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// The merge form's row packing (glm_kv_shard.cu), shared with the fused MLA
// front (glm_mla_fuse.cu), which localizes a causal owner's IDs beside its
// other per-row work.
#include <cuda_runtime.h>
#include <cstddef>

#include "glm_barrier.cuh"

// Pack one row's selected ids (`in == nullptr`: the causal ids of row `r`,
// tokens [0, causal_start + r + 1)) that `keep(token, &value)` keeps, as
// `value`, to the front of `out`'s row in selected order (a stable
// partition); the rest of the row is -1 and the number kept goes to
// `count[r]`. Threads `t` < 256 of the CTA, `bar` a barrier over them;
// thread `t` owns the `ceil(width / 256)` consecutive ids from
// `t * ceil(width / 256)`. `out` must not alias `in`. `s_kept` is 256 words
// of shared memory, reused only after a barrier.
#define GLM_KV_SHARD_COMPACT_CHUNK 16u
template <class Keep, class Bar>
__device__ __forceinline__ void glm_kv_shard_pack_row(
    const int* __restrict__ in, int* __restrict__ out, unsigned int* __restrict__ count,
    unsigned int r, unsigned int width, unsigned int causal_start, unsigned int* s_kept,
    Keep keep, Bar bar) {
    const unsigned int t = threadIdx.x;
    const unsigned int chunk = (width + 255u) / 256u;
    const unsigned int begin = min(t * chunk, width);
    const unsigned int end = min(begin + chunk, width);
    const size_t row = (size_t)r * width;
    int value[GLM_KV_SHARD_COMPACT_CHUNK];
    unsigned int kept = 0;
    for (unsigned int c = begin; c < end; ++c) {
        const int token = in != nullptr ? in[row + c] : (c < causal_start + r + 1u ? (int)c : -1);
        if (token >= 0 && keep((unsigned int)token, value[kept])) ++kept;
    }
    s_kept[t] = kept;
    bar();
    unsigned int before = 0, total = 0;
    for (unsigned int i = 0; i < 256u; ++i) {
        if (i == t) before = total;
        total += s_kept[i];
    }
    for (unsigned int k = 0; k < kept; ++k) out[row + before + k] = value[k];
    // The ids this thread dropped land after every kept id, in order.
    const unsigned int dropped = begin - before;
    for (unsigned int k = 0; k < (end - begin) - kept; ++k) out[row + total + dropped + k] = -1;
    if (t == 0) count[r] = total;
}

// glm_kv_shard_localize_compact's row `r`: the ids whose block this rank
// stores, as local token ids, packed and counted (see that kernel).
template <class Bar>
__device__ __forceinline__ void glm_kv_shard_localize_row(
    const int* __restrict__ in, int* __restrict__ out, unsigned int* __restrict__ counts,
    const unsigned int* __restrict__ block_table, unsigned int r, unsigned int width,
    unsigned int block_size, unsigned int rank, unsigned int world, unsigned int causal_start,
    unsigned int* s_kept, Bar bar) {
    glm_kv_shard_pack_row(in, out, counts, r, width, causal_start, s_kept,
                          [&](unsigned int id, int& local) {
                              const unsigned int block = block_table[id / block_size];
                              if (block % world != rank) return false;
                              local = (int)((block / world) * block_size + id % block_size);
                              return true;
                          },
                          bar);
}

// glm_kv_shard_localize_row with no selection (`in == nullptr`), a block at a
// time, on `nth` threads `tid`: row `r` keeps tokens [0, causal_start + r +
// 1) (at most `width`), whole logical blocks but the last, each stored by one
// rank, so the stable partition is the owned blocks' tokens in order: the
// same IDs, -1 tail and count as the per-token pack, from one table read per
// block in the prefix. `s_blk` holds ceil(width / block_size) + 1 words.
template <class Bar>
__device__ __forceinline__ void glm_kv_shard_localize_causal_row(
    int* __restrict__ out, unsigned int* __restrict__ counts,
    const unsigned int* __restrict__ block_table, unsigned int r, unsigned int width,
    unsigned int block_size, unsigned int rank, unsigned int world, unsigned int causal_start,
    unsigned int* s_blk, unsigned int tid, unsigned int nth, Bar bar) {
    const unsigned int tokens = min(causal_start + r + 1u, width);
    const unsigned int blocks = (tokens + block_size - 1u) / block_size;
    for (unsigned int l = tid; l < blocks; l += nth) {
        const unsigned int size = min(block_size, tokens - l * block_size);
        s_blk[l] = block_table[l] % world == rank ? size : 0u;
    }
    bar();
    if (tid == 0) {
        unsigned int total = 0;
        for (unsigned int l = 0; l < blocks; ++l) {
            const unsigned int size = s_blk[l];
            s_blk[l] = total;
            total += size;
        }
        s_blk[blocks] = total;
    }
    bar();
    const unsigned int total = s_blk[blocks];
    const size_t row = (size_t)r * width;
    for (unsigned int c = tid; c < tokens; c += nth) {
        const unsigned int l = c / block_size, block = block_table[l];
        if (block % world == rank)
            out[row + s_blk[l] + c % block_size] = (int)((block / world) * block_size + c % block_size);
    }
    for (unsigned int p = total + tid; p < width; p += nth) out[row + p] = -1;
    if (tid == 0) counts[r] = total;
}
