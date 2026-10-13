// SPDX-License-Identifier: AGPL-3.0-only
// GLM token-sharded latent storage (ATLAS_GLM_KV_SHARD=1): the device half of
// the ownership rule in crates/spark-runtime/src/kv_cache/latent_shard.rs —
// physical block `b` is stored by rank `b % world` at local slot `b / world`.
// The ranks' ids differ, but the allocator gives the block at logical index
// `l` an id with `b % world == l % world`, so the ranks agree on owners.
// Keep the two in step.
//
// The merge form's attention entry points live here too, instantiated from
// the bodies of the split kernels' files: the shard's (counted split, FP32
// and extra-partition merges) and the canonical form an unsharded TP pair
// runs for the same owners (partition, paired split and paired merge), which
// computes per head exactly what the shard computes across both ranks.
// GLM_KV_SHARD_MODULE drops those files' own entry points;
// scripts/dev/glm_kv_shard_bench.cu, which launches both kinds, includes them
// whole before this file.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#ifndef GLM_KV_SHARD_BODIES_INCLUDED
#define GLM_KV_SHARD_MODULE
#include "glm_sparse_prefill_kv_reuse.cu"
#include "glm_sparse_decode_split_merge.cu"
#endif

// Cache-write slot remap: global slot -> this rank's local slot, or -1 when
// another rank owns the block (every GLM latent writer skips slot < 0).
// Grid: ceil(n / 256), block 256.
extern "C" __global__ void glm_kv_shard_map_slots(
    const long long* __restrict__ slots,
    long long* __restrict__ local,
    unsigned int n,
    unsigned int block_size,
    unsigned int rank,
    unsigned int world) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long slot = slots[i];
    if (slot < 0) {
        local[i] = -1;
        return;
    }
    const long long bs = (long long)block_size;
    const long long block = slot / bs;
    local[i] = (block % (long long)world == (long long)rank)
        ? (block / (long long)world) * bs + slot % bs
        : -1;
}

// Pack one row's selected ids (`in == nullptr`: the causal ids of row `r`,
// tokens [0, causal_start + r + 1)) that `keep(token, &value)` keeps, as
// `value`, to the front of `out`'s row in selected order (a stable
// partition); the rest of the row is -1 and the number kept goes to
// `count[r]`. One CTA of 256 threads per row; thread `t` owns the
// `ceil(width / 256)` consecutive ids from `t * ceil(width / 256)`. `out`
// must not alias `in`. `s_kept` is 256 words of shared memory, reused only
// after a barrier.
#define GLM_KV_SHARD_COMPACT_CHUNK 16u
template <class Keep>
__device__ __forceinline__ void glm_kv_shard_pack_row(
    const int* __restrict__ in, int* __restrict__ out, unsigned int* __restrict__ count,
    unsigned int r, unsigned int width, unsigned int causal_start, unsigned int* s_kept,
    Keep keep) {
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
    __syncthreads();
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

// Selected-ID remap for this rank's partial attention: each row's ids whose
// block this rank stores, as local token ids (read through an identity block
// table over the local pool), packed to the front in selected order, their
// number in `counts[row]`; the attention kernels' `*_split_counted` variants
// walk only that prefix. `in == nullptr` generates causal ids: row `r` keeps
// tokens [0, causal_start + r + 1). Grid: rows, block 256; width <= 256 * 16.
extern "C" __global__ void glm_kv_shard_localize_compact(
    const int* __restrict__ in,
    int* __restrict__ out,
    unsigned int* __restrict__ counts,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int width,
    unsigned int block_size,
    unsigned int rank,
    unsigned int world,
    unsigned int causal_start) {
    __shared__ unsigned int s_kept[256];
    const unsigned int r = blockIdx.x;
    if (r >= rows || blockDim.x != 256u || (width + 255u) / 256u > GLM_KV_SHARD_COMPACT_CHUNK) return;
    glm_kv_shard_pack_row(in, out, counts, r, width, causal_start, s_kept,
                          [&](unsigned int id, int& local) {
                              const unsigned int block = block_table[id / block_size];
                              if (block % world != rank) return false;
                              local = (int)((block / world) * block_size + id % block_size);
                              return true;
                          });
}

// The canonical form of an unsharded TP pair (crates/spark-model/src/layers/
// ops/glm_sparse_canonical.rs): each row's ids split as the shard would store
// them, the ids rank `rank` would store packed to `own` and the rest to
// `peer`, each in selected order with its count, as glm_kv_shard_localize_
// compact packs them on each rank of a shard, but as the global token ids the
// sequence's own table addresses. Token `t` belongs to its logical block's
// residue `(t / block_size) % 2`, which a sharded pool's allocator keeps equal
// to the residue of the physical block storing it (latent_shard.rs).
// Grid: rows, block 256; width <= 256 * 16.
extern "C" __global__ void glm_kv_canonical_partition(
    const int* __restrict__ in,
    int* __restrict__ own,
    unsigned int* __restrict__ own_counts,
    int* __restrict__ peer,
    unsigned int* __restrict__ peer_counts,
    unsigned int rows,
    unsigned int width,
    unsigned int block_size,
    unsigned int rank,
    unsigned int causal_start) {
    __shared__ unsigned int s_kept[256];
    const unsigned int r = blockIdx.x;
    if (r >= rows || blockDim.x != 256u || (width + 255u) / 256u > GLM_KV_SHARD_COMPACT_CHUNK) return;
    auto group = [&](unsigned int residue) {
        return [=](unsigned int id, int& global) {
            global = (int)id;
            return (id / block_size) % 2u == residue;
        };
    };
    glm_kv_shard_pack_row(in, own, own_counts, r, width, causal_start, s_kept, group(rank));
    __syncthreads();  // s_kept is read by every thread above
    glm_kv_shard_pack_row(in, peer, peer_counts, r, width, causal_start, s_kept, group(1u - rank));
}

// Block gather/scatter: dst block `dst_idx[i]` (or `i` when null) receives
// src block `src_idx[i]` (or `i` when null), for i < n. A block is
// `block_vecs` 16-byte vectors. Grid: n (never 0), block 256.
extern "C" __global__ void glm_kv_shard_copy_blocks(
    const uint4* __restrict__ src,
    const unsigned int* __restrict__ src_idx,
    uint4* __restrict__ dst,
    const unsigned int* __restrict__ dst_idx,
    unsigned int n,
    unsigned int block_vecs) {
    const unsigned int i = blockIdx.x;
    if (i >= n) return;
    const size_t s = (size_t)(src_idx != nullptr ? src_idx[i] : i) * block_vecs;
    const size_t d = (size_t)(dst_idx != nullptr ? dst_idx[i] : i) * block_vecs;
    for (unsigned int v = threadIdx.x; v < block_vecs; v += blockDim.x) {
        dst[d + v] = src[s + v];
    }
}

// Counted split variants (the merge form): as `*_split`, over each row's
// first `row_counts[row]` selected IDs.
extern "C" __global__ void glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split_counted(
    GLM_KV_PAD_ARGS, float* __restrict__ part_o, float* __restrict__ part_lse,
    const unsigned int* __restrict__ row_counts) {
    (void)V_cache;
    glm_kv_pad_body<false, true>(GLM_KV_PAD_FORWARD, part_o, part_lse, row_counts, blockIdx.z,
                                 gridDim.z);
}

extern "C" __global__ void glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted(
    GLM_KV_PAD_ARGS, float* __restrict__ part_o, float* __restrict__ part_lse,
    const unsigned int* __restrict__ row_counts) {
    (void)V_cache;
    glm_kv_pad_body<true, true>(GLM_KV_PAD_FORWARD, part_o, part_lse, row_counts, blockIdx.z,
                                gridDim.z);
}

// Two groups' counted splits in one launch: grid z = 2 x splits; CTA
// z < splits is partition z of the first group (`Q` over `token_indices`,
// `row_counts` -> `part_o`, `part_lse`), the others partition z - splits of
// the second (`Q_b` over `ids_b`, `counts_b` -> `part_o_b`, `part_lse_b`).
// Each CTA is the one the counted split launch of its group runs for that
// partition. The canonical form pairs one head group's two token groups; a
// shard rank pairs its own and its peer's heads over the tokens it stores.
#define GLM_KV_PAD_PAIR(NAME, FP8) \
    extern "C" __global__ void NAME( \
        GLM_KV_PAD_ARGS, float* __restrict__ part_o, float* __restrict__ part_lse, \
        const unsigned int* __restrict__ row_counts, const int* __restrict__ ids_b, \
        float* __restrict__ part_o_b, float* __restrict__ part_lse_b, \
        const unsigned int* __restrict__ counts_b, const __nv_bfloat16* Q_b) { \
        (void)V_cache; \
        const unsigned int splits = gridDim.z / 2u; \
        const bool b = blockIdx.z >= splits; \
        glm_kv_pad_body<FP8, true>(b ? Q_b : Q, K_cache, b ? ids_b : token_indices, O, \
                                   block_table, rows, \
                                   num_heads, head_dim, index_width, cache_block_size, \
                                   inv_sqrt_d, b ? part_o_b : part_o, \
                                   b ? part_lse_b : part_lse, b ? counts_b : row_counts, \
                                   b ? blockIdx.z - splits : blockIdx.z, splits); \
    }
GLM_KV_PAD_PAIR(glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split_counted_pair, false)
GLM_KV_PAD_PAIR(glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted_pair, true)
#undef GLM_KV_PAD_PAIR

// FP32-output twin: the normalized merged partial and its LSE, for a second
// exact LSE merge (ATLAS_GLM_KV_SHARD=1 combines both ranks' partials).
extern "C" __global__ void glm_sparse_decode_split_merge_f32(const float* __restrict__ part_o,
                                             const float* __restrict__ part_lse,
                                             float* __restrict__ out_f32,
                                             float* __restrict__ out_lse,
                                             unsigned rows, unsigned heads,
                                             unsigned dim, unsigned splits) {
    if (rows == 0) return;
    GLM_SPLIT_MERGE_SHARED;
    glm_split_merge_body<float, false>(s_w, s_nempty, s_bad, part_o, part_lse, out_f32, out_lse,
                                       rows, heads, dim, splits, nullptr);
}

// The BF16 merge over `splits` local partitions plus one more, `extra`
// (FP32 output then LSE, the layout `_f32` writes), as partition `splits`:
// ATLAS_GLM_KV_SHARD_COMPACT=1 merges the peer's partial where it landed.
extern "C" __global__ void glm_sparse_decode_split_merge_extra(const float* __restrict__ part_o,
                                             const float* __restrict__ part_lse,
                                             __nv_bfloat16* __restrict__ out_bf16,
                                             float* __restrict__ out_lse,
                                             unsigned rows, unsigned heads,
                                             unsigned dim, unsigned splits,
                                             const float* __restrict__ extra) {
    if (rows == 0) return;
    GLM_SPLIT_MERGE_SHARED;
    glm_split_merge_body<__nv_bfloat16, true>(s_w, s_nempty, s_bad, part_o, part_lse, out_bf16,
                                              out_lse, rows, heads, dim, splits, extra);
}

// The canonical form's merges in one launch, per (row, head) what a shard
// runs on the two ranks: the second group's `splits` partitions merged to one
// FP32 partial at `extra` (output then LSEs: glm_sparse_decode_split_merge_f32
// on the rank storing them), then the first group's partitions with that
// partial as partition `splits` to BF16 (glm_sparse_decode_split_merge_extra
// on the head's own rank). With one split the second group's only partition
// is already at `extra` (as a shard sends it unmerged) and is merged as is.
extern "C" __global__ void glm_sparse_decode_split_merge_pair(const float* __restrict__ part_o,
                                             const float* __restrict__ part_lse,
                                             __nv_bfloat16* __restrict__ out_bf16,
                                             float* __restrict__ out_lse,
                                             unsigned rows, unsigned heads,
                                             unsigned dim, unsigned splits,
                                             const float* __restrict__ part_o_b,
                                             const float* __restrict__ part_lse_b,
                                             float* __restrict__ extra) {
    if (rows == 0) return;
    GLM_SPLIT_MERGE_SHARED;
    if (splits > 1) {
        glm_split_merge_body<float, false>(s_w, s_nempty, s_bad, part_o_b, part_lse_b, extra,
                                           extra + (size_t)rows * heads * dim, rows, heads, dim,
                                           splits, nullptr);
        __syncthreads();  // the partial lands for every thread; s_w is reused
    }
    glm_split_merge_body<__nv_bfloat16, true>(s_w, s_nempty, s_bad, part_o, part_lse, out_bf16,
                                              out_lse, rows, heads, dim, splits, extra);
}
