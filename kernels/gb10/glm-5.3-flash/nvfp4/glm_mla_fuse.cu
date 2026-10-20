// SPDX-License-Identifier: AGPL-3.0-only

// GLM MLA layer front in two launches (ATLAS_GLM_MLA_FUSE=1).
//
// A canonical verify block (1..32 rows) of a GLM sparse-MLA layer runs, from
// the attention norm to q_b, twelve dependent launches: the MXFP8 q_a and
// kv_a projections (each followed by its RMS norm), the latent cache
// assembly, the KV shard's slot map and the fp8_g128 latent write, the
// indexer's key and pool-gate projections (canonical BF16 tensor-core
// tier, eight CTAs each) around the key LayerNorm, the raw tail write, the
// pool finalize and, under a shard, the causal ID localization. Most are a
// few microseconds of work on one to eight CTAs. Here:
//
// * glm_mla_down_fuse: the four projections of the normed input side by side
//   (the index key and gate CTAs first, then q_a, then kv_a), each CTA
//   running its tier's own body (dense_gemv_bf16_tc_cta / mxfp8_gemv_tc_cta,
//   the same NT for a row count), so every output is the same BF16 word. The
//   weights are asked into L2 while the launch waits on its PDL predecessor
//   (atlas_pdl_touch.cuh), as the q_a / kv_a touch twins did.
// * glm_mla_mid_fuse: per row, the q_a RMS norm (rms_norm_vanilla_regs_row on
//   its 1024-thread CTA) and the kv_a RMS norm followed by the fp8_g128
//   latent write of that row at its (shard-local) slot; one CTA running the
//   index cache update of every row (the key LayerNorm and tail write on
//   eight 128-thread groups, then the pool finalize after a CTA barrier); and
//   optionally the merge form's causal localization of each row, a logical
//   block at a time (the IDs are a stable partition, whatever computes it).
//
// Bit identity: every body is the per-row code of the launch it replaces
// (shared through glm_indexer_rows.cuh, glm_kv_shard_pack.cuh and the
// row/CTA entry points of the GEMV and norm files), with that launch's
// thread count where the arithmetic depends on it. The kv_a norm runs at
// 1024 threads instead of 512: a row of at most 2 * blockDim values gives
// each pair its own thread, so the extra warps add exact zeros. The latent
// write's extra threads join its amax reduction's barriers with entries the
// reduction never reads. The module is built with the target's
// --fmad=false, so a body inlined here contracts nothing its own launch
// does not. glm_mla_fuse_test_gpu.rs compares every byte to the twelve
// launches at 1..=32 rows, sharded and not.

#include "mxfp8_gemv.cu"
#include "dense_gemv_bf16_batchm.cu"
#include "glm_rms_norm_regs.cu"
#include "glm_indexer_rows.cuh"
#include "glm_fp8g128.cuh"
#include "glm_kv_shard_pack.cuh"
#include "atlas_pdl_touch.cuh"

// The indexer's key and gate CTAs, then q_a's, then kv_a's (16 output rows
// each). `wk` / `wg` are BF16 [ni, K]; q_a / kv_a MXFP8 values [n, K] and
// E8M0 scales [n, K/32].
template <int NT>
__device__ __forceinline__ void glm_mla_down_fuse_impl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wqa,
    const unsigned char* __restrict__ sqa, const unsigned char* __restrict__ wkva,
    const unsigned char* __restrict__ skva, __nv_bfloat16* __restrict__ keys,
    __nv_bfloat16* __restrict__ gates, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_out, unsigned int M, unsigned int K, unsigned int ni,
    unsigned int nq, unsigned int nkv
) {
    const unsigned int ic = (ni + 15u) / 16u, qc = (nq + 15u) / 16u;
    unsigned int b = blockIdx.x;
    if (b < 2u * ic) {
        const bool gate = b >= ic;
        dense_gemv_bf16_tc_cta<NT>(A, reinterpret_cast<const __nv_bfloat16*>(gate ? wg : wk),
                                   gate ? gates : keys, M, ni, K, ni, K, gate ? b - ic : b);
        return;
    }
    b -= 2u * ic;
    if (b < qc) {
        mxfp8_gemv_tc_cta<NT>(A, wqa, sqa, q_out, M, nq, K, K, nq, b);
        return;
    }
    mxfp8_gemv_tc_cta<NT>(A, wkva, skva, kv_out, M, nkv, K, K, nkv, b - qc);
}

// This CTA's (projection, CTA within it, touching CTAs of it) for
// atlas_pdl_enter_touch_tri: 2 the index pair, 0 q_a, 1 kv_a.
struct GlmMlaDownRole {
    unsigned int sel, cta, ctas;
};

__device__ __forceinline__ GlmMlaDownRole glm_mla_down_role(
    unsigned int ni, unsigned int nq, unsigned int nkv, unsigned int touch_ctas) {
    const unsigned int ic2 = 2u * ((ni + 15u) / 16u), qc = (nq + 15u) / 16u;
    const unsigned int b = blockIdx.x;
    if (b < ic2) return {2u, b, min(touch_ctas, ic2)};
    if (b < ic2 + qc) return {0u, b - ic2, min(touch_ctas, qc)};
    return {1u, b - ic2 - qc, min(touch_ctas, (nkv + 15u) / 16u)};
}

// `touch` = {q_a values, q_a scales, kv_a values, kv_a scales, index key,
// index gate} bytes asked into L2 from each weight's start (0: none), by the
// first `touch_ctas` CTAs of each projection. Grid (2 * ceil(ni/16) +
// ceil(nq/16) + ceil(nkv/16), 1, 1), block (256, 1, 1). `_tc{8,16,32}` take
// up to 8, 16 and 32 rows, the activation tiles (NT) of the canonical tiers
// the four launches ran at that row count.
template <int NT>
__device__ __forceinline__ void glm_mla_down_fuse_entry(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wqa,
    const unsigned char* __restrict__ sqa, const unsigned char* __restrict__ wkva,
    const unsigned char* __restrict__ skva, __nv_bfloat16* __restrict__ keys,
    __nv_bfloat16* __restrict__ gates, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_out, unsigned int M, unsigned int K, unsigned int ni,
    unsigned int nq, unsigned int nkv, unsigned int touch_qa, unsigned int touch_sqa,
    unsigned int touch_kva, unsigned int touch_skva, unsigned int touch_wk,
    unsigned int touch_wg, unsigned int touch_ctas
) {
    atlas_pdl_enter_touch_tri({wqa, touch_qa, 0ull}, {sqa, touch_sqa, 0ull},
                              {wkva, touch_kva, 0ull}, {skva, touch_skva, 0ull},
                              {wk, touch_wk, 0ull}, {wg, touch_wg, 0ull}, 1u,
                              glm_mla_down_role(ni, nq, nkv, touch_ctas));
    glm_mla_down_fuse_impl<NT>(A, wk, wg, wqa, sqa, wkva, skva, keys, gates, q_out, kv_out, M, K,
                               ni, nq, nkv);
}

extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE) glm_mla_down_fuse_tc8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wqa,
    const unsigned char* __restrict__ sqa, const unsigned char* __restrict__ wkva,
    const unsigned char* __restrict__ skva, __nv_bfloat16* __restrict__ keys,
    __nv_bfloat16* __restrict__ gates, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_out, unsigned int M, unsigned int K, unsigned int ni,
    unsigned int nq, unsigned int nkv, unsigned int touch_qa, unsigned int touch_sqa,
    unsigned int touch_kva, unsigned int touch_skva, unsigned int touch_wk,
    unsigned int touch_wg, unsigned int touch_ctas
) {
    glm_mla_down_fuse_entry<1>(A, wk, wg, wqa, sqa, wkva, skva, keys, gates, q_out, kv_out, M, K,
                               ni, nq, nkv, touch_qa, touch_sqa, touch_kva, touch_skva, touch_wk,
                               touch_wg, touch_ctas);
}

extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE) glm_mla_down_fuse_tc16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wqa,
    const unsigned char* __restrict__ sqa, const unsigned char* __restrict__ wkva,
    const unsigned char* __restrict__ skva, __nv_bfloat16* __restrict__ keys,
    __nv_bfloat16* __restrict__ gates, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_out, unsigned int M, unsigned int K, unsigned int ni,
    unsigned int nq, unsigned int nkv, unsigned int touch_qa, unsigned int touch_sqa,
    unsigned int touch_kva, unsigned int touch_skva, unsigned int touch_wk,
    unsigned int touch_wg, unsigned int touch_ctas
) {
    glm_mla_down_fuse_entry<2>(A, wk, wg, wqa, sqa, wkva, skva, keys, gates, q_out, kv_out, M, K,
                               ni, nq, nkv, touch_qa, touch_sqa, touch_kva, touch_skva, touch_wk,
                               touch_wg, touch_ctas);
}

extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE) glm_mla_down_fuse_tc32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wqa,
    const unsigned char* __restrict__ sqa, const unsigned char* __restrict__ wkva,
    const unsigned char* __restrict__ skva, __nv_bfloat16* __restrict__ keys,
    __nv_bfloat16* __restrict__ gates, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_out, unsigned int M, unsigned int K, unsigned int ni,
    unsigned int nq, unsigned int nkv, unsigned int touch_qa, unsigned int touch_sqa,
    unsigned int touch_kva, unsigned int touch_skva, unsigned int touch_wk,
    unsigned int touch_wg, unsigned int touch_ctas
) {
    glm_mla_down_fuse_entry<4>(A, wk, wg, wqa, sqa, wkva, skva, keys, gates, q_out, kv_out, M, K,
                               ni, nq, nkv, touch_qa, touch_sqa, touch_kva, touch_skva, touch_wk,
                               touch_wg, touch_ctas);
}

#define MLA_MID_THREADS 1024u
#define MLA_MID_INDEX_GROUP 128u

// The index cache update of rows [0, rows) on one CTA: each 128-thread group
// g LayerNorms and tail-writes rows g, g + 8, ...; after a CTA barrier the
// groups finalize the pools those rows end. The original launches ran each
// row on a 128-thread CTA (glm_index_layernorm_bf16, _tail_write_bf16,
// _kpool_finalize_bf16), in that order.
__device__ __forceinline__ void glm_mla_mid_index(
    __nv_bfloat16* keys, const __nv_bfloat16* gates, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* k_norm_b, float index_eps, unsigned int dim, __nv_bfloat16* tail,
    const unsigned int* tail_map, unsigned long long tail_stride, const __nv_bfloat16* ape,
    __nv_bfloat16* index_cache, unsigned long long index_stride, unsigned int pool_size,
    const long long* slots, unsigned int block_size, unsigned int rows, float* s_reduce
) {
    const unsigned int groups = MLA_MID_THREADS / MLA_MID_INDEX_GROUP;
    const unsigned int g = threadIdx.x / MLA_MID_INDEX_GROUP;
    const unsigned int lt = threadIdx.x % MLA_MID_INDEX_GROUP;
    const GlmNamedBarrier bar{1u + g, MLA_MID_INDEX_GROUP};
    for (unsigned int r = g; r < rows; r += groups) {
        glm_index_layernorm_row(keys + (unsigned long long)r * dim, k_norm_w, k_norm_b, dim,
                                index_eps, s_reduce + g * MLA_MID_INDEX_GROUP, lt,
                                MLA_MID_INDEX_GROUP, bar);
        // The row's normalized key from every lane; the group's reduce
        // scratch is free for its next row only past this barrier too.
        bar();
        glm_index_tail_write_row(keys, gates, tail, tail_map, slots, r, block_size, dim,
                                 tail_stride, lt, MLA_MID_INDEX_GROUP);
    }
    __syncthreads();  // every raw tail of the block before any pool reads it
    for (unsigned int r = g; r < rows; r += groups) {
        glm_index_kpool_finalize_row(tail, tail_map, ape, index_cache, slots, r, block_size,
                                     pool_size, dim, tail_stride, index_stride, lt,
                                     MLA_MID_INDEX_GROUP);
    }
}

// CTA 0: the index cache update; CTAs [1, 1 + rows): row r's kv_a norm (in
// place) and its fp8_g128 latent at the shard-local slot (`world` 1: the
// slot itself; glm_kv_shard_map_slots' rule otherwise); [1 + rows,
// 1 + 2 rows): row r's q_a norm (in place); [1 + 2 rows, 1 + 2 rows +
// loc_rows): row r's causal localization (glm_kv_shard_localize_compact's
// IDs with no selection, a logical block at a time; `loc_width` <=
// 1023 * block_size). Block (1024, 1, 1).
extern "C" __global__ void __launch_bounds__(MLA_MID_THREADS) glm_mla_mid_fuse(
    __nv_bfloat16* __restrict__ q_latent, const __nv_bfloat16* __restrict__ q_norm_w,
    __nv_bfloat16* __restrict__ kv_latent, const __nv_bfloat16* __restrict__ kv_norm_w,
    unsigned int q_dim, unsigned int kv_dim, float eps, unsigned char* __restrict__ latent_cache,
    const long long* __restrict__ slots, unsigned int block_size, unsigned int rank,
    unsigned int world, __nv_bfloat16* keys, const __nv_bfloat16* __restrict__ gates,
    const __nv_bfloat16* __restrict__ k_norm_w, const __nv_bfloat16* __restrict__ k_norm_b,
    float index_eps, unsigned int index_dim, __nv_bfloat16* tail,
    const unsigned int* __restrict__ tail_map, unsigned long long tail_stride,
    const __nv_bfloat16* __restrict__ ape, __nv_bfloat16* __restrict__ index_cache,
    unsigned long long index_stride, unsigned int pool_size,
    const unsigned int* __restrict__ block_table, int* __restrict__ loc_ids,
    unsigned int* __restrict__ loc_counts, unsigned int loc_width, unsigned int causal_start,
    unsigned int rows, unsigned int loc_rows
) {
    atlas_pdl_enter();
    __shared__ float s_buf[MLA_MID_THREADS];
    unsigned int b = blockIdx.x;
    if (b == 0u) {
        glm_mla_mid_index(keys, gates, k_norm_w, k_norm_b, index_eps, index_dim, tail, tail_map,
                          tail_stride, ape, index_cache, index_stride, pool_size, slots,
                          block_size, rows, s_buf);
        return;
    }
    b -= 1u;
    if (b < rows) {
        rms_norm_vanilla_regs_row(kv_latent, kv_norm_w, kv_latent, kv_dim, eps, b);
        __syncthreads();  // the normalized row from every thread
        const long long slot = slots[b];
        if (slot < 0) return;
        const long long bs = (long long)block_size, block = slot / bs;
        if (block % (long long)world != (long long)rank) return;
        const long long local = (block / (long long)world) * bs + slot % bs;
        glm_latent_cache_write_fp8g128_row<true>(kv_latent + (unsigned long long)b * kv_dim,
                                                 latent_cache, local, block_size, s_buf);
        return;
    }
    b -= rows;
    if (b < rows) {
        rms_norm_vanilla_regs_row(q_latent, q_norm_w, q_latent, q_dim, eps, b);
        return;
    }
    b -= rows;
    if (b >= loc_rows) return;
    glm_kv_shard_localize_causal_row(loc_ids, loc_counts, block_table, b, loc_width, block_size,
                                     rank, world, causal_start,
                                     reinterpret_cast<unsigned int*>(s_buf), threadIdx.x,
                                     MLA_MID_THREADS, GlmCtaBarrier{});
}
