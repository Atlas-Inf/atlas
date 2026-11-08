// SPDX-License-Identifier: AGPL-3.0-only

// MiMo-V2.6-Flash paged prefill flash attention — tensor-core (mma.sync)
// variant for the 192/128 head geometry.
//
// Fork of kernels/gb10/common/inferspark_prefill_paged.cu built on
// prefill_paged_compute_mimo.cuh (a dim-split fork of
// prefill_paged_compute.cuh): Q/K head dim 192, V head dim 128 (output 128).
// Replaces the scalar mimo_prefill_attn for chunked (paged) prefill, which
// was CUDA-core bound and made long-context prefill quadratic-slow.
//
// K/V paged layout (kv_layer_dims = 192 for both):
//   [block, block_size, nkv, 192] BF16; V uses only the first 128 dims of
//   each 192-wide head slot. SWA layers with eviction address KV through
//   `swa_map` (logical block -> ring slot), full-attention layers pass null
//   and use block_table directly. `sinks` is the per-head attention-sink
//   bias ([nq]) on SWA layers, null on full-attention layers.
//
// Grid: (num_q_heads, ceil(q_len/BR=32), 1)  Block: (128, 1, 1).

#include <cuda_bf16.h>

#define HDIM 192
#define ATLAS_MIMO_NO_Q64
#define MIMO_QK_DIM 192
#define MIMO_V_DIM 128

// K tile: BC(32) rows x 192 cols, 16-byte cp.async chunks of 8 bf16.
#define MIMO_TC_CPR_K (MIMO_QK_DIM / 8)
#define MIMO_TC_CHUNKS_K (BC * MIMO_TC_CPR_K)
#define LOAD_K_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned long long _ps = \
            (unsigned long long)cache_block_size * num_kv_heads * MIMO_QK_DIM; \
        const unsigned long long _rs = \
            (unsigned long long)num_kv_heads * MIMO_QK_DIM; \
        for (unsigned int _i = (t); _i < MIMO_TC_CHUNKS_K; _i += (stride)) { \
            unsigned int _row = _i / MIMO_TC_CPR_K, _col = (_i % MIMO_TC_CPR_K) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _bt = (unsigned int)(bt)[_lb]; \
                unsigned int _phys = swa_map ? (unsigned int)swa_map[_bt] : _bt; \
                const void* _gm = (const void*)((cache) + _phys * _ps + _bo * _rs + \
                    (kvh) * MIMO_QK_DIM + _col); \
                atlas_cp16(&(smem)[_row][_col], _gm); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

// V tile: same paged layout (192-wide head slots), only the first 128 cols.
#define MIMO_TC_CPR_V (MIMO_V_DIM / 8)
#define MIMO_TC_CHUNKS_V (BC * MIMO_TC_CPR_V)
#define LOAD_V_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned long long _ps = \
            (unsigned long long)cache_block_size * num_kv_heads * MIMO_QK_DIM; \
        const unsigned long long _rs = \
            (unsigned long long)num_kv_heads * MIMO_QK_DIM; \
        for (unsigned int _i = (t); _i < MIMO_TC_CHUNKS_V; _i += (stride)) { \
            unsigned int _row = _i / MIMO_TC_CPR_V, _col = (_i % MIMO_TC_CPR_V) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _bt = (unsigned int)(bt)[_lb]; \
                unsigned int _phys = swa_map ? (unsigned int)swa_map[_bt] : _bt; \
                const void* _gm = (const void*)((cache) + _phys * _ps + _bo * _rs + \
                    (kvh) * MIMO_QK_DIM + _col); \
                atlas_cp16(&(smem)[_row][_col], _gm); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define ATLAS_ATTN_SINKS
#define KERNEL_NAME mimo_prefill_tc
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d, const float out_scale, const __nv_bfloat16* __restrict__ sinks
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute_mimo.cuh"
