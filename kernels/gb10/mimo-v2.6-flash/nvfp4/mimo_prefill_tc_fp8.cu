// SPDX-License-Identifier: AGPL-3.0-only

// MiMo-V2.6-Flash paged prefill flash attention — tensor-core (mma.sync)
// variant for the 192/128 head geometry, FP8 E4M3 KV cache.
//
// FP8 counterpart of mimo_prefill_tc.cu: reads the FP8 paged K/V written by
// mimo_qkv_rope_cache (same 192-stride element layout as BF16, one byte per
// element) and runs the tensor-core prefill with K/V kept as raw E4M3 bytes
// in shared memory (ATLAS_ATTN_FP8_SMEM), dequantized in-register before
// each MMA via prefill_paged_compute_mimo.cuh's fp8x2_to_*_bits. This
// restores tensor-core prefill speed under --kv-cache-dtype fp8 (the scalar
// mimo_prefill_attn fallback made cold long-context ingest ~3.4x slower).
//
// K/V paged layout (kv_layer_dims = 192 for both, FP8 = 1 byte/elem):
//   [block, block_size, nkv, 192]; V uses only the first 128 dims of each
//   192-wide head slot. SWA layers with eviction address KV through `swa_map`;
//   full-attention layers pass null. `sinks` is the per-head attention-sink
//   bias ([nq]) on SWA layers, null on full-attention layers.
//
// Grid: (num_q_heads, ceil(q_len/BR=32), 1)  Block: (128, 1, 1).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

#define HDIM 192
#define ATLAS_MIMO_NO_Q64
#define MIMO_QK_DIM 192
#define MIMO_V_DIM 128

// FP8-smem + cp.async pipelining: K/V kept in shared memory as raw E4M3 bytes
// (1 B) instead of dequantized BF16 (2 B); dequant deferred to in-register MMA
// read. PAD_KV=16 gives a 16-byte-aligned FP8 row stride (HDIM_PAD=208,
// HDIM_V_PAD=144, both multiples of 16) for the 16-byte atlas_cp16 / uint4
// stores — the default PAD_KV=8 would leave odd rows misaligned.
#define ATLAS_ATTN_FP8_SMEM
#ifdef ATLAS_ATTN_FP8_SMEM
#define PAD_KV 16
#endif

// K tile (FP8): BC(32) rows x 192 cols. One atlas_cp16 = 16 bytes = 16 FP8
// elements, so 12 chunks/row (192/16), 384 total. cache_stride is in FP8
// elements (1 byte each); the paged layout keeps the 192-stride head slots.
#define MIMO_TC_CPR_K (MIMO_QK_DIM / 16)
#define MIMO_TC_CHUNKS_K (BC * MIMO_TC_CPR_K)
#define LOAD_K_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned long long _ps = \
            (unsigned long long)cache_block_size * num_kv_heads * MIMO_QK_DIM; \
        const unsigned long long _rs = \
            (unsigned long long)num_kv_heads * MIMO_QK_DIM; \
        for (unsigned int _i = (t); _i < MIMO_TC_CHUNKS_K; _i += (stride)) { \
            unsigned int _row = _i / MIMO_TC_CPR_K, _col = (_i % MIMO_TC_CPR_K) * 16; \
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

// V tile (FP8): same paged layout (192-wide head slots), only the first 128
// dims (128 FP8 elements = 8 chunks of 16 bytes/row, 256 total).
#define MIMO_TC_CPR_V (MIMO_V_DIM / 16)
#define MIMO_TC_CHUNKS_V (BC * MIMO_TC_CPR_V)
#define LOAD_V_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned long long _ps = \
            (unsigned long long)cache_block_size * num_kv_heads * MIMO_QK_DIM; \
        const unsigned long long _rs = \
            (unsigned long long)num_kv_heads * MIMO_QK_DIM; \
        for (unsigned int _i = (t); _i < MIMO_TC_CHUNKS_V; _i += (stride)) { \
            unsigned int _row = _i / MIMO_TC_CPR_V, _col = (_i % MIMO_TC_CPR_V) * 16; \
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
#define KERNEL_NAME mimo_prefill_tc_fp8
#define K_CACHE_TYPE const __nv_fp8_storage_t* __restrict__
#define V_CACHE_TYPE const __nv_fp8_storage_t* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d, const float k_scale, const float v_scale, const float out_scale, const __nv_bfloat16* __restrict__ sinks
#define KERNEL_PREAMBLE /* nothing — FP8-SMEM defers dequant to in-register MMA reads */

#include "prefill_paged_compute_mimo.cuh"
