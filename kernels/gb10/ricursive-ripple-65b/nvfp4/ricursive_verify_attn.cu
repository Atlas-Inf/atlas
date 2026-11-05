// SPDX-License-Identifier: AGPL-3.0-only

// DFlash verify attention for one sequence's K rows: the BF16 paged prefill
// kernel (`common/inferspark_prefill_paged.cu`'s body) with its KV extent read
// from the verify step's device metadata instead of host arguments, so it
// can live inside the captured K=gamma verify graph. `dev_kv_len` points at
// the last verify row's `seq_len` (= length + 1, i.e. the KV extent that row
// attends); the rows are the extent's final q_len positions (causal).
//
// Against the per-row decode kernel (one CTA per row x q head, each reading
// its KV head) this reads each KV tile once per q head for all rows.
//
// Grid: (num_q_heads, ceil(q_len / 32), 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>

#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                atlas_cp16(&(smem)[_row][_col], _gm); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME ricursive_verify_attn
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d, const int* __restrict__ dev_kv_len
#define KERNEL_PREAMBLE kv_len = (unsigned int)__ldg(dev_kv_len); q_offset = kv_len - q_len;

#include "../../common/prefill_paged_compute.cuh"
