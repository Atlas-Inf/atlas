// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 prefill attention: Atlas's tensor-core paged prefill body
// (common/prefill_paged_compute.cuh, the per-head-sink build of
// common/inferspark_prefill_paged_sink_h128.cu) with Inkling's relative-position
// bias and per-token sinks switched on. Reference: transformers PR #51050
// `Inkling2RelativePositionBias` and `eager_attention_forward` (Apache-2.0).
//
// inkling2_rel_bias_table: bias[r, h, d] = BF16(sum_j rel[r, h, j] * rel_proj[j, d]) for
// d < rel_extent, the table the attention kernel adds at distance d = query - key (the
// reference adds the bias in BF16). grid (ceil(rel_extent / 256), rows * heads), block 256.

#include <cuda_bf16.h>

#define INK_MAX_D_REL 32

extern "C" __global__ void inkling2_rel_bias_table(const __nv_bfloat16* __restrict__ rel,      // [rows * heads, d_rel]
                                                   const __nv_bfloat16* __restrict__ rel_proj, // [d_rel, rel_extent]
                                                   __nv_bfloat16* __restrict__ bias,           // [rows * heads, rel_extent]
                                                   const unsigned int d_rel, const unsigned int rel_extent
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                                                   , const unsigned int rows_heads
#endif
) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    // AMD (strix-hip) takes this table for every prefill row count (no flash prefill there),
    // so rows * heads passes gridDim.y's 65535: grid-stride over the (row, head) pairs.
    __shared__ float r[INK_MAX_D_REL];
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long rh = blockIdx.y; rh < rows_heads; rh += gridDim.y) {
        __syncthreads();
        if (threadIdx.x < d_rel) r[threadIdx.x] = __bfloat162float(rel[rh * d_rel + threadIdx.x]);
        __syncthreads();
        if (d < rel_extent) {
            float acc = 0.0f;
            for (unsigned int j = 0; j < d_rel; j++) acc += r[j] * __bfloat162float(rel_proj[(unsigned long long)j * rel_extent + d]);
            bias[rh * rel_extent + d] = __float2bfloat16(acc);
        }
    }
#else
    const unsigned long long rh = blockIdx.y;
    __shared__ float r[INK_MAX_D_REL];
    if (threadIdx.x < d_rel) r[threadIdx.x] = __bfloat162float(rel[rh * d_rel + threadIdx.x]);
    __syncthreads();
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= rel_extent) return;
    float acc = 0.0f;
    for (unsigned int j = 0; j < d_rel; j++) acc += r[j] * __bfloat162float(rel_proj[(unsigned long long)j * rel_extent + d]);
    bias[rh * rel_extent + d] = __float2bfloat16(acc);
#endif
}

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

#define ATLAS_ATTN_SINKS
#define INKLING2_ROW_SINKS
#define INKLING2_PREFILL_BIAS
#define HDIM 128
#define KERNEL_NAME inkling2_prefill_attn
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d, const __nv_bfloat16* __restrict__ sinks, \
    const __nv_bfloat16* __restrict__ rel_bias, const unsigned int rel_extent
#define KERNEL_PREAMBLE /* nothing */

#include "../../common/prefill_paged_compute.cuh"
