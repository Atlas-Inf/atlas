// SPDX-License-Identifier: AGPL-3.0-only
// ling_mla_prefill over one key tile of an expanded latent cache, emitting
// FP32 output and each row's softmax (max, sum) for ling_prompt_tile_merge.
#include "ling_cache_layout.cuh"
#define HDIM 192
#define ATLAS_PREFILL_GROUP_QUERY_TILES
#define PREFILL_PARTIAL_LSE
#define inferspark_prefill_paged ling_mla_prefill_part
#define KERNEL_EXTRA_PARAMS_OVERRIDE , const float inv_sqrt_d, float* __restrict__ prefill_ml
#include "../../common/inferspark_prefill_paged.cu"
