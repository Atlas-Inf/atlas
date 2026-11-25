// SPDX-License-Identifier: AGPL-3.0-only
// Keep the logical attention width while reading the shared RoPE tail once per token.
#include "ling_cache_layout.cuh"
#define HDIM 192
#define ATLAS_PREFILL_GROUP_QUERY_TILES
#define inferspark_prefill_paged ling_mla_prefill
#include "../../common/inferspark_prefill_paged.cu"

