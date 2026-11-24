// SPDX-License-Identifier: AGPL-3.0-only
// Ling caches 192 K channels and pads each 128-channel V head to the same
// width. Reuse Atlas's causal, BF16 paged flash-attention tiles at that width.
#define HDIM 192
#define ATLAS_PREFILL_GROUP_QUERY_TILES
#define inferspark_prefill_paged ling_mla_prefill
#include "../../common/inferspark_prefill_paged.cu"
